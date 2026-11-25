#!/usr/bin/env python3
"""
vietasr_mini.py  -  compact, plain-PyTorch re-implementation of the VietASR recipe
(Zhuo et al., Interspeech 2025, arXiv:2505.21527)

Pipeline (same 4 stages as the paper, see Fig.1 of the paper):
  Stage 1  initial ASR on a SMALL labelled set                       -> `train-asr`
  Stage 2  k-means on the final-layer encoder features -> labels      -> `kmeans`
  Stage 3  HuBERT-style masked prediction on UNLABELLED audio (CE)    -> `pretrain`
  Stage 4  fine-tune the pre-trained encoder on the small labelled set -> `finetune`
  Iteration n+1 restarts at Stage 2 using the fine-tuned encoder of iteration n.

Differences w.r.t. the original code base (icefall/k2 + Zipformer + pruned RNN-T), on purpose:
  * encoder: Conformer (pre-norm, macaron FFN, relative-free abs. sin PE), ~14M params instead of
    Zipformer-base 68M  -> pure PyTorch, no k2/icefall install on Kaggle
  * fine-tuning loss: CTC instead of pruned RNN-T
  * AdamW + time-based cosine schedule instead of ScaledAdam + Eden
  * data: ~50h labelled / ~150h unlabelled from ViCocktail instead of 50h / 70,000h YouTube
v2 additions (all aimed at making the weak/first model converge and the demo model accurate):
  * per-clip loudness normalisation inside `fbank` (recordings of any volume look alike)
  * CTC training aids: intermediate-CTC loss, EMA weights, length curriculum in epoch 0,
    automatic retry with a lower LR when the model collapses (dev WER > --max-ok-wer)
  * k-means guard: refuse a collapsed codebook (perplexity < --min-perplexity) instead of wasting hours
  * `full` store (all labelled train hours) for a *bonus* demo fine-tune, and `samples` command that exports demo clips
Everything that defines VietASR (Fbank front-end, masking on Fbank, CE loss on cluster ids,
ASR-biased codebook from a weak supervised model, multi-iteration) is kept.

CLI:  python vietasr_mini.py <prep|train-asr|finetune|kmeans|pretrain|eval|export|samples> ...
"""
from __future__ import annotations

import argparse
import collections
import copy
import io
import json
import math
import random
import re
import time
import unicodedata
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_MELS = 80
REPO = "nguyenvulebinh/ViCocktail"
TEST_SETS = ["test"] + [f"test_snr_{s}_interferer_{k}" for s in ("n5", "0", "5", "10") for k in (1, 2)]


# ----------------------------------------------------------------------------------------------
# text / audio utilities
# ----------------------------------------------------------------------------------------------
_NON_WORD = re.compile(r"[^\w\s]|_", re.UNICODE)


def norm_text(s: str) -> str:
    """lower-case, NFC, strip punctuation -> space separated Vietnamese syllables."""
    s = unicodedata.normalize("NFC", s).lower()
    s = _NON_WORD.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", "ignore")
    if isinstance(x, (list, tuple)):
        return bytes(x).decode("utf-8", "ignore")
    return str(x)


def to_bytes(x) -> bytes:
    if isinstance(x, dict):
        x = x.get("bytes", None)
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, (list, tuple)):
        return bytes(x)
    raise TypeError(f"cannot turn {type(x)} into bytes (is the 'video' column decoded by `datasets`?)")


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def decode_audio(b: bytes, sr: int = SR) -> np.ndarray:
    """Decode the audio track of an mp4/any container held in memory -> float32 mono @ sr (PyAV)."""
    import av

    out = []
    with av.open(io.BytesIO(b)) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        rs = av.AudioResampler(format="s16", layout="mono", rate=sr)
        for fr in c.decode(audio=0):
            for r in _as_list(rs.resample(fr)):
                out.append(r.to_ndarray().reshape(-1))
        for r in _as_list(rs.resample(None)):
            out.append(r.to_ndarray().reshape(-1))
    if not out:
        return np.zeros(0, np.float32)
    return np.concatenate(out).astype(np.float32) / 32768.0


def load_audio_file(path: str, sr: int = SR) -> np.ndarray:
    """Any audio/video file -> float32 mono @ sr using the ffmpeg binary."""
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:  # pragma: no cover
            raise RuntimeError("ffmpeg not found (install ffmpeg or `pip install imageio-ffmpeg`)") from e
    cmd = [exe, "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode("utf-8", "ignore")[-400:])
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


_MEL = {}


def _mel_filterbank(n_fft=512, n_mels=N_MELS, sr=SR, fmin=20.0, fmax=8000.0) -> torch.Tensor:
    key = (n_fft, n_mels, sr, fmin, fmax)
    if key not in _MEL:
        hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
        bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
        fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
        for i in range(n_mels):
            lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
            fb[i] = np.maximum(0.0, np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)))
        _MEL[key] = torch.from_numpy(fb)
    return _MEL[key]


def level_norm(w, target=0.06):
    """Scale a waveform so that its *speech* level (mean power of the louder half of the 25 ms frames)
    has RMS = target.  Makes quiet phone recordings and loud studio audio look alike to the model."""
    n = (w.numel() // 400) * 400
    if n < 400:
        return w
    ms = (w[:n].reshape(-1, 400) ** 2).mean(1)
    loud = ms[ms >= ms.median()]
    rms = float(loud.mean().sqrt())
    if rms < 1e-5:
        return w
    return w * min(max(target / rms, 0.05), 100.0)


def fbank(wave, level=True) -> torch.Tensor:
    """80-dim log-mel Fbank, 25 ms window / 10 ms shift  ->  [T, 80] float32 (T = 1 + (L-400)//160)."""
    w = torch.as_tensor(wave, dtype=torch.float32).reshape(-1)
    if level:
        w = level_norm(w)
    if w.numel() < 400:
        w = F.pad(w, (0, 400 - w.numel()))
    win = torch.hann_window(400)
    spec = torch.stft(w, n_fft=512, hop_length=160, win_length=400, window=win, center=False, return_complex=True)
    mel = _mel_filterbank() @ (spec.abs() ** 2)
    return torch.log(mel.clamp(min=1e-7)).transpose(0, 1).contiguous()


def sub_len(n):
    """#frames after the 4x Conv2d subsampling (== avg_pool1d(kernel=7, stride=4) length)."""
    return ((n - 1) // 2 - 1) // 2


# ----------------------------------------------------------------------------------------------
# WER
# ----------------------------------------------------------------------------------------------
def edit_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer_corpus(refs, hyps):
    errs = n = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = r.split(), h.split()
        errs += edit_distance(rw, hw)
        n += len(rw)
    return 100.0 * errs / max(n, 1), n


# ----------------------------------------------------------------------------------------------
# feature stores (flat fp16 memmap + json index)
# ----------------------------------------------------------------------------------------------
class StoreWriter:
    def __init__(self, prefix, keep_wave=False):
        self.prefix = str(prefix)
        Path(self.prefix).parent.mkdir(parents=True, exist_ok=True)
        self.fb = open(self.prefix + ".bin", "wb")
        self.wf = open(self.prefix + ".wav16", "wb") if keep_wave else None
        self.items, self.off, self.woff = [], 0, 0

    def add(self, uid, text, feats: np.ndarray, wave: np.ndarray | None = None):
        feats = np.ascontiguousarray(feats, dtype=np.float16)
        it = {"id": uid, "off": self.off, "n": int(feats.shape[0])}
        if text is not None:
            it["text"] = text
        self.fb.write(feats.tobytes())
        self.off += feats.shape[0]
        if self.wf is not None and wave is not None:
            wave = np.ascontiguousarray(wave, dtype=np.int16)
            self.wf.write(wave.tobytes())
            it["woff"], it["wn"] = self.woff, int(wave.size)
            self.woff += wave.size
        self.items.append(it)

    @property
    def hours(self):
        return self.off / 100.0 / 3600.0

    def close(self):
        self.fb.close()
        if self.wf is not None:
            self.wf.close()
        json.dump({"items": self.items, "n_mels": N_MELS}, open(self.prefix + ".json", "w"), ensure_ascii=False)


class Store:
    def __init__(self, prefix):
        self.prefix = str(prefix)
        self.items = json.load(open(self.prefix + ".json"))["items"]
        self.lens = [it["n"] for it in self.items]
        self._mm = None
        self._wm = None

    def __len__(self):
        return len(self.items)

    @property
    def hours(self):
        return sum(self.lens) / 100.0 / 3600.0

    def get(self, i) -> np.ndarray:
        if self._mm is None:
            self._mm = np.memmap(self.prefix + ".bin", dtype=np.float16, mode="r").reshape(-1, N_MELS)
        it = self.items[i]
        return np.asarray(self._mm[it["off"]: it["off"] + it["n"]])

    def wave(self, i) -> np.ndarray:
        if self._wm is None:
            self._wm = np.memmap(self.prefix + ".wav16", dtype=np.int16, mode="r")
        it = self.items[i]
        return np.asarray(self._wm[it["woff"]: it["woff"] + it["wn"]]).astype(np.float32) / 32768.0

    def texts(self):
        return [it.get("text", "") for it in self.items]


class LabelStore:
    """k-means cluster ids per output frame, aligned with a Store (same item order)."""

    def __init__(self, prefix):
        self.prefix = str(prefix)
        m = json.load(open(self.prefix + ".json"))
        self.offs, self.lens, self.k = m["offs"], m["lens"], m["k"]
        self._mm = None

    def get(self, i):
        if self._mm is None:
            self._mm = np.memmap(self.prefix + ".bin", dtype=np.int16, mode="r")
        return np.asarray(self._mm[self.offs[i]: self.offs[i] + self.lens[i]]).astype(np.int64)


def load_stats(work):
    s = json.load(open(Path(work) / "stats.json"))
    return np.array(s["mean"], np.float32), np.array(s["std"], np.float32)


def make_batches(lens, max_frames, shuffle, seed, min_len=0, max_len=10 ** 9, ok=None, curriculum=False):
    """Length-bucketed batches.  curriculum=True: shortest utterances first, batches in that order
    (an 'epoch 0' warm-up that makes CTC from scratch escape the all-blank plateau)."""
    idx = [i for i, l in enumerate(lens) if min_len <= l <= max_len and (ok is None or ok[i])]
    rng = random.Random(seed)
    if curriculum:
        shuffle = False
    jitter = {i: lens[i] * (1.0 + (rng.uniform(-0.1, 0.1) if shuffle else 0.0)) for i in idx}
    idx.sort(key=lambda i: jitter[i])
    batches, cur, mx = [], [], 0
    for i in idx:
        L = lens[i]
        if cur and max(mx, L) * (len(cur) + 1) > max_frames:
            batches.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, L)
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


class BatchDS(torch.utils.data.Dataset):
    """Each item is a fully collated batch (use DataLoader(batch_size=None))."""

    def __init__(self, store, batches, stats, tokens=None, labels=None):
        self.store, self.batches, self.tokens, self.labels = store, batches, tokens, labels
        self.mean, self.std = stats

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, bi):
        idxs = self.batches[bi]
        arrs = [(self.store.get(i).astype(np.float32) - self.mean) / self.std for i in idxs]
        lens = np.array([a.shape[0] for a in arrs], np.int64)
        x = np.zeros((len(arrs), int(lens.max()), N_MELS), np.float16)
        for k, a in enumerate(arrs):
            x[k, : a.shape[0]] = a
        out = {"x": torch.from_numpy(x), "lens": torch.from_numpy(lens), "idx": torch.tensor(idxs)}
        if self.tokens is not None:
            ys = [self.tokens[i] for i in idxs]
            out["y"] = torch.from_numpy(np.concatenate(ys).astype(np.int64))
            out["ylens"] = torch.tensor([len(y) for y in ys], dtype=torch.int64)
        if self.labels is not None:
            zs = [self.labels.get(i) for i in idxs]
            tp = int(sub_len(lens.max()))
            z = np.full((len(zs), tp), -100, np.int64)
            for k, a in enumerate(zs):
                z[k, : a.shape[0]] = a
            out["z"] = torch.from_numpy(z)
        return out


def make_loader(ds, workers):
    return torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), prefetch_factor=4 if workers > 0 else None,
    )


# ----------------------------------------------------------------------------------------------
# model: Fbank -> Conv2d/4 -> N x Conformer block ;  heads: cls_head (SSL) / ctc_head (ASR)
# ----------------------------------------------------------------------------------------------
class Conv2dSub(nn.Module):
    def __init__(self, idim, odim, ch=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, ch, 3, 2), nn.ReLU(), nn.Conv2d(ch, ch, 3, 2), nn.ReLU())
        self.out = nn.Linear(ch * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        b, c, t, f = x.shape
        return self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))


class FFN(nn.Module):
    def __init__(self, d, ff, drop):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff, d), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.drop = h, drop
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, pad):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=(~pad)[:, None, None, :],
                                           dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class ConvMod(nn.Module):
    def __init__(self, d, k, drop):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        y = self.ln(x).masked_fill(pad.unsqueeze(-1), 0.0).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = self.dw(y).transpose(1, 2)
        y = F.silu(self.ln2(y)).transpose(1, 2)
        return self.drop(self.pw2(y).transpose(1, 2))


class Block(nn.Module):
    def __init__(self, d, h, ff, k, drop):
        super().__init__()
        self.ff1, self.ff2 = FFN(d, ff, drop), FFN(d, ff, drop)
        self.ln_att, self.att = nn.LayerNorm(d), MHSA(d, h, drop)
        self.conv, self.ln, self.drop = ConvMod(d, k, drop), nn.LayerNorm(d), nn.Dropout(drop)

    def forward(self, x, pad):
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.att(self.ln_att(x), pad))
        x = x + self.conv(x, pad)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class AcousticModel(nn.Module):
    def __init__(self, d=256, heads=4, ff=1024, layers=8, kernel=31, drop=0.1, n_cls=500, vocab=256, max_len=4000):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, ff=ff, layers=layers, kernel=kernel, drop=drop, n_cls=n_cls, vocab=vocab, max_len=max_len)
        self.d = d
        self.sub = Conv2dSub(N_MELS, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, kernel, drop) for _ in range(layers)])
        self.mask_emb = nn.Parameter(torch.zeros(N_MELS))
        self.drop = nn.Dropout(drop)
        self.cls_head = nn.Linear(d, n_cls)
        self.ctc_head = nn.Linear(d, vocab)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def encode(self, x, lens, mask=None, layer=None, collect=None):
        """x [B,T,80] (normalised), lens [B]; mask [B,T] bool (True = replace frame by mask_emb).
        collect=i : remember the output of block i in self.mid (used by the intermediate-CTC loss)."""
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_emb.to(x.dtype), x)
        h = self.sub(x)
        hl = sub_len(lens)
        T = h.size(1)
        h = self.drop(h * math.sqrt(self.d) + self.pe[:T].to(h.dtype))
        pad = torch.arange(T, device=h.device)[None, :] >= hl[:, None]
        self.mid = None
        for i, b in enumerate(self.blocks):
            h = b(h, pad)
            if collect is not None and i == collect:
                self.mid = h
            if layer is not None and i == layer:
                break
        return h, hl


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def save_ckpt(path, model, **extra):
    torch.save({"model": model.state_dict(), "cfg": model.cfg, **extra}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = AcousticModel(**ck["cfg"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


# ----------------------------------------------------------------------------------------------
# augmentation / masking
# ----------------------------------------------------------------------------------------------
def spec_augment(x, lens, n_freq=2, f_max=27, n_time=3, t_max=40):
    B, T, Fd = x.shape
    dev = x.device
    x = x.clone()
    ar_f = torch.arange(Fd, device=dev)[None, None, :]
    for _ in range(n_freq):
        w = torch.randint(0, f_max + 1, (B, 1, 1), device=dev)
        s = (torch.rand(B, 1, 1, device=dev) * (Fd - w)).long()
        x = x.masked_fill((ar_f >= s) & (ar_f < s + w), 0.0)
    ar_t = torch.arange(T, device=dev)[None, :, None]
    for _ in range(n_time):
        wmax = torch.clamp((lens.float() * 0.1), max=t_max).long()
        w = (torch.rand(B, device=dev) * (wmax + 1)).long()
        s = (torch.rand(B, device=dev) * (lens - w).clamp(min=1)).long()
        x = x.masked_fill((ar_t >= s[:, None, None]) & (ar_t < (s + w)[:, None, None]), 0.0)
    return x


def make_mask(lens, T, prob, span):
    """HuBERT-style span mask applied on Fbank frames: each frame starts a span w.p. `prob`."""
    dev = lens.device
    valid = torch.arange(T, device=dev)[None, :] < lens[:, None]
    starts = ((torch.rand(len(lens), T, device=dev) < prob) & valid).float()[:, None]
    m = F.max_pool1d(F.pad(starts, (span - 1, 0)), kernel_size=span, stride=1)[:, 0] > 0
    return m & valid


# ----------------------------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------------------------
def ctc_collapse(ids):
    out, prev = [], -1
    for t in ids:
        if t != prev and t != 0:
            out.append(int(t))
        prev = t
    return out


@torch.no_grad()
def decode_store(model, store, sp, stats, idx=None, max_frames=300000, workers=0, return_time=False):
    dev = next(model.parameters()).device
    model.eval()
    idx = list(range(len(store))) if idx is None else list(idx)
    lens = [store.lens[i] for i in idx]
    sub_batches = make_batches(lens, max_frames, False, 0)
    ds = BatchDS(store, [[idx[j] for j in b] for b in sub_batches], stats)
    hyps, tsum = {}, 0.0
    for batch in make_loader(ds, workers):
        x, l = batch["x"].to(dev).float(), batch["lens"].to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            h, hl = model.encode(x, l)
            lg = model.ctc_head(h)
        am = lg.argmax(-1).cpu().numpy()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tsum += time.time() - t0
        hl = hl.cpu().numpy()
        for k, i in enumerate(batch["idx"].tolist()):
            hyps[i] = norm_text(sp.decode(ctc_collapse(am[k, : hl[k]])))
    out = [hyps[i] for i in idx]
    return (out, tsum) if return_time else out


# ----------------------------------------------------------------------------------------------
# training loop (time-budgeted, resumable)
# ----------------------------------------------------------------------------------------------
def lr_at(frac, peak, warm=0.08, floor=0.02):
    if frac < warm:
        return peak * max(frac / warm, 0.02)
    p = (frac - warm) / (1 - warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def _swap(model, vals):
    """put `vals` (list of tensors) into the model parameters, return the previous values"""
    plist = list(model.parameters())
    old = [p.detach().clone() for p in plist]
    with torch.no_grad():
        for p, v in zip(plist, vals):
            p.copy_(v)
    return old


def train_loop(model, loader_fn, loss_fn, args, outdir, peak_lr, warm=0.08, freeze_frac=0.0, dev_eval=None, ema_decay=0.0):
    """Time-budgeted, resumable training.  ema_decay>0 keeps an exponential moving average of the weights;
    dev evaluations and the final model use the averaged weights (usually a few % relative WER better)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    plist = list(model.parameters())
    ema = [p.detach().clone() for p in plist] if ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    budget = args.minutes * 60.0
    elapsed0, step, epoch = 0.0, 0, 0
    last = outdir / "last.pt"
    if args.force and last.exists():
        last.unlink()
    resumed = last.exists()
    if resumed:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema is not None and ck.get("ema") is not None:
            ema = [e.to(dev) for e in ck["ema"]]
        elapsed0, step, epoch = ck["elapsed"], ck["step"], ck["epoch"]
        print(f"[resume] {last} at {elapsed0/60:.1f} min, step {step}", flush=True)
    enc_params = [p for n, p in model.named_parameters() if not n.startswith(("ctc_head", "cls_head"))]
    logf = open(outdir / "log.jsonl", "a" if resumed else "w")
    t0 = time.time()
    last_save = last_eval = t0
    acc = collections.defaultdict(float)
    nacc = 0
    done = False
    model.train()
    while not done:
        got = 0
        for batch in loader_fn(epoch):
            got += 1
            el = elapsed0 + time.time() - t0
            frac = el / budget
            if frac >= 1.0:
                done = True
                break
            lr = lr_at(frac, peak_lr, warm)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(dev.type, dtype=torch.float16, enabled=cuda):
                loss, st = loss_fn(model, batch)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if frac < freeze_frac:
                for p in enc_params:
                    p.grad = None
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            nacc += 1
            if ema is not None:
                d = min(ema_decay, (1 + step) / (10 + step))
                torch._foreach_mul_(ema, d)
                torch._foreach_add_(ema, [p.detach() for p in plist], alpha=1 - d)
            acc["loss"] += float(loss)
            for k, v in st.items():
                acc[k] += float(v)
            if step % args.log_every == 0:
                rec = {"step": step, "epoch": epoch, "min": round(el / 60, 2), "lr": lr, "gn": float(gn)}
                rec.update({k: v / nacc for k, v in acc.items()})
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
                acc.clear()
                nacc = 0
            now = time.time()
            if dev_eval is not None and now - last_eval > args.eval_every * 60:
                backup = _swap(model, ema) if ema is not None else None
                w = dev_eval(model)
                if backup is not None:
                    _swap(model, backup)
                model.train()
                logf.write(json.dumps({"step": step, "min": round(el / 60, 2), "dev_wer": w}) + "\n")
                logf.flush()
                print(f"[dev] step {step} WER {w:.2f}", flush=True)
                last_eval = time.time()
            if now - last_save > 600:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "elapsed": el, "step": step, "epoch": epoch,
                            "ema": ema}, last)
                last_save = now
        epoch += 1
        if got == 0:
            raise RuntimeError("data loader produced no batches (check filters / max-duration)")
    if ema is not None:
        _swap(model, ema)          # the averaged weights become the final model
    return step


# ----------------------------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------------------------
def ctc_loss_fn(specaug, inter_weight=0.0):
    """CTC loss on the last layer (+ optional 'intermediate CTC' on the middle layer, same output head).
    The extra loss gives the lower layers a short gradient path, which is what lets a randomly
    initialised encoder leave the all-blank plateau."""
    def one(model, h, hl, batch, dev):
        lp = F.log_softmax(model.ctc_head(h).float(), -1).transpose(0, 1)
        loss = F.ctc_loss(lp, batch["y"].to(dev), hl, batch["ylens"].to(dev), blank=0, reduction="sum", zero_infinity=True)
        return loss / hl.sum()

    def fn(model, batch):
        dev = next(model.parameters()).device
        x, lens = batch["x"].to(dev, non_blocking=True).float(), batch["lens"].to(dev)
        if specaug:
            x = spec_augment(x, lens)
        mid = len(model.blocks) // 2 - 1
        h, hl = model.encode(x, lens, collect=mid if inter_weight > 0 else None)
        loss = one(model, h, hl, batch, dev)
        if inter_weight > 0 and model.mid is not None:
            loss = (1 - inter_weight) * loss + inter_weight * one(model, model.mid, hl, batch, dev)
        return loss, {}
    return fn


def pretrain_loss_fn(prob, span):
    def fn(model, batch):
        dev = next(model.parameters()).device
        x, lens, z = batch["x"].to(dev, non_blocking=True).float(), batch["lens"].to(dev), batch["z"].to(dev)
        m = make_mask(lens, x.size(1), prob, span)
        h, hl = model.encode(x, lens, mask=m)
        tp = h.size(1)
        sel = m[:, 3::4][:, :tp] & (z[:, :tp] != -100)
        if not bool(sel.any()):
            return h.sum() * 0.0, {}
        lg, tgt = model.cls_head(h)[sel].float(), z[:, :tp][sel]
        return F.cross_entropy(lg, tgt), {"acc": (lg.argmax(-1) == tgt).float().mean()}
    return fn


# ----------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_args(p):
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--kernel", type=int, default=31)
    p.add_argument("--n-cls", type=int, default=500)


def _common(p):
    p.add_argument("--work", required=True, help="dir with the feature stores (stats.json, bpe.model, *.bin)")
    p.add_argument("--exp", required=True, help="dir for experiments")
    p.add_argument("--name", required=True)
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max-duration", type=float, default=600, help="seconds of (padded) audio per batch")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=float, default=15, help="minutes between dev-WER evaluations")
    p.add_argument("--dev-utts", type=int, default=300)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off)")
    p.add_argument("--train-store", default="lab", help="which labelled store to train on: lab | full")
    p.add_argument("--curriculum", action="store_true", help="epoch 0 = shortest utterances first")
    p.add_argument("--inter-weight", type=float, default=0.0, help="weight of the intermediate-CTC loss")
    p.add_argument("--warm", type=float, default=-1, help="warm-up fraction of the budget (-1 = default)")
    p.add_argument("--retries", type=int, default=0, help="retrain with lr*0.6 if dev WER > --max-ok-wer")
    p.add_argument("--max-ok-wer", type=float, default=0.0, help="0 = no collapse check")
    _model_args(p)


def _spm(work):
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(Path(work) / "bpe.model"))


def _new_model(a, sp):
    return AcousticModel(d=a.d, heads=a.heads, ff=a.ff, layers=a.layers, kernel=a.kernel, n_cls=a.n_cls, vocab=sp.get_piece_size())


def cmd_train_asr(a, finetune=False):
    out = Path(a.exp) / a.name
    if (out / "final.pt").exists() and not a.force:
        print(f"[skip] {out/'final.pt'} exists")
        return
    work = Path(a.work)
    sp, stats = _spm(work), load_stats(work)
    lab, devs = Store(work / a.train_store), Store(work / "dev")
    tokens = [np.array(sp.encode(t), np.int64) for t in lab.texts()]
    ok = [0 < len(tokens[i]) <= sub_len(lab.lens[i]) for i in range(len(lab))]
    dev_idx = list(range(min(a.dev_utts, len(devs))))
    dev = _device()

    def dev_eval(m):
        hy = decode_store(m, devs, sp, stats, dev_idx)
        return wer_corpus([devs.items[i]["text"] for i in dev_idx], hy)[0]

    final, model = None, None
    for attempt in range(a.retries + 1):
        run = copy.copy(a)
        run.lr = a.lr * (0.6 ** attempt)
        run.seed = a.seed + 17 * attempt
        if attempt > 0:
            run.force = True            # drop last.pt of the collapsed attempt
            print(f"[{a.name}] attempt {attempt}: previous dev WER {final:.1f}% -> retry with lr={run.lr:.2g}", flush=True)
        torch.manual_seed(run.seed)
        random.seed(run.seed)
        model = _new_model(a, sp)
        if finetune:
            ck = torch.load(a.init, map_location="cpu", weights_only=False)
            sd = {k: v for k, v in ck["model"].items() if not k.startswith("ctc_head")}
            print("[finetune] init from", a.init, model.load_state_dict(sd, strict=False))
        model.to(dev)
        print(f"[{a.name}] params={count_params(model)/1e6:.2f}M  train={a.train_store} {lab.hours:.1f}h  device={dev}  lr={run.lr:.2g}", flush=True)

        def loader_fn(epoch, run=run):
            b = make_batches(lab.lens, int(run.max_duration * 100), True, run.seed + epoch, 50, 3000, ok,
                             curriculum=run.curriculum and epoch == 0)
            return make_loader(BatchDS(lab, b, stats, tokens=tokens), run.workers)

        warm = a.warm if a.warm >= 0 else (0.1 if finetune else 0.15)
        train_loop(model, loader_fn, ctc_loss_fn(True, a.inter_weight), run, out, run.lr, warm=warm,
                   freeze_frac=a.freeze_frac if finetune else 0.0, dev_eval=dev_eval, ema_decay=a.ema)
        final = dev_eval(model)
        print(f"[{a.name}] final dev WER {final:.2f}", flush=True)
        if a.max_ok_wer <= 0 or final <= a.max_ok_wer:
            break
    if a.max_ok_wer > 0 and final > a.max_ok_wer:
        save_ckpt(out / "final_FAILED.pt", model, vocab=sp.get_piece_size(), dev_wer=final)
        raise RuntimeError(f"[{a.name}] did not converge (dev WER {final:.1f}% > {a.max_ok_wer}%) after {a.retries + 1} attempt(s)")
    save_ckpt(out / "final.pt", model, vocab=sp.get_piece_size(), dev_wer=final, stage="finetune" if finetune else "asr")


def cmd_pretrain(a):
    out = Path(a.exp) / a.name
    if (out / "final.pt").exists() and not a.force:
        print(f"[skip] {out/'final.pt'} exists")
        return
    torch.manual_seed(a.seed)
    dev = _device()
    work = Path(a.work)
    sp, stats = _spm(work), load_stats(work)
    unl = Store(work / "unl")
    labels = LabelStore(Path(a.work) / f"labels_{a.labels}")
    model = _new_model(a, sp).to(dev)
    print(f"[{a.name}] params={count_params(model)/1e6:.2f}M  unlabelled={unl.hours:.1f}h  labels={a.labels} (K={labels.k})", flush=True)

    def loader_fn(epoch):
        b = make_batches(unl.lens, int(a.max_duration * 100), True, a.seed + epoch, 100, 2000)
        return make_loader(BatchDS(unl, b, stats, labels=labels), a.workers)

    train_loop(model, loader_fn, pretrain_loss_fn(a.mask_prob, a.mask_span), a, out, a.lr, warm=0.08)
    save_ckpt(out / "final.pt", model, vocab=sp.get_piece_size(), stage="pretrain", labels=a.labels)


def kmeans_fit(X, k, iters=100, seed=0, tol=1e-4):
    g = torch.Generator(device=X.device).manual_seed(seed)
    N = X.size(0)
    C = X[torch.randperm(N, generator=g, device=X.device)[:k]].clone()
    for it in range(iters):
        sums, cnt = torch.zeros_like(C), torch.zeros(k, device=X.device)
        for ch in X.split(200000):
            a = kmeans_assign(C, ch)
            sums.index_add_(0, a, ch)
            cnt.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
        newC = sums / cnt.clamp(min=1)[:, None]
        empty = cnt == 0
        if bool(empty.any()):
            newC[empty] = X[torch.randint(0, N, (int(empty.sum()),), generator=g, device=X.device)]
        shift = float((newC - C).norm() / C.norm().clamp(min=1e-8))
        C = newC
        if it % 10 == 0:
            print(f"  kmeans iter {it} shift {shift:.5f} empty {int(empty.sum())}", flush=True)
        if shift < tol:
            break
    return C


def kmeans_assign(C, f):
    d = (f ** 2).sum(1, keepdim=True) - 2 * f @ C.t() + (C ** 2).sum(1)[None]
    return d.argmin(1)


@torch.no_grad()
def cmd_kmeans(a):
    """Fit k-means on frame features and write one cluster id per output frame for every unlabelled utt.
       --source fbank          : HuBERT iteration-1 codebook (k-means on 80-d Fbank, pooled to the encoder frame rate)
       --source <ckpt>         : features of an encoder; --layer -1 = final layer (VietASR), i>=0 = i-th block (HuBERT it>=2)
    """
    binp = Path(a.work) / f"labels_{a.out}.bin"
    if binp.exists() and not a.force:
        print(f"[skip] {binp} exists")
        return
    dev = _device()
    work = Path(a.work)
    stats = load_stats(work)
    unl = Store(work / "unl")
    model = None
    if a.source != "fbank":
        model, _ = load_model(a.source, dev)

    def feats(x, lens):
        x = x.to(dev).float()
        if model is None:
            return F.avg_pool1d(x.transpose(1, 2), kernel_size=7, stride=4).transpose(1, 2)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            h, _ = model.encode(x, lens.to(dev), layer=None if a.layer < 0 else a.layer)
        return h.float()

    # ---- pass 1: sample ~km_hours of frames and fit
    rng = random.Random(a.seed)
    order = list(range(len(unl)))
    rng.shuffle(order)
    pick, tot = [], 0.0
    for i in order:
        if tot >= a.km_hours:
            break
        pick.append(i)
        tot += unl.lens[i] / 360000.0
    local = make_batches([unl.lens[i] for i in pick], 300000, False, 0)
    ds = BatchDS(unl, [[pick[j] for j in b] for b in local], stats)
    chunks = []
    for b in make_loader(ds, 2):
        f = feats(b["x"], b["lens"])
        tl = sub_len(b["lens"])
        for k in range(f.size(0)):
            chunks.append(f[k, : int(tl[k])])
    X = torch.cat(chunks)
    print(f"[kmeans {a.out}] fitting K={a.k} on {X.size(0)} frames x {X.size(1)} dims ({tot:.1f} h)", flush=True)
    C = kmeans_fit(X, a.k, a.iters, a.seed)
    del X, chunks
    # ---- pass 2: label everything
    allb = make_batches(unl.lens, 300000, False, 0)
    ds = BatchDS(unl, allb, stats)
    offs, lens_out = [0] * len(unl), [0] * len(unl)
    hist = torch.zeros(a.k, device=dev)
    pos = 0
    with open(binp, "wb") as fo:
        res = {}
        for b in make_loader(ds, 2):
            f = feats(b["x"], b["lens"])
            tl = sub_len(b["lens"])
            for k, i in enumerate(b["idx"].tolist()):
                lab = kmeans_assign(C, f[k, : int(tl[k])]).to(torch.int16)
                hist += torch.bincount(lab.long(), minlength=a.k).float()
                res[i] = lab.cpu().numpy()
        for i in range(len(unl)):          # write in store order so offsets are monotone
            offs[i], lens_out[i] = pos, int(res[i].shape[0])
            res[i].tofile(fo)
            pos += lens_out[i]
    p = hist / hist.sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum())
    if math.exp(ent) < a.min_perplexity:
        binp.unlink()
        raise RuntimeError(f"[kmeans {a.out}] codebook collapsed (perplexity {math.exp(ent):.1f} < {a.min_perplexity}): "
                           f"the source model '{a.source}' produces near-constant features, so these labels are useless for pre-training")
    json.dump({"offs": offs, "lens": lens_out, "k": a.k, "source": a.source, "layer": a.layer,
               "entropy_nats": ent, "perplexity": math.exp(ent), "used": int((hist > 0).sum()),
               "hist": hist.cpu().tolist()}, open(work / f"labels_{a.out}.json", "w"))
    torch.save(C.cpu(), work / f"centroids_{a.out}.pt")
    print(f"[kmeans {a.out}] done: {pos} labels, codebook perplexity {math.exp(ent):.1f} / {a.k}, used {int((hist>0).sum())}", flush=True)


def cmd_eval(a):
    dev = _device()
    work = Path(a.work)
    sp, stats = _spm(work), load_stats(work)
    model, ck = load_model(a.ckpt, dev)
    res = {"name": a.name or Path(a.ckpt).parent.name, "params_M": count_params(model) / 1e6, "wer": {}, "words": {}, "samples": {}}
    sets = ["dev"] + TEST_SETS
    tot_audio = tot_time = 0.0
    for s in sets:
        pf = work / (s if s == "dev" else f"test__{s}")
        if not Path(str(pf) + ".json").exists():
            continue
        st = Store(pf)
        hy, t = decode_store(model, st, sp, stats, return_time=True)
        refs = st.texts()
        w, n = wer_corpus(refs, hy)
        res["wer"][s], res["words"][s] = w, n
        res["samples"][s] = [{"ref": refs[i], "hyp": hy[i]} for i in range(min(8, len(refs)))]
        tot_audio += st.hours * 3600
        tot_time += t
        print(f"[eval {res['name']}] {s:28s} WER {w:6.2f}  ({n} words)", flush=True)
    res["rtf"] = tot_time / max(tot_audio, 1e-9)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


def cmd_export(a):
    """Package a fine-tuned model for the Streamlit demo."""
    import shutil

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()}
    torch.save({"model": sd, "cfg": ck["cfg"], "vocab": ck.get("vocab"), "level_norm": not a.no_level_norm}, out / "model.pt")
    shutil.copy(Path(a.work) / "bpe.model", out / "bpe.model")
    shutil.copy(Path(a.work) / "stats.json", out / "stats.json")
    meta = {"name": a.name, "clean_test_wer": a.wer if a.wer >= 0 else None, "note": a.note, "level_norm": not a.no_level_norm}
    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False, indent=1)
    print("exported to", out)


# ---------------------------------------------------------------- data preparation (Kaggle)
def _init_worker():
    torch.set_num_threads(1)


def _decode_job(job):
    sid, label, vb, keep_wave = job
    try:
        wave = decode_audio(vb)
        if wave.size < 1600:
            return None
        return sid, label, fbank(wave).numpy().astype(np.float16), (np.clip(wave, -1, 1) * 32767).astype(np.int16) if keep_wave else None
    except Exception as e:  # corrupt clip -> skip
        print("decode error", sid, repr(e)[:100], flush=True)
        return None


def _pmap(pool, jobs, window=64):
    q = deque()
    for j in jobs:
        q.append(pool.apply_async(_decode_job, (j,)))
        if len(q) >= window:
            yield q.popleft().get()
    while q:
        yield q.popleft().get()


def _stream(split, seed=None, buffer=0, retries=6):
    from datasets import load_dataset

    consumed = 0
    for attempt in range(retries):
        try:
            ds = load_dataset(REPO, split=split, streaming=True)
            if buffer:
                ds = ds.shuffle(seed=seed, buffer_size=buffer)
            if consumed:
                ds = ds.skip(consumed)
            for ex in ds:
                consumed += 1
                yield ex
            return
        except Exception as e:
            print(f"[stream {split}] error after {consumed} samples: {repr(e)[:150]} - retry {attempt+1}/{retries}", flush=True)
            time.sleep(15)
    raise RuntimeError(f"stream {split} failed")


def _fields(ex):
    sid = to_text(ex.get("sample_id")) or to_text(ex.get("__key__"))
    if "video" not in ex or "label" not in ex:
        raise KeyError(f"unexpected columns: {list(ex.keys())}")
    return sid, norm_text(to_text(ex["label"])), to_bytes(ex["video"])


def _video_id(sid):
    m = re.split(r"_\d{5}_", sid)
    return m[0] if len(m) > 1 else sid[:11]


def train_bpe(texts, work, vocab):
    import sentencepiece as spm

    tf = Path(work) / "lab_text.txt"
    tf.write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(tf), model_prefix=str(Path(work) / "bpe"), vocab_size=vocab, model_type="bpe",
        character_coverage=1.0, pad_id=0, pad_piece="<blk>", unk_id=1, bos_id=-1, eos_id=-1,
        normalization_rule_name="identity", hard_vocab_limit=False, input_sentence_size=2000000, shuffle_input_sentence=True)


def compute_stats(work, n=3000):
    st = Store(Path(work) / "lab")
    sel = np.linspace(0, len(st) - 1, min(n, len(st))).astype(int)
    X = np.concatenate([st.get(int(i)).astype(np.float32) for i in sel])
    json.dump({"mean": X.mean(0).tolist(), "std": X.std(0).clip(1e-3).tolist()}, open(Path(work) / "stats.json", "w"))


def cmd_prep(a):
    import multiprocessing as mp

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    if a.what in ("train", "all") and not ((work / "lab.json").exists() and (work / "full.json").exists()):
        # split by VIDEO (crc32 of the video id) so no speaker/video appears in two roles:
        #   dev  : 2 % of videos      lab : 20 % of videos (<= lab_hours, with transcripts)
        #   unl  : the rest (<= unl_hours, transcripts thrown away)
        #   full : every non-dev clip with transcript (<= full_hours) - only for the bonus demo fine-tune
        W = {k: StoreWriter(work / k) for k in ("lab", "unl", "dev", "full")}
        cap_h = {"lab": a.lab_hours, "unl": a.unl_hours, "full": a.full_hours}
        n = 0
        t0 = time.time()

        def want(role, has_label):
            t = []
            if role == "dev":
                if len(W["dev"].items) < a.dev_utts and has_label:
                    t.append("dev")
                return t
            if role == "lab" and has_label and W["lab"].hours < cap_h["lab"]:
                t.append("lab")
            if role == "unl" and W["unl"].hours < cap_h["unl"]:
                t.append("unl")
            if has_label and W["full"].hours < cap_h["full"]:
                t.append("full")
            return t

        def finished():
            return (len(W["dev"].items) >= a.dev_utts and all(W[k].hours >= cap_h[k] for k in cap_h))

        targets = deque()

        def tagged():
            for ex in _stream("train", seed=1, buffer=1000):
                sid, label, vb = _fields(ex)
                h = zlib.crc32(_video_id(sid).encode())
                role = "dev" if h % 50 == 1 else ("lab" if h % 5 == 0 else "unl")
                t = want(role, bool(label))
                if not t:
                    continue
                targets.append(t)
                yield (sid, label, vb, False)
                if finished():
                    return

        with ctx.Pool(a.workers, initializer=_init_worker) as pool:
            for r in _pmap(pool, tagged()):
                t = targets.popleft()
                if r is None:
                    continue
                sid, label, feats, _ = r
                for k in t:
                    W[k].add(sid, None if k == "unl" else label, feats)
                n += 1
                if n % 4000 == 0:
                    print(f"[prep] {n} utts  lab {W['lab'].hours:.1f}h  unl {W['unl'].hours:.1f}h  full {W['full'].hours:.1f}h  "
                          f"dev {len(W['dev'].items)}  {(time.time()-t0)/60:.1f} min", flush=True)
        for w in W.values():
            w.close()
        print(f"[prep] train done: lab {W['lab'].hours:.1f}h, unl {W['unl'].hours:.1f}h, full {W['full'].hours:.1f}h, "
              f"dev {len(W['dev'].items)} utts", flush=True)
        train_bpe(Store(work / "lab").texts(), work, a.vocab)
        compute_stats(work)
    if a.what in ("test", "all"):
        for s in TEST_SETS:
            pf = work / f"test__{s}"
            if Path(str(pf) + ".json").exists():
                continue
            w = StoreWriter(pf, keep_wave=True)
            test_jobs = ((*_fields(ex)[:3], True) for ex in _stream(s))
            with ctx.Pool(a.workers, initializer=_init_worker) as pool:
                for r in _pmap(pool, test_jobs):
                    if r is not None:
                        w.add(r[0], r[1], r[2], r[3])
            w.close()
            print(f"[prep] {s}: {len(w.items)} utts, {w.hours:.2f} h", flush=True)


def _write_wav(path, wave, sr=SR):
    import wave as wavemod

    pcm = (np.clip(np.asarray(wave, np.float32), -1, 1) * 32767).astype(np.int16)
    with wavemod.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def cmd_samples(a):
    """Export a handful of demo clips (wav + reference + what the model says) from the test sets."""
    import shutil

    dev = _device()
    work = Path(a.work)
    sp, stats = _spm(work), load_stats(work)
    model, _ = load_model(a.ckpt, dev)
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    def scored(setname):
        st = Store(work / f"test__{setname}")
        hy = decode_store(model, st, sp, stats)
        rows = []
        for i, (it, h) in enumerate(zip(st.items, hy)):
            ref = it["text"]
            nw = len(ref.split())
            dur = it["n"] / 100.0
            if nw < 8 or not (a.min_dur <= dur <= a.max_dur):
                continue
            rows.append({"set": setname, "i": i, "vid": _video_id(it["id"]), "dur": dur, "ref": ref, "hyp": h,
                         "wer": 100.0 * edit_distance(ref.split(), h.split()) / nw})
        return st, rows

    def pick(rows, n, used, key):
        got = []
        for r in sorted(rows, key=key):
            if r["vid"] in used:
                continue
            used.add(r["vid"])
            got.append(r)
            if len(got) == n:
                break
        return got

    stores, picked, used = {}, [], set()
    stores["test"], clean = scored("test")
    med = float(np.median([r["wer"] for r in clean])) if clean else 0.0
    for cat, rows in (("de_wer_thap_nhat", pick(clean, a.n_easy, used, lambda r: (r["wer"], abs(r["dur"] - 9)))),
                      ("trung_binh", pick(clean, a.n_typical, used, lambda r: abs(r["wer"] - med)))):
        picked += [dict(r, cat=cat) for r in rows]
    for s2, cat in (("test_snr_10_interferer_1", "nhieu_10dB_1_nguoi_chen"), ("test_snr_0_interferer_1", "nhieu_0dB_1_nguoi_chen")):
        stores[s2], rr = scored(s2)
        picked += [dict(r, cat=cat) for r in pick(rr, a.n_noisy, set(), lambda r: r["wer"])]

    meta = []
    for k, r in enumerate(picked, 1):
        name = f"{k:02d}_{r['cat']}.wav"
        _write_wav(out / name, stores[r["set"]].wave(r["i"]))
        meta.append({"file": name, "category": r["cat"], "set": r["set"], "duration_s": round(r["dur"], 1), "reference": r["ref"],
                     "model_output": r["hyp"], "wer_percent": round(r["wer"], 1)})

    # simulated conversation: easy clips from different videos (= different speakers), 0.8 s apart
    easy = [r for r in picked if r["cat"] == "de_wer_thap_nhat"][: a.n_convo]
    if len(easy) >= 2:
        gap = np.zeros(int(0.8 * SR), np.float32)
        parts, lines, t = [], [], 0.0
        for k, r in enumerate(easy):
            w = stores[r["set"]].wave(r["i"])
            lines.append(f"[{t:5.1f}s] Nguoi {k % 2 + 1}: {r['ref']}")
            parts += [w, gap]
            t += len(w) / SR + 0.8
        _write_wav(out / "00_hoi_thoai_gia_lap_nhieu_nguoi.wav", np.concatenate(parts))
        meta.insert(0, {"file": "00_hoi_thoai_gia_lap_nhieu_nguoi.wav", "category": "hoi_thoai_gia_lap", "set": "test",
                        "duration_s": round(t, 1), "reference": "\n".join(lines), "model_output": "", "wer_percent": None})
    json.dump(meta, open(out / "samples.json", "w"), ensure_ascii=False, indent=1)
    (out / "README.txt").write_text(
        "Cac clip mau lay tu tap test cua ViCocktail (KHONG nam trong du lieu huan luyen).\n"
        "de_wer_thap_nhat : chon clip co WER thap nhat -> de demo dep, KHONG phai chat luong trung binh.\n"
        "trung_binh       : clip co WER sat trung vi -> chat luong dien hinh.\n"
        "nhieu_*          : co 1 nguoi khac noi chen vao (SNR 10 dB / 0 dB); model khong tach duoc giong chong lan.\n"
        "00_hoi_thoai...  : ghep cac clip cua nhieu video (nhieu giong) de thu che do nhieu nguoi noi.\n"
        "samples.json chua loi thoai tham chieu va ket qua model.\n", encoding="utf-8")
    shutil.make_archive(str(out), "zip", out)
    print(f"[samples] {len(meta)} files -> {out} (+ {out}.zip)")
    for m in meta:
        print(f"  {m['file']:48s} {m['duration_s']:5.1f}s  WER {m['wer_percent']}")


# ----------------------------------------------------------------------------------------------
# inference engine (used by the Streamlit demo)
# ----------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, release_dir, device=None):
        import sentencepiece as spm

        d = Path(release_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        self.model = AcousticModel(**ck["cfg"])
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        self.model.to(self.device).eval()
        self.level = bool(ck.get("level_norm", False))     # old releases were trained without loudness normalisation
        self.meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
        self.sp = spm.SentencePieceProcessor(model_file=str(d / "bpe.model"))
        s = json.load(open(d / "stats.json"))
        self.mean, self.std = np.array(s["mean"], np.float32), np.array(s["std"], np.float32)

    @torch.no_grad()
    def transcribe(self, waves, batch_size=8):
        """list of float32 mono 16 kHz arrays -> list of (text, confidence in [0,1])."""
        out = [None] * len(waves)
        order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
        for s in range(0, len(order), batch_size):
            ids = order[s: s + batch_size]
            fb = [((fbank(waves[i], self.level).numpy() - self.mean) / self.std).astype(np.float32) for i in ids]
            lens = torch.tensor([f.shape[0] for f in fb])
            x = torch.zeros(len(fb), int(lens.max()), N_MELS)
            for k, f in enumerate(fb):
                x[k, : f.shape[0]] = torch.from_numpy(f)
            x, lens = x.to(self.device), lens.to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                h, hl = self.model.encode(x, lens)
                lg = self.model.ctc_head(h)
            pr = F.softmax(lg.float(), -1)
            mx, am = pr.max(-1)
            am, mx, hl = am.cpu().numpy(), mx.cpu().numpy(), hl.cpu().numpy()
            for k, i in enumerate(ids):
                a, m = am[k, : hl[k]], mx[k, : hl[k]]
                nb = a != 0
                conf = float(m[nb].mean()) if nb.any() else 0.0
                out[i] = (norm_text(self.sp.decode(ctc_collapse(a))), conf)
        return out


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--work", required=True)
    p.add_argument("--what", default="all", choices=["train", "test", "all"])
    p.add_argument("--lab-hours", type=float, default=50)
    p.add_argument("--unl-hours", type=float, default=150)
    p.add_argument("--full-hours", type=float, default=300)
    p.add_argument("--dev-utts", type=int, default=400)
    p.add_argument("--vocab", type=int, default=500)
    p.add_argument("--workers", type=int, default=3)

    p = sub.add_parser("train-asr")
    _common(p)
    p.set_defaults(freeze_frac=0.0)
    p = sub.add_parser("finetune")
    _common(p)
    p.add_argument("--init", required=True)
    p.add_argument("--freeze-frac", type=float, default=0.05)

    p = sub.add_parser("pretrain")
    _common(p)
    p.add_argument("--labels", required=True)
    p.add_argument("--mask-prob", type=float, default=0.08)
    p.add_argument("--mask-span", type=int, default=10)

    p = sub.add_parser("kmeans")
    p.add_argument("--work", required=True)
    p.add_argument("--source", required=True, help="'fbank' or path to a checkpoint")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--km-hours", type=float, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-perplexity", type=float, default=30, help="refuse codebooks that use fewer than ~this many clusters")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--work", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="")

    p = sub.add_parser("export")
    p.add_argument("--work", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--wer", type=float, default=-1, help="clean-test WER to display in the app")
    p.add_argument("--note", default="")
    p.add_argument("--no-level-norm", action="store_true", help="for checkpoints trained before v2")

    p = sub.add_parser("samples")
    p.add_argument("--work", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-easy", type=int, default=6)
    p.add_argument("--n-typical", type=int, default=3)
    p.add_argument("--n-noisy", type=int, default=2)
    p.add_argument("--n-convo", type=int, default=5)
    p.add_argument("--min-dur", type=float, default=5.0)
    p.add_argument("--max-dur", type=float, default=15.0)

    a = ap.parse_args(argv)
    {"prep": cmd_prep, "train-asr": cmd_train_asr, "finetune": lambda x: cmd_train_asr(x, True),
     "pretrain": cmd_pretrain, "kmeans": cmd_kmeans, "eval": cmd_eval, "export": cmd_export,
     "samples": cmd_samples}[a.cmd](a)


if __name__ == "__main__":
    main()
