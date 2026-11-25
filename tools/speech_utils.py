"""
speech_utils.py - helpers for the Streamlit demo: VAD, light-weight speaker clustering, SRT/TXT/JSON export.

Only numpy + scipy are required.  Silero-VAD (`pip install silero-vad`) and SpeechBrain ECAPA
(`pip install speechbrain`) are used automatically when installed and give better results.
"""
from __future__ import annotations

import json

import numpy as np

SR = 16000


# ------------------------------------------------------------------------------------------ VAD
def _frame_db(wave, frame=320, hop=160):
    if len(wave) < frame:
        wave = np.pad(wave, (0, frame - len(wave)))
    n = 1 + (len(wave) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((wave[idx] ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def _median_smooth(x, k=7):
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)


def energy_vad(wave, sr=SR, sensitivity=0.5, min_speech=0.30, min_silence=0.35, max_seg=18.0, pad=0.12):
    """Adaptive-threshold energy VAD -> list of (start_s, end_s).  sensitivity in [0,1]: higher = more speech."""
    hop = 160
    db = _median_smooth(_frame_db(wave, 320, hop))
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    span = max(peak - floor, 6.0)
    thr = floor + span * (0.55 - 0.45 * sensitivity)       # sens 0 -> 55% of range, sens 1 -> 10%
    sp = db > thr
    segs, start = [], None
    for i, v in enumerate(sp):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, len(sp)])
    fps = sr / hop
    merged = []
    for s, e in segs:                                         # merge short gaps
        if merged and (s - merged[-1][1]) / fps < min_silence:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    dur = len(wave) / sr
    for s, e in merged:
        if (e - s) / fps < min_speech:
            continue
        out.append((max(0.0, s / fps - pad), min(dur, e / fps + pad)))
    return split_long(out, db, fps, max_seg)


def split_long(segs, db, fps, max_seg):
    """split segments longer than max_seg at the quietest frame in their last 40 %."""
    out = []
    for s, e in segs:
        while e - s > max_seg:
            lo, hi = int((s + 0.6 * max_seg) * fps), int((s + max_seg) * fps)
            hi = min(hi, len(db) - 1)
            cut = lo + int(np.argmin(db[lo:hi + 1])) if hi > lo else hi
            out.append((s, cut / fps))
            s = cut / fps
        out.append((s, e))
    return out


def silero_vad(wave, sr=SR, max_seg=18.0, min_silence=0.35):
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(wave), model, sampling_rate=sr, return_seconds=True,
                               min_silence_duration_ms=int(min_silence * 1000))
    segs = [(t["start"], t["end"]) for t in ts]
    db = _median_smooth(_frame_db(wave))
    return split_long(segs, db, sr / 160, max_seg)


def vad_segments(wave, method="auto", **kw):
    """method: auto | energy | silero.   Returns (segments, method_used)."""
    if method in ("auto", "silero"):
        try:
            return silero_vad(wave, max_seg=kw.get("max_seg", 18.0), min_silence=kw.get("min_silence", 0.35)), "silero"
        except Exception:
            if method == "silero":
                raise
    return energy_vad(wave, **kw), "energy"


# ------------------------------------------------------------------------------- speaker clustering
def _logmel_np(wave, n_mels=40, n_fft=512, hop=160, win=400):
    w = np.pad(wave, (0, max(0, win - len(wave))))
    n = 1 + (len(w) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(w[idx] * np.hanning(win), n_fft, axis=1)) ** 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(50), hz2mel(7600), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)))
    return np.log(spec @ fb.T + 1e-8)


def mfcc_embedding(wave, n_mfcc=20):
    """Weak but dependency-free speaker descriptor: mean+std of MFCCs over voiced frames."""
    from scipy.fft import dct

    lm = _logmel_np(wave)
    e = lm.mean(1)
    lm = lm[e > np.percentile(e, 35)] if len(lm) > 20 else lm       # keep the louder (voiced) frames
    m = dct(lm, type=2, axis=1, norm="ortho")[:, 1:n_mfcc + 1]
    return np.concatenate([m.mean(0), m.std(0)])


_ECAPA = {}


def ecapa_embedding(wave):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    if "m" not in _ECAPA:
        _ECAPA["m"] = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                     savedir="pretrained_ecapa", run_opts={"device": "cpu"})
    with torch.no_grad():
        return _ECAPA["m"].encode_batch(torch.from_numpy(wave)[None]).squeeze().cpu().numpy()


def cluster_speakers(waves, durations, n_speakers=0, threshold=None, method="auto", min_dur=0.8):
    """Return (labels, method_used).  n_speakers=0 -> choose automatically with a distance threshold."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(waves)
    if n == 0:
        return [], method
    use = [i for i in range(n) if durations[i] >= min_dur] or list(range(n))
    emb, used = None, "mfcc"
    if method in ("auto", "ecapa"):
        try:
            emb = np.stack([ecapa_embedding(waves[i]) for i in use])
            used = "ecapa"
        except Exception:
            if method == "ecapa":
                raise
    if emb is None:
        emb = np.stack([mfcc_embedding(waves[i]) for i in use])
        emb = emb - emb.mean(0, keepdims=True)                       # remove the channel/global component
    if threshold is None:
        threshold = 0.6 if used == "ecapa" else 0.9
    if len(use) < 2 or n_speakers == 1:
        lab_use = np.ones(len(use), int)
    else:
        Z = linkage(emb + 1e-9, method="average", metric="cosine")
        lab_use = fcluster(Z, t=n_speakers, criterion="maxclust") if n_speakers >= 2 else fcluster(Z, t=threshold, criterion="distance")
    labels = [0] * n
    for k, i in enumerate(use):
        labels[i] = int(lab_use[k])
    last = int(labels[use[0]])
    for i in range(n):                                               # short segments inherit neighbours
        if i in use:
            last = labels[i]
        else:
            labels[i] = last
    order = {}
    for l in labels:                                                 # renumber by first appearance: 1,2,3..
        order.setdefault(l, len(order) + 1)
    return [order[l] for l in labels], used


# ----------------------------------------------------------------------------------------- export
def _ts(t, srt=True):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{',' if srt else '.'}{int(round((s - int(s)) * 1000)):03d}"


def to_srt(rows):
    out = []
    for i, r in enumerate(rows, 1):
        who = f"[Người {r['speaker']}] " if r.get("speaker") else ""
        out.append(f"{i}\n{_ts(r['start'])} --> {_ts(r['end'])}\n{who}{r['text']}\n")
    return "\n".join(out)


def to_txt(rows):
    return "\n".join(f"[{_ts(r['start'], False)[:-4]}] " + (f"Người {r['speaker']}: " if r.get("speaker") else "") + r["text"] for r in rows)


def to_json(rows):
    return json.dumps(rows, ensure_ascii=False, indent=1)
