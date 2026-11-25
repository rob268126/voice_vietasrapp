"""
VietASR-mini 
Tải lên (hoặc ghi âm / chọn clip mẫu) một file audio/video có một hoặc nhiều người nói -> bản ghi tiếng Việt có mốc thời gian.

Chạy:   streamlit run app.py
Cần:    thư mục `release/` (và tuỳ chọn `samples/`) lấy từ vietasr_demo_kit.zip do notebook Kaggle tạo ra.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

import tools.speech_utils as su
import tools.vietasr_mini as vm

st.set_page_config(page_title="VietASR demo", page_icon="🎙️", layout="wide")

SPEAKER_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]
AUDIO_EXT = ["wav", "mp3", "m4a", "flac", "ogg", "opus", "aac", "wma"]
VIDEO_EXT = ["mp4", "mkv", "mov", "webm", "avi", "m4v"]
LOW_CONF = 0.55          # ngưỡng uncertain


@st.cache_resource(show_spinner="Load mô hình")
def load_engine(release_dir: str, device: str):
    return vm.Engine(release_dir, None if device == "auto" else device)


def table(rows):
    try:
        st.dataframe(rows, width="stretch")
    except TypeError:
        st.dataframe(rows, use_container_width=True)


def run_pipeline(engine, wave, mode, n_speakers, vad_method, sens, min_sil, max_seg, spk_method, batch, progress):
    """wave (float32, 16 kHz) -> list of rows {start,end,speaker,text,conf}."""
    t0 = time.time()
    progress(0.05, "Phát hiện đoạn có tiếng nói (VAD)…")
    segs, vad_used = su.vad_segments(wave, vad_method, sensitivity=sens, min_silence=min_sil, max_seg=max_seg)
    if not segs:                                           # không phát hiện gì -> thử cắt đều
        segs = [(s, min(s + max_seg, len(wave) / vm.SR)) for s in np.arange(0, len(wave) / vm.SR, max_seg)]
    waves = [wave[int(s * vm.SR): int(e * vm.SR)] for s, e in segs]
    durs = [e - s for s, e in segs]

    speakers, spk_used = [0] * len(segs), "-"
    if mode == "multi":
        progress(0.20, "Phân tách người nói…")
        speakers, spk_used = su.cluster_speakers(waves, durs, n_speakers=n_speakers, method=spk_method)

    progress(0.35, f"Nhận dạng {len(segs)} đoạn…")
    texts = []
    for s in range(0, len(waves), batch):
        texts += engine.transcribe(waves[s: s + batch], batch_size=batch)
        progress(0.35 + 0.6 * min(1.0, (s + batch) / max(len(waves), 1)), f"Nhận dạng {min(s + batch, len(waves))}/{len(waves)} đoạn…")
    rows = [
        {"start": round(s, 2), "end": round(e, 2), "speaker": sp if mode == "multi" else 0, "text": t, "conf": round(c, 3)}
        for (s, e), sp, (t, c) in zip(segs, speakers, texts, strict=True)
    ]
    rows = [r for r in rows if r["text"].strip()]
    info = {"vad": vad_used, "speaker_method": spk_used, "elapsed": time.time() - t0, "audio_sec": len(wave) / vm.SR}
    progress(1.0, "Xong")
    return rows, info


def render_transcript(rows, multi):
    html = []
    for r in rows:
        ts = f"{int(r['start'] // 60):02d}:{r['start'] % 60:05.2f}"
        chip = ""
        if multi:
            c = SPEAKER_COLORS[(r["speaker"] - 1) % len(SPEAKER_COLORS)]
            chip = f"<span style='background:{c};color:#fff;border-radius:10px;padding:1px 8px;margin-right:6px;font-size:0.8em'>Người {r['speaker']}</span>"
        style = "opacity:0.55;font-style:italic" if r["conf"] < LOW_CONF else ""
        html.append(f"<div style='margin:4px 0;{style}'><code>{ts}</code> {chip}{r['text']}</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)
    if any(r["conf"] < LOW_CONF for r in rows):
        st.caption("Dòng chữ xám nghiêng = mô hình không chắc chắn (độ tin cậy thấp), thường do nhiễu, nhạc nền hoặc tiếng nói chồng.")


def render_timeline(rows, total, multi):
    import matplotlib.pyplot as plt

    spk = sorted({r["speaker"] for r in rows}) if multi else [0]
    fig, ax = plt.subplots(figsize=(10, 0.7 + 0.5 * len(spk)))
    for r in rows:
        y = spk.index(r["speaker"] if multi else 0)
        c = SPEAKER_COLORS[(r["speaker"] - 1) % len(SPEAKER_COLORS)] if multi else SPEAKER_COLORS[0]
        ax.broken_barh([(r["start"], r["end"] - r["start"])], (y - 0.35, 0.7), facecolors=c)
    ax.set_yticks(range(len(spk)))
    ax.set_yticklabels([f"Người {s}" if multi else "Giọng nói" for s in spk])
    ax.set_xlim(0, max(total, 1))
    ax.set_xlabel("giây")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def load_samples(folder: Path):
    f = folder / "samples.json"
    if not f.exists():
        return []
    try:
        return [m for m in json.load(open(f, encoding="utf-8")) if (folder / m["file"]).exists()]
    except Exception:
        return []


def main():
    st.title("VietASR demo: nhận dạng tiếng nói tiếng Việt")
    st.caption("Reproduce lại Mô hình VietASR với ít loop và sample hơn (pre-training tự giám sát HuBERT với codebook từ mô hình ASR -> fine-tune CTC). "
               "Có hỗ trợ audio/video, một hoặc nhiều người nói.")

    with st.sidebar:
        st.header("Cài đặt")
        release = st.text_input("Thư mục mô hình (release)", value=os.environ.get("VIETASR_RELEASE", "release"))
        samples_dir = st.text_input("Thư mục clip mẫu (samples)", value=os.environ.get("VIETASR_SAMPLES", "samples"))
        device = st.selectbox("Thiết bị", ["auto", "cpu", "cuda"], index=0)
        mode_label = st.radio("Chế độ", ["Một người nói", "Nhiều người nói (tách người nói)"], index=0)
        multi = mode_label.startswith("Nhiều")
        n_speakers = 0
        spk_method = "auto"
        if multi:
            n_speakers = st.number_input("Số người nói (0 = tự động)", min_value=0, max_value=8, value=0, step=1)
            spk_method = st.selectbox("Embedding người nói", ["auto", "mfcc", "ecapa"], index=0,
                                      help="ecapa cần `pip install speechbrain` (chính xác hơn). mfcc: không cần cài thêm, kém hơn.")
        with st.expander("VAD / phân đoạn"):
            vad_method = st.selectbox("Phương pháp VAD", ["auto", "energy", "silero"], index=0)
            sens = st.slider("Độ nhạy (cao = giữ nhiều tiếng nhỏ)", 0.0, 1.0, 0.5, 0.05)
            min_sil = st.slider("Khoảng lặng tối thiểu để tách đoạn (s)", 0.1, 1.5, 0.30, 0.05)
            max_seg = st.slider("Độ dài đoạn tối đa (s)", 5, 25, 12, 1,
                                help="Mô hình được huấn luyện trên clip 3–15 s; đoạn quá dài sẽ nhận dạng kém")
        batch = st.slider("Batch size", 1, 16, 8)
        st.info("Lưu ý: mô hình không tách được giọng nói bị chồng lấn")

    if not (Path(release) / "model.pt").exists():
        st.error(f"Không tìm thấy `{release}/model.pt`")
        st.stop()
    engine = load_engine(release, device)
    meta = getattr(engine, "meta", {}) or {}
    wer = meta.get("clean_test_wer")
    st.sidebar.success(f"Mô hình: {meta.get('name') or 'release'} {vm.count_params(engine.model) / 1e6:.1f}M params, "
                       + (f"WER test sạch khoảng {wer:.1f}%" if wer else ""))
    if wer and wer > 50:
        st.warning(f"Mô hình này có WER {wer:.0f}% trên tập test, kết quả sẽ còn nhiều lỗi.")

    samples = load_samples(Path(samples_dir))
    c1, c2, c3 = st.columns(3)
    with c1:
        up = st.file_uploader("Tải lên audio / video", type=AUDIO_EXT + VIDEO_EXT)
    rec = None
    with c2:
        if hasattr(st, "audio_input"):
            rec = st.audio_input("hoặc ghi âm trực tiếp")
    sample = None
    with c3:
        if samples:
            labels = ["(không chọn)"] + [f"{m['file']}  ·  {m['duration_s']}s" for m in samples]
            pick = st.selectbox("…hoặc dùng clip mẫu", labels)
            if pick != labels[0]:
                sample = samples[labels.index(pick) - 1]
        else:
            st.caption(f"Chưa có clip mẫu (không thấy `{samples_dir}/samples.json`).")

    src_path, src_name, is_video, ref_text = None, None, False, None
    tmpdir = tempfile.mkdtemp(prefix="vietasr_")
    if up is not None:
        ext = Path(up.name).suffix.lower()
        src_path = os.path.join(tmpdir, "input" + ext)
        Path(src_path).write_bytes(up.getvalue())
        src_name, is_video = up.name, ext.lstrip(".") in VIDEO_EXT
    elif rec is not None:
        src_path = os.path.join(tmpdir, "record.wav")
        Path(src_path).write_bytes(rec.getvalue())
        src_name = "ghi âm"
    elif sample is not None:
        src_path = str(Path(samples_dir) / sample["file"])
        src_name, ref_text = sample["file"], sample.get("reference")

    if src_path is None:
        st.info("Hãy tải một file lên, ghi âm hoặc chọn clip mẫu để bắt đầu.")
        st.stop()

    (st.video if is_video else st.audio)(Path(src_path).read_bytes())
    if ref_text:
        with st.expander("Lời thoại gốc của clip mẫu"):
            st.text(ref_text)

    if st.button("Nhận dạng", type="primary"):
        bar = st.progress(0.0, text="Đang đọc file")
        try:
            wave = vm.load_audio_file(src_path)
        except Exception as e:
            st.error(f"Không đọc được file: {e}")
            st.stop()
        if len(wave) < vm.SR * 0.3:
            st.error("Audio quá ngắn.")
            st.stop()
        rows, info = run_pipeline(engine, wave, "multi" if multi else "single", int(n_speakers), vad_method, sens, min_sil,
                                  max_seg, spk_method, batch, lambda p, t: bar.progress(min(max(p, 0.0), 1.0), text=t))
        bar.empty()
        st.session_state["result"] = {"rows": rows, "info": info, "multi": multi, "name": src_name, "ref": ref_text}

    res = st.session_state.get("result")
    if res and res["name"] == src_name:
        rows, info = res["rows"], res["info"]
        n_spk = len({r["speaker"] for r in rows}) if res["multi"] else 1
        m = st.columns(5)
        m[0].metric("Độ dài", f"{info['audio_sec']:.1f}s")
        m[1].metric("Số đoạn", len(rows))
        m[2].metric("Số người nói", n_spk if res["multi"] else "—")
        m[3].metric("Thời gian xử lý", f"{info['elapsed']:.1f}s")
        m[4].metric("RTF", f"{info['elapsed'] / max(info['audio_sec'], 1e-6):.3f}")
        st.caption(f"VAD: {info['vad']} · embedding người nói: {info['speaker_method']}")
        if res.get("ref") and "\n" not in res["ref"] and rows:        # clip mẫu một người nói -> tính WER thật
            hyp = vm.norm_text(" ".join(r["text"] for r in rows))
            w, _ = vm.wer_corpus([vm.norm_text(res["ref"])], [hyp])
            st.metric("WER so với lời thoại gốc", f"{w:.1f}%")
        if not rows:
            st.warning("Không nhận dạng được nội dung do file không có tiếng nói hoặc không rõ")
        else:
            t1, t2, t3, t4 = st.tabs(["Bản ghi", "Văn bản liền mạch", "Bảng chi tiết", "Dòng thời gian"])
            with t1:
                render_transcript(rows, res["multi"])
            with t2:
                st.text_area("Toàn văn", " ".join(r["text"] for r in rows), height=200)
            with t3:
                table(rows)
            with t4:
                render_timeline(rows, info["audio_sec"], res["multi"])
            srt_rows = rows if res["multi"] else [{**r, "speaker": None} for r in rows]
            d = st.columns(3)
            d[0].download_button("Tải TXT", su.to_txt(srt_rows), file_name="transcript.txt")
            d[1].download_button("Tải SRT (phụ đề)", su.to_srt(srt_rows), file_name="transcript.srt")
            d[2].download_button("Tải JSON", su.to_json(rows), file_name="transcript.json")


main()
