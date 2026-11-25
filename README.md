# VietASR-mini demo (Streamlit) – v2

1. Chạy notebook `vietasr_kaggle_v2.ipynb` trên Kaggle, tải `vietasr_demo_kit.zip` từ tab Output và giải nén
   -> có `release/` (mô hình) và `samples/` (clip mẫu + samples.json).
2. Đặt `release/` và `samples/` cạnh `app.py`, rồi:
   `pip install -r requirements.txt` và `streamlit run app.py`

Chức năng: upload audio/video, ghi âm hoặc chọn clip mẫu -> VAD -> (tùy chọn) tách người nói -> nhận dạng -> bản ghi có mốc thời gian, tải TXT/SRT/JSON.
Giới hạn: đoạn nói chồng không được tách; tách người nói mặc định dùng MFCC (yếu), cài `speechbrain` để dùng ECAPA.
Mẹo: giọng rõ, một người nói một lúc, ít nhạc nền; clip mẫu "de_wer_thap_nhat" là clip mô hình nhận dạng tốt nhất, "trung_binh" mới là chất lượng điển hình.
