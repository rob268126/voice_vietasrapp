# VietASR demo

Bài báo: [`VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data and Large-Scale Speech Pretraining`](https://arxiv.org/pdf/2505.21527)  
Bộ dataset: [`nguyenvulebinh/ViCocktail`](https://huggingface.co/datasets/nguyenvulebinh/ViCocktail)  

# Các bước chạy
1. Chạy notebook `vietasr_kaggle_v2.ipynb` (Kaggle), tải `vietasr_demo_kit.zip` từ Output và giải nén
   -> có `release/` (mô hình) và `samples/` (clip mẫu + samples.json).
2. Đặt `release/` và `samples/` cạnh `app.py`, sau đó:
   `pip install -r requirements.txt` và `streamlit run app.py`

# Thông tin về webapp
Chức năng: upload audio/video, ghi âm hoặc chọn clip mẫu -> VAD -> (tùy chọn) tách người nói -> nhận dạng -> bản ghi có mốc thời gian, tải TXT/SRT/JSON.  
Hạn chế: đoạn nói chồng không được tách; tách người nói mặc định dùng MFCC (yếu), cài `speechbrain` để dùng ECAPA.  
