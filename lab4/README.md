# DS201 Lab 4 — Encoder–Decoder và Attention

Lab 4 thực hành dịch máy Anh–Việt trên `small-PhoMT` với ba mô hình LSTM:

- Encoder–Decoder không attention
- Bahdanau attention
- Luong attention

Các mô hình được đánh giá bằng ROUGE-L và BLEU.

## Cấu trúc repository

```text
lab4/
├── README.md
├── assignments.ipynb          # Notebook đề bài và điều phối thí nghiệm
├── Lab 4.pdf                  # Tài liệu hướng dẫn
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── __main__.py            # Chạy từ terminal
│   ├── config.py              # Tham số và seed
│   ├── data.py                # Đọc, tiền xử lý và tạo batch
│   ├── model.py               # Encoder, decoder và attention
│   ├── decode.py              # Greedy decoding
│   ├── metrics.py             # ROUGE-L và BLEU
│   ├── train.py               # Huấn luyện và đánh giá
│   ├── experiment.py          # Điều phối thí nghiệm
│   └── artifacts.py           # Lưu CSV, đồ thị và metadata
├── small-PhoMT/               # Không lưu trên Git
│   ├── small-train.json
│   ├── small-dev.json
│   └── small-test.json
└── outputs/                   # Kết quả runtime, bị loại khỏi Git
    ├── assignment_1_no_attention/
    ├── assignment_2_bahdanau/
    ├── assignment_3_luong/
    └── comparison.csv
```

`small-PhoMT/` và `outputs/` chỉ là thư mục runtime local. Không commit dữ liệu, checkpoint hoặc kết quả sinh tự động vào repository.

Mỗi thư mục thí nghiệm trong `outputs/` có thể chứa `logs/`, `metrics/`, `plots/`, `checkpoints/` và `config.json`. Thí nghiệm Bahdanau tạo thêm attention heatmaps.

## Chuẩn bị dữ liệu

Tải [small-PhoMT](https://drive.google.com/drive/folders/186OAOuSEYEDVcry7WP5UBdqECXo26QAb) và đặt ba file JSON vào một thư mục có tên `small-PhoMT/`. Mỗi bản ghi cần có hai trường `english` và `vietnamese`.

Trên Kaggle, attach dataset vào notebook. Trên Colab, upload dữ liệu hoặc mount Google Drive rồi đặt `DATA_DIR` trong notebook trỏ tới thư mục chứa ba file JSON.

## Cài đặt

Yêu cầu Python 3.12 hoặc tương thích với PyTorch trong `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Chạy bằng terminal

Từ thư mục gốc của lab:

```bash
python -m src --data-dir ./small-PhoMT --output-dir ./outputs --evaluate-test
```

Chạy riêng từng mô hình:

```bash
python -m src --data-dir ./small-PhoMT --model none --evaluate-test
python -m src --data-dir ./small-PhoMT --model bahdanau --evaluate-test
python -m src --data-dir ./small-PhoMT --model luong --evaluate-test
```


