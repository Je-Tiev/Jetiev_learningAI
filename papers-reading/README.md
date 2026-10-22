# Tổng Hợp & Đọc Các Bài Báo Nghiên Cứu AI (AI Papers Reading)

Kho lưu trữ và phân tích chuyên sâu các bài báo nghiên cứu khoa học (Research Papers) mang tính bước ngoặt định hình nên kỷ nguyên Trí tuệ Nhân tạo hiện đại.

Mỗi nghiên cứu được chuẩn hóa cấu trúc thành một **thư mục độc lập** bao gồm 4 thành phần thiết yếu:
1. **`README.md`**: Tài liệu phân tích lý thuyết chuyên sâu, bối cảnh lịch sử, công thức toán học và giải thích chi tiết, trực quan bằng tiếng Việt.
2. **`*_scratch.py`**: Cài đặt mô hình từ con số 0 (from scratch) bằng Python thuần / NumPy để thấu hiểu bản chất giải thuật và lan truyền gradient.
3. **`*_pytorch.py`**: Cài đặt mô hình chuẩn hóa sử dụng thư viện hiện đại (PyTorch), hỗ trợ huấn luyện, đánh giá và trực quan hóa thực nghiệm.
4. **`resources/`**: Thư mục lưu trữ bài báo gốc (PDF) và các tài nguyên đính kèm.

---

## Danh Mục Các Chuyên Đề Nghiên Cứu

| Chuyên Đề | Mô Tả | Số Lượng Bài Báo | Đường Dẫn |
|---|---|:---:|---|
| **20 AI Research Papers Quan Trọng Nhất Thế Kỷ 21** | Các cột mốc định hình Deep Learning: Autoencoder, CNN, GAN, Transformer, BERT, GPT, ViT,... | 19 bài | [`20 AI Research Paper quan trọng nhất của TK 21/`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021) |
| **10 Research Papers Quan Trọng Nhất Về NLP** | Các đột phá xử lý ngôn ngữ tự nhiên: T5, Scaling Laws, LoRA, CoT, Instruction Tuning, DeepSeek-R1,... | 10 bài | [`10 research paper quan trọng nhất về NLP/`](./10%20research%20paper%20quan%20trọng%20nhất%20về%20NLP) |
| **10 Research Papers Quan Trọng Nhất Về RAG** | Các kiến trúc Truy xuất Tăng cường Tạo sinh: Self-RAG, Corrective RAG, GraphRAG, Agentic RAG, ReAct,... | 10 bài | [`10 research paper quan trọng nhất về RAG/`](./10%20research%20paper%20quan%20trọng%20nhất%20về%20RAG) |

---

## Cấu Trúc Tiêu Chuẩn Cho Mỗi Bài Báo

```text
Day XX - [Tên Bài Báo]/
├── README.md               # Lý thuyết chi tiết, bối cảnh, kiến trúc, toán học & giải thích bằng tiếng Việt
├── [model]_scratch.py      # Code from scratch thuần NumPy / Python (hiểu sâu bản chất toán & gradient)
├── [model]_pytorch.py      # Code triển khai bằng PyTorch (huấn luyện, trực quan hóa, baseline)
└── resources/              # Bài báo PDF gốc & tài liệu hỗ trợ
    └── Day XX [Tên].pdf
```
