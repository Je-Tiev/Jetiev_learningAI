# 10 Research Papers Quan Trọng Nhất Về NLP (Natural Language Processing)

Tập hợp 10 bài báo khoa học then chốt định hình lĩnh vực Xử lý Ngôn ngữ Tự nhiên & Mô hình Ngôn ngữ Lớn (LLMs).

> 📖 **Quy chuẩn triển khai:**  
> Toàn bộ các bài báo trong chuyên đề này được định hướng triển khai đồng bộ theo **Quy chuẩn 4 thành phần & Khung lý thuyết 10 mục** đã hoàn thành mẫu tại [`Day 02 - Autoencoder`](../20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2002%20-%20Autoencoder).  
> Chi tiết tiêu chuẩn kỹ thuật xem tại: [Tài liệu hướng dẫn tổng quan `papers-reading/README.md`](../README.md).

---

## Cấu Trúc Chuẩn Cho Từng Bài Báo

Mỗi thư mục bài báo độc lập bao gồm 4 phần:
1. 📄 **`README.md`**: Báo cáo phân tích chuyên sâu 10 mục bằng Tiếng Việt (Ý nghĩa lịch sử, bối cảnh toán học, kiến trúc, đột phá, quy trình huấn luyện, bảng đối chuẩn thực nghiệm, di sản/tiến hóa, hướng dẫn code, câu hỏi ôn tập).
2. ⚙️ **`[model]_scratch.py`**: Tự cài đặt thuật toán từ đầu bằng Python thuần + NumPy (tokenization/attention/adaptation from scratch, kiểm thử toán học).
3. 🚀 **`[model]_pytorch.py`**: Triển khai tối ưu hóa bằng PyTorch (`torch.nn.Module`, fallback dataset an toàn khi offline, trực quan hóa biểu đồ ra thư mục `outputs/`).
4. 📚 **`resources/`**: Chứa tệp bài báo PDF gốc.

---

## Danh Mục Các Bài Báo & Tiến Độ Thực Hiện

| STT | Bài Báo / Chủ Đề | Tiêu Đề Bài Báo Gốc | Thư Mục | Tiến Độ & Trạng Thái |
|:---:|---|---|---|:---:|
| 01 | **T5** | *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* | [`Day 01 - T5/`](./Day%2001%20-%20T5) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 02 | **Scaling Laws** | *Scaling Laws for Neural Language Models* | [`Day 02 - Scaling Laws/`](./Day%2002%20-%20Scaling%20Laws) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 03 | **LoRA** | *LoRA: Low-Rank Adaptation of Large Language Models* | [`Day 03 - LoRA/`](./Day%2003%20-%20LoRA) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 04 | **Self-Consistency CoT** | *Self-Consistency Improves Chain of Thought Reasoning in Language Models* | [`Day 04 - Self-Consistency improves CoT/`](./Day%2004%20-%20Self-Consistency%20improves%20CoT) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 05 | **Calibration of LMs** | *Language Models (Mostly) Know What They Know* | [`Day 05 - Language Models know what they know/`](./Day%2005%20-%20Language%20Models%20know%20what%20they%20know) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 06 | **Instruction Tuning** | *Finetuned Language Models are Zero-Shot Learners* | [`Day 06 - Instruction Tuning/`](./Day%2006%20-%20Instruction%20Tuning) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 07 | **Toolformer** | *Language Models Can Teach Themselves to Use Tools* | [`Day 07 - Toolformer/`](./Day%2007%20-%20Toolformer) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 08 | **ColBERTv2** | *Effective and Efficient Retrieval via Lightweight Late Interaction* | [`Day 08 - ColBERTv2/`](./Day%2008%20-%20ColBERTv2) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 09 | **LLM-as-a-Judge** | *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [`Day 09 - LLM-as-a-judge/`](./Day%2009%20-%20LLM-as-a-judge) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 10 | **DeepSeek-R1** | *Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* | [`Day 10 - DeepSeek-R1/`](./Day%2010%20-%20DeepSeek-R1) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
