# 10 Research Papers Quan Trọng Nhất Về RAG (Retrieval-Augmented Generation)

Tập hợp 10 bài báo khoa học tiên phong đặt nền móng và định hình các hệ thống RAG & AI Agent hiện đại.

> 📖 **Quy chuẩn triển khai:**  
> Toàn bộ các bài báo trong chuyên đề này được định hướng triển khai đồng bộ theo **Quy chuẩn 4 thành phần & Khung lý thuyết 10 mục** đã hoàn thành mẫu tại [`Day 02 - Autoencoder`](../20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2002%20-%20Autoencoder).  
> Chi tiết tiêu chuẩn kỹ thuật xem tại: [Tài liệu hướng dẫn tổng quan `papers-reading/README.md`](../README.md).

---

## Cấu Trúc Chuẩn Cho Từng Bài Báo

Mỗi thư mục bài báo độc lập bao gồm 4 phần:
1. 📄 **`README.md`**: Báo cáo phân tích chuyên sâu 10 mục bằng Tiếng Việt (Ý nghĩa lịch sử, bối cảnh toán học, kiến trúc, đột phá, quy trình huấn luyện/truy xuất, bảng đối chuẩn thực nghiệm, di sản/tiến hóa, hướng dẫn code, câu hỏi ôn tập).
2. ⚙️ **`[model]_scratch.py`**: Tự cài đặt thuật toán từ đầu bằng Python thuần + NumPy (BM25/vector search/self-reflection from scratch, kiểm thử logic).
3. 🚀 **`[model]_pytorch.py`**: Triển khai tối ưu hóa bằng PyTorch/Modern Framework (`torch.nn.Module`, fallback dataset an toàn khi offline, trực quan hóa luồng hoạt động ra thư mục `outputs/`).
4. 📚 **`resources/`**: Chứa tệp bài báo PDF gốc.

---

## Danh Mục Các Bài Báo & Tiến Độ Thực Hiện

| STT | Bài Báo / Chủ Đề | Tiêu Đề Bài Báo Gốc | Thư Mục | Tiến Độ & Trạng Thái |
|:---:|---|---|---|:---:|
| 01 | **RAG Foundations** | *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | [`Day 01 - RAG for Knowledge-Intensive NLP Tasks/`](./Day%2001%20-%20RAG%20for%20Knowledge-Intensive%20NLP%20Tasks) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 02 | **RAG for LLMs** | *Retrieval-Augmented Generation for Large Language Models: A Survey* | [`Day 02 - Retrieval-Augmented Generation for LLMs/`](./Day%2002%20-%20Retrieval-Augmented%20Generation%20for%20LLMs) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 03 | **Self-RAG** | *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* | [`Day 03 - Self-RAG/`](./Day%2003%20-%20Self-RAG) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 04 | **Corrective RAG (CRAG)** | *Corrective Retrieval Augmented Generation* | [`Day 04 - Corrective RAG/`](./Day%2004%20-%20Corrective%20RAG) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 05 | **Adaptive RAG** | *Adaptive RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity* | [`Day 05 - Adaptive RAG Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity/`](./Day%2005%20-%20Adaptive%20RAG%20Learning%20to%20Adapt%20Retrieval-Augmented%20Large%20Language%20Models%20through%20Question%20Complexity) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 06 | **Adaptive RAG Conversational** | *Adaptive RAG for Conversational Systems* | [`Day 06 - Adaptive RAG for Conversational Systems/`](./Day%2006%20-%20Adaptive%20RAG%20for%20Conversational%20Systems) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 07 | **GraphRAG** | *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* | [`Day 07 - GraphRAG/`](./Day%2007%20-%20GraphRAG) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 08 | **Agentic RAG Survey** | *Agentic Retrieval-Augmented Generation: A Survey* | [`Day 08 - Agentic Retrieval-Augmented Generation A Survey/`](./Day%2008%20-%20Agentic%20Retrieval-Augmented%20Generation%20A%20Survey) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 09 | **Reasoning RAG** | *Reasoning RAG* | [`Day 09 - Reasoning RAG/`](./Day%2009%20-%20Reasoning%20RAG) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 10 | **ReAct** | *ReAct: Synergizing Reasoning and Acting in Language Models* | [`Day 10 - ReAct/`](./Day%2010%20-%20ReAct) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
