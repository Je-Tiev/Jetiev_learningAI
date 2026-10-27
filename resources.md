# Tài Liệu Tham Khảo & Học Liệu Bổ Trợ (Resources & References)

Tài liệu này tổng hợp toàn bộ sách giáo trình, tài liệu tham khảo, bảng tra cứu nhanh (cheatsheets), bài báo khoa học và công cụ phục vụ cho lộ trình học tập Trí tuệ Nhân tạo trong kho lưu trữ **Jetiev_learningAI**.

---

## 📌 Mục Lục
1. [Sách Giáo Khoa Cốt Lõi](#1-sách-giáo-khoa-cốt-lõi)
2. [Tủ Sách Tham Khảo Chuyên Sâu (Classic Textbooks)](#2-tủ-sách-tham-khảo-chuyên-sâu-classic-textbooks)
3. [Bảng Tra Cứu Nhanh (Cheatsheets)](#3-bảng-tra-cứu-nhanh-cheatsheets)
4. [Chuyên Đề Đọc & Triển Khai Bài Báo Nghiên Cứu (AI Papers Reading)](#4-chuyên-đề-đọc--triển-khai-bài-báo-nghiên-cứu-ai-papers-reading)
5. [Các Khóa Học Trực Tuyến Uy Tín](#5-các-khóa-học-trực-tuyến-uy-tín)
6. [Công Cụ & Môi Trường Phát Triển](#6-công-cụ--môi-trường-phát-triển)
7. [Nguồn Tra Cứu Bài Báo & Diễn Đàn Thảo Luận](#7-nguồn-tra-cứu-bài-báo--diễn-đàn-thảo-luận)

---

## 1. Sách Giáo Khoa Cốt Lõi

Toàn bộ khung chương trình từ `part1` đến `part5` của dự án được biên soạn và cấu trúc bám sát giáo trình:

- **Tên sách:** *AI hiện đại từ nguyên lý cổ điển đến mô hình nền tảng*
- **Tác giả:** GS.TS Thái Trà My – PGS.TS Nguyễn Phi Lê (Đại học Bách Khoa Hà Nội & VinAI)
- **Phạm vi kiến thức:** 
  - Phần 1: Nền tảng của trí tuệ nhân tạo (Tác tử, tìm kiếm không gian trạng thái, biểu diễn tri thức & lập luận).
  - Phần 2: Cơ sở toán học cho học máy (Đại số tuyến tính, xác suất & thống kê, giải tích ma trận & tối ưu hóa).
  - Phần 3: Học máy (Nguyên lý cơ bản, mô hình cổ điển, mạng nơ-ron truyền thẳng & lan truyền ngược).
  - Phần 4: Học sâu & Các mô hình nền tảng (Thị giác máy tính, Xử lý ngôn ngữ tự nhiên, Transformer, Pre-training, AI tạo sinh).
  - Phần 5: AI đáng tin cậy & Trách nhiệm (Minh bạch/XAI, An toàn/Bảo mật, Thiên kiến/Công bằng, Đạo đức & Quản trị AI).

---

## 2. Tủ Sách Tham Khảo Chuyên Sâu (Classic Textbooks)

Kho tài liệu lưu trữ cục bộ tại thư mục [`resources/Book/`](./resources/Book) chứa các đầu sách kinh điển hàng đầu thế giới về Khoa học Máy tính và Trí tuệ Nhân tạo:

| STT | Tên Sách & Tác Giả | Nhà Xuất Bản | Tệp Cục Bộ | Ứng Dụng Trong Lộ Trình |
|:---:|---|---|---|---|
| 01 | **Mathematics for Machine Learning**<br>Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong | Cambridge Univ. Press | [`Mathematics for Machine learning.pdf`](./resources/Book/Mathematics%20for%20Machine%20learning.pdf) | Nền tảng toán học trực tiếp cho **Part 2** (Đại số tuyến tính, SVD, Tối ưu hóa lồi, Giải tích ma trận). |
| 02 | **Pattern Recognition and Machine Learning (PRML)**<br>Christopher M. Bishop | Springer | [`Pattern Recognition and Machine Learning.pdf`](./resources/Book/Pattern%20Recognition%20and%20Machine%20Learning.pdf) | Giáo trình chuẩn mực về Học máy xác suất (Bayesian Machine Learning), mô hình đồ thị và EM algorithm (**Part 2 & Part 3**). |
| 03 | **The Elements of Statistical Learning (ESL)**<br>Trevor Hastie, Robert Tibshirani, Jerome Friedman | Springer | [`The Elements of Statistical Learning.pdf`](./resources/Book/The%20Elements%20of%20Statistical%20Learning.pdf) | Lý thuyết học thống kê nghiêm ngặt (Bias-Variance tradeoff, SVM, Regularization, Boosting) (**Part 3**). |
| 04 | **Deep Learning** (The MIT Press Book)<br>Ian Goodfellow, Yoshua Bengio, Aaron Courville | MIT Press | [`Deep Learning.pdf`](./resources/Book/Deep%20Learning.pdf) | Cuốn "Kinh thánh của Deep Learning" bao quát từ MLP, CNN, Sequence Modeling, Autoencoder đến Generative Models (**Part 3 & Part 4**). |
| 05 | **Machine Learning: A Probabilistic Perspective**<br>Kevin P. Murphy | MIT Press | [`Machine Learning A Probabilistic Perspective.pdf`](./resources/Book/Machine%20Learning%20A%20Probabilistic%20Perspective.pdf) | Bách khoa toàn thư toàn diện kết hợp cả học máy cổ điển và học sâu qua lăng kính mô hình hóa xác suất. |
| 06 | **Machine Learning Cơ Bản**<br>Vũ Hữu Tiệp | machinelearningcoban.com | [`Machine Learning Co Ban.pdf`](./resources/Book/Machine%20Learning%20Co%20Ban.pdf) | Tài liệu diễn giải tiếng Việt trực quan, kèm mã nguồn Python mẫu cho các thuật toán nền tảng. |
| 07 | **Machine Learning**<br>Tom M. Mitchell | McGraw-Hill | [`Machine Learning - Tom Mitchell.pdf`](./resources/Book/Machine%20Learning%20-%20Tom%20Mitchell.pdf) | Giáo trình kinh điển định hình định nghĩa học máy, không gian giả thuyết và lý thuyết học tính toán (**Part 1 & Part 3**). |
| 08 | **Machine Learning in Python**<br>Michael Bowles | Wiley | [`Machine Learning in Python.pdf`](./resources/Book/Machine%20Learning%20in%20Python.pdf) | Thực hành trích chọn đặc trưng (feature engineering) và xây dựng pipeline học máy thực chiến với Python. |

---

## 3. Bảng Tra Cứu Nhanh (Cheatsheets)

Bộ sưu tập 16 bảng tra cứu tóm tắt công thức, cú pháp và kiến trúc đặt tại [`resources/Cheatsheet/`](./resources/Cheatsheet):

### 3.1. Toán Học & Xác Suất Thống Kê
- [`Probability Cheat sheet Havard.pdf`](./resources/Cheatsheet/Probability%20Cheat%20sheet%20Havard.pdf): Tóm lược toàn bộ phân phối xác suất, định lý kỳ vọng, phương sai, biến đổi ngẫu nhiên từ Đại học Harvard.

### 3.2. Ngôn Ngữ Python & Xử Lý Dữ Liệu
- [`PythonBasic Cheat sheet 02.pdf`](./resources/Cheatsheet/PythonBasic%20Cheat%20sheet%2002.pdf): Cú pháp cốt lõi, kiểu dữ liệu, hàm và cấu trúc điều khiển Python.
- [`Numpy Cheat sheet 01.pdf`](./resources/Cheatsheet/Numpy%20Cheat%20sheet%2001.pdf) & [`Numpy Cheat sheet 02.pdf`](./resources/Cheatsheet/Numpy%20Cheat%20sheet%2002.pdf): Các phép toán ma trận, mảng nhiều chiều, broadcasting và đại số tuyến tính trong NumPy.
- [`Pandas Cheat sheet 01.pdf`](./resources/Cheatsheet/Pandas%20Cheat%20sheet%2001.pdf): Thao tác DataFrame, làm sạch dữ liệu, pivot table, groupby và xử lý chuỗi thời gian.
- [`PySpark Cheat sheet 02.pdf`](./resources/Cheatsheet/PySpark%20Cheat%20sheet%2002.pdf): Xử lý dữ liệu lớn (Big Data) phân tán với Spark DataFrame và RDD.

### 3.3. Trực Quan Hóa Dữ Liệu
- [`PythonMatplotlib Cheat sheet 02.pdf`](./resources/Cheatsheet/PythonMatplotlib%20Cheat%20sheet%2002.pdf): Tùy biến hình vẽ, subplots, màu sắc và trực quan hóa dữ liệu đa chiều với Matplotlib.
- [`Seaborn Cheat sheet 01.pdf`](./resources/Cheatsheet/Seaborn%20Cheat%20sheet%2001.pdf): Vẽ biểu đồ phân phối thống kê, heatmap và tương quan đặc trưng nhanh chóng.

### 3.4. Học Máy Cổ Điển (Classical Machine Learning)
- [`Scikit-Learn Algorithm 02.pdf`](./resources/Cheatsheet/Scikit-Learn%20Algorithm%2002.pdf): Bản đồ tư duy chọn lựa thuật toán Scikit-Learn tối ưu theo dạng bài toán (Phân loại, Hồi quy, Gom cụm, Giảm chiều).
- [`Scikit-Learn Cheat sheet 01.pdf`](./resources/Cheatsheet/Scikit-Learn%20Cheat%20sheet%2001.pdf) & [`Scikit-Learn Cheat sheet 02.pdf`](./resources/Cheatsheet/Scikit-Learn%20Cheat%20sheet%2002.pdf): Quy trình tiền xử lý, phân chia tập dữ liệu, cross-validation, huấn luyện và đánh giá mô hình.

### 3.5. Học Sâu & Khung Tính Toán Nơ-ron (Deep Learning Frameworks)
- [`NeuralNetworks Cheat sheet.pdf`](./resources/Cheatsheet/NeuralNetworks%20Cheat%20sheet.pdf): Cấu trúc trực quan của các loại mạng nơ-ron (Perceptron, CNN, RNN, LSTM, Autoencoder, GAN,...).
- [`Pytorch Cheat sheet 02.pdf`](./resources/Cheatsheet/Pytorch%20Cheat%20sheet%2002.pdf): Cú pháp `torch.Tensor`, `autograd`, `nn.Module`, vòng lặp huấn luyện, di chuyển thiết bị GPU/CPU.
- [`TensorFlow Cheat sheet 02.pdf`](./resources/Cheatsheet/TensorFlow%20Cheat%20sheet%2002.pdf): Đồ thị tính toán, `tf.data`, `tf.function` và tối ưu hóa hiệu năng TensorFlow.
- [`Keras Cheat sheet 01.pdf`](./resources/Cheatsheet/Keras%20Cheat%20sheet%2001.pdf) & [`Keras Cheat sheet 02.pdf`](./resources/Cheatsheet/Keras%20Cheat%20sheet%2002.pdf): Xây dựng mô hình Sequential và Functional API, callbacks, fine-tuning.

---

## 4. Chuyên Đề Đọc & Triển Khai Bài Báo Nghiên Cứu (AI Papers Reading)

Thư mục [`papers-reading/`](./papers-reading) được tổ chức thành 3 chuyên đề độc lập gồm tổng cộng 40 công trình nghiên cứu bản lề:

1. **[20 AI Research Papers Quan Trọng Nhất Thế Kỷ 21](./papers-reading/20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021)**: Các cột mốc khai sinh kỷ nguyên Machine Learning & Deep Learning (Random Forest, Autoencoder, AlexNet, Word2Vec, VAE, GAN, ResNet, Transformer, BERT, GPT-3, ViT,...).
2. **[10 Research Papers Quan Trọng Nhất Về NLP](./papers-reading/10%20research%20paper%20quan%20trọng%20nhất%20về%20NLP)**: Các đột phá mô hình ngôn ngữ lớn (T5, Scaling Laws, LoRA, Chain-of-Thought, Toolformer, DeepSeek-R1,...).
3. **[10 Research Papers Quan Trọng Nhất Về RAG](./papers-reading/10%20research%20paper%20quan%20trọng%20nhất%20về%20RAG)**: Các kiến trúc nâng cao độ chính xác truy xuất và tri thức bên ngoài (RAG Foundations, Self-RAG, Corrective RAG, GraphRAG, Agentic RAG, ReAct,...).

> 💎 **Chuẩn mực cấu trúc mẫu (Benchmark Template):**  
> Mọi bài báo trong hệ thống đều tuân thủ theo tiêu chuẩn 4 phần khép kín đã hoàn thành tại [`Day 02 - Autoencoder`](./papers-reading/20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2002%20-%20Autoencoder):
> - `README.md`: Báo cáo lý thuyết chuyên sâu 10 phần bằng tiếng Việt kèm công thức toán học và sơ đồ ASCII.
> - `*_scratch.py`: Cài đặt thuần Python/NumPy từ con số 0 (hiểu sâu gradient & giải thuật).
> - `*_pytorch.py`: Triển khai chuẩn hóa trên PyTorch với pipeline huấn luyện, so sánh baseline và xuất biểu đồ trực quan.
> - `resources/`: Chứa tệp bài báo gốc định dạng PDF.

---

## 5. Các Khóa Học Trực Tuyến Uy Tín

- **[DeepLearning.AI / Andrew Ng — Machine Learning Specialization & Deep Learning Specialization](https://www.deeplearning.ai/):** Khóa học nền tảng toàn diện nhất cho người bắt đầu đến nâng cao.
- **[CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/):** Giáo trình đại học Stanford về mạng tích chập và thị giác máy tính hiện đại.
- **[CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/):** Giáo trình đại học Stanford về biểu diễn từ, Transformers và LLMs.
- **[Fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/):** Phương pháp tiếp cận từ trên xuống (top-down), chú trọng thực nghiệm nhanh và triển khai ứng dụng thực tế.
- **[Full Stack Deep Learning](https://fullstackdeeplearning.com/):** Đào tạo kỹ năng triển khai mô hình học sâu ra môi trường production (MLOps, Docker, CI/CD).

---

## 6. Công Cụ & Môi Trường Phát Triển

| Công Cụ | Phiên Bản Khuyến Nghị | Vai Trò & Mục Đích Trong Dự Án |
|---|:---:|---|
| **Python** | `>= 3.10` | Ngôn ngữ lập trình cốt lõi cho toàn bộ dự án. |
| **NumPy** | `>= 1.24` | Tính toán số học mảng nhiều chiều, đại số tuyến tính và cài đặt thuật toán from-scratch. |
| **pandas** | `>= 2.0` | Thao tác, làm sạch và chuẩn bị các tập dữ liệu bảng. |
| **scikit-learn** | `>= 1.3` | Các giải thuật Học máy cổ điển, trích xuất đặc trưng và các độ đo đánh giá. |
| **PyTorch & Torchvision** | `>= 2.0` | Xây dựng đồ thị tính toán động, huấn luyện mạng nơ-ron sâu trên GPU/CPU. |
| **Matplotlib & Seaborn** | Khuyến nghị mới nhất | Trực quan hóa dữ liệu, vẽ biểu đồ không gian tiềm ẩn (latent space), loss curve. |
| **Jupyter Notebook** | Mới nhất | Môi trường thử nghiệm tương tác và trình bày trực quan (`demo.ipynb`). |

---

## 7. Nguồn Tra Cứu Bài Báo & Diễn Đàn Thảo Luận

### Cổng tra cứu học thuật
- **[arXiv.org](https://arxiv.org/):** Cổng lưu trữ các bản thảo nghiên cứu (preprints) mở cập nhật từng ngày về AI, Computer Vision, Machine Learning, NLP.
- **[Papers with Code](https://paperswithcode.com/):** Nguồn tra cứu bài báo khoa học đi kèm mã nguồn mở thực nghiệm và bảng xếp hạng benchmark (SOTA).
- **[Google Scholar](https://scholar.google.com/):** Theo dõi số lượng trích dẫn học thuật và mạng lưới bài viết liên quan.
- **[Connected Papers](https://www.connectedpapers.com/):** Công cụ trực quan hóa đồ thị mối quan hệ giữa các công trình nghiên cứu.

### Diễn đàn & Cộng đồng
- **[Hugging Face Hub & Forums](https://discuss.huggingface.co/):** Hệ sinh thái mô hình mã nguồn mở, bộ dữ liệu và trao đổi kỹ thuật NLP/Vision.
- **[Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/):** Diễn đàn thảo luận nghiên cứu học máy quốc tế.
- **[PyTorch Discussion Forums](https://discuss.pytorch.org/):** Hỗ trợ gỡ lỗi và giải đáp chuyên sâu về triển khai PyTorch.
