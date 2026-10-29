# Day 06: Word2Vec — Efficient Estimation of Word Representations in Vector Space

> **Bài báo gốc:** *Efficient Estimation of Word Representations in Vector Space*  
> **Tác giả:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean  
> **Xuất bản:** Hội nghị *International Conference on Learning Representations (ICLR)*, Scottsdale, Arizona, 2013.  
> **Bài báo mở rộng:** *Distributed Representations of Words and Phrases and their Compositionality* (NeurIPS 2013)  
> **Tài liệu PDF gốc:** [`resources/Day 6 Word2Vec.pdf`](./resources/Day%206%20Word2Vec.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Bài Toán Biểu Diễn Ngôn Ngữ](#2-bối-cảnh-lịch-sử--bài-toán-biểu-diễn-ngôn-ngữ)
3. [Kiến Trúc Mô Hình Chi Tiết: CBOW & Skip-Gram](#3-kiến-trúc-mô-hình-chi-tiết-cbow--skip-gram)
4. [Đột Phá Kỹ Thuật: Negative Sampling & Subsampling](#4-đột-phá-kỹ-thuật-negative-sampling--subsampling)
5. [Đại Số Ngữ Nghĩa Tuyệt Đẹp (Word Analogy Arithmetic)](#5-đại-số-ngữ-nghĩa-tuyệt-đẹp-word-analogy-arithmetic)
6. [Quy Trình Huấn Luyện & Tối Ưu Hóa](#6-quy-trình-huấn-luyện--tối-ưu-hóa)
7. [Kết Quả Thực Nghiệm Trong Bài Báo](#7-kết-quả-thực-nghiệm-trong-bài-báo)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Trước năm 2013, Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing - NLP) bị thống trị bởi các biểu diễn rời rạc: **One-Hot Encoding** và ma trận đếm đồng xuất hiện **Bag-of-Words (BoW)**.
- Mỗi từ trong từ điển kích thước $V$ được mã hóa thành một vector nhị phân thưa thớt có chiều dài $V$, trong đó chỉ có duy nhất một vị trí mang giá trị $1$ và tất cả các vị trí còn lại là $0$.
- **Hạn chế chết người:** Tích vô hướng giữa hai vector One-Hot bất kỳ luôn bằng $0$ ($v_{\text{cat}} \cdot v_{\text{dog}} = 0$). Máy tính hoàn toàn không nhận thức được sự tương đồng giữa *"chó"* và *"mèo"*, xem chúng hoàn toàn xa lạ như *"chó"* với *"tủ lạnh"*.

Năm 2013, **Tomas Mikolov** cùng các nhà khoa học tại Google (Greg Corrado, Kai Chen, Jeff Dean, Ilya Sutskever) đã công bố công trình lịch sử **Word2Vec**:
- Chuyển đổi toàn bộ từ vựng sang một không gian vector thực liên tục, dày đặc (Dense Vector Space) với số chiều thấp ($D \approx 100 - 300$).
- Tự động nắm bắt các mối quan hệ ngữ nghĩa (Semantics) và cú pháp (Syntax) tinh tế mà không cần bất kỳ tri thức ngôn ngữ nào do con người biên soạn.
- Giới thiệu khả năng đại số vector kỳ diệu:  
  $$\vec{v}_{\text{King}} - \vec{v}_{\text{Man}} + \vec{v}_{\text{Woman}} \approx \vec{v}_{\text{Queen}}$$

```text
KỶ NGUYÊN RỜI RẠC (Trước 2013)                     KỶ NGUYÊN WORD2VEC (2013)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
│ One-Hot Encoding:                    │     │ Dense Word Embeddings:                       │
│ dog = [1, 0, 0, ..., 0]  (Chiều 100K)│ ==> │ dog = [ 0.25, -0.12, 0.88, ..., 0.31] (D=300)│
│ cat = [0, 1, 0, ..., 0]              │     │ cat = [ 0.28, -0.09, 0.84, ..., 0.29]       │
│ dot(dog, cat) = 0.0 (Không ngữ nghĩa)│     │ Cosine(dog, cat) = 0.92 (Rất gần gũi!)      │
└──────────────────────────────────────┘     └──────────────────────────────────────────────┘
```

Word2Vec trở thành viên gạch nền móng đầu tiên cho toàn bộ ngành NLP hiện đại, đặt nền tảng cho **GloVe, FastText, ELMo, BERT, và các Mô Hình Ngôn Ngữ Lớn (LLMs)** ngày nay.

---

## 2. Bối Cảnh Lịch Sử & Bài Toán Biểu Diễn Ngôn Ngữ

### 2.1. Giả thuyết Phân bố (Distributional Hypothesis)
Ý tưởng cốt lõi của Word2Vec bắt nguồn từ triết lý ngôn ngữ học của J.R. Firth (1957):  
> *"You shall know a word by the company it keeps"* (Bạn sẽ hiểu ý nghĩa của một từ thông qua các từ thường xuyên xuất hiện xung quanh nó).

Ví dụ: Nếu trong hàng triệu câu văn bản, các từ *"trà"*, *"cà phê"*, *"sữa"* đều thường xuyên xuất hiện cạnh các từ *"uống"*, *"nóng"*, *"cốc"*, thì các từ này phải có biểu diễn hình học gần nhau trong không gian vector.

### 2.2. Điểm nghẽn tính toán của Mạng Ngôn Ngữ Nơ-ron (NNLM)
Yoshua Bengio et al. (2003) từng đề xuất mô hình mạng nơ-ron ngôn ngữ NNLM. Tuy nhiên:
- NNLM sử dụng tầng ẩn phi tuyến tính (Non-linear Hidden Layer) phức tạp.
- Với kích thước từ điển $V \approx 1,000,000$, tầng Softmax đầu ra đòi hỏi tính toán phép tính ma trận khổng lồ:
  $$O(N \times D + N \times D \times H + H \times V)$$
- Mất hàng tuần lễ trên cụm siêu máy tính chỉ để huấn luyện vài triệu từ.

Tomas Mikolov nhận ra rằng: **Để học được biểu diễn từ chất lượng trên tập dữ liệu hàng tỷ từ, phải loại bỏ hoàn toàn tầng ẩn phi tuyến tính**, giữ cấu trúc mạng đơn giản nhất có thể và tập trung vào hiệu năng tính toán!

---

## 3. Kiến Trúc Mô Hình Chi Tiết: CBOW & Skip-Gram

Bài báo đề xuất hai kiến trúc đối xứng phục vụ hai nhu cầu khác nhau:

```text
       KIẾN TRÚC CBOW                                KIẾN TRÚC SKIP-GRAM
 (Dự đoán Từ Trung Tâm từ Ngữ Cảnh)             (Dự đoán Ngữ Cảnh từ Từ Trung Tâm)

      Ngữ cảnh (Context)                              Ngữ cảnh dự đoán
   w(t-2) ──┐                                         ┌──► w(t-2)
   w(t-1) ──┼──► [Chiếu Trung Bình] ──► w(t)          │
   w(t+1) ──┤            (SUM)         (Target) w(t) ─┼──► w(t-1)
   w(t+2) ──┘                                (Input)  ├──► w(t+1)
                                                      └──► w(t+2)
```

### 3.1. Continuous Bag-of-Words (CBOW)
- **Mục tiêu:** Cho trước một cửa sổ ngữ cảnh xung quanh $2C$ từ $\{w_{t-C}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+C}\}$, dự đoán từ trung tâm $w_t$.
- **Cơ chế tính toán:**
  1. Lấy vector biểu diễn của các từ ngữ cảnh từ ma trận nhúng đầu vào $W_{\text{in}} \in \mathbb{R}^{V \times D}$.
  2. Tính trung bình cộng của các vector ngữ cảnh:
     $$\bar{v} = \frac{1}{2C} \sum_{-C \le j \le C, j \neq 0} v_{w_{t+j}}$$
  3. Nhân với ma trận trọng số đầu ra $W_{\text{out}} \in \mathbb{R}^{D \times V}$ để tính điểm số (logits) của tất cả các từ trong từ điển.
- **Ưu điểm:** Huấn luyện rất nhanh, làm mịn dữ liệu tốt, hiệu quả cao với các từ phổ biến.

### 3.2. Continuous Skip-Gram
- **Mục tiêu:** Cho trước từ trung tâm $w_t$, dự đoán xác suất xuất hiện của các từ ngữ cảnh $w_{t+j}$ nằm trong cửa sổ kích thước $C$.
- **Hàm mục tiêu (Log-Likelihood):**
  $$\mathcal{L} = \frac{1}{T} \sum_{t=1}^T \sum_{-C \le j \le C, j \neq 0} \log P(w_{t+j} \mid w_t)$$
- **Ưu điểm:** Hoạt động xuất sắc trên các tập dữ liệu lớn, nắm bắt cực tốt các mối quan hệ ngữ nghĩa của các **từ hiếm gặp (rare words)**.

---

## 4. Đột Phá Kỹ Thuật: Negative Sampling & Subsampling

Nếu sử dụng Softmax thông thường, xác suất dự đoán từ ngữ cảnh $w_O$ khi biết từ trung tâm $w_I$ là:

$$P(w_O \mid w_I) = \frac{\exp({v'_{w_O}}^\top v_{w_I})}{\sum_{w=1}^V \exp({v'_w}^\top v_{w_I})}$$

Mẫu số đòi hỏi tính tổng trên toàn bộ từ điển $V$ (thường từ $100,000$ đến $1,000,000$ từ) tại **mỗi bước lan truyền ngược của từng từ**, khiến việc huấn luyện bất khả thi.

### 4.1. Lấy Mẫu Tiêu Cực (Negative Sampling - NEG)
Lấy cảm hứng từ kỹ thuật Noise Contrastive Estimation (NCE), Mikolov biến bài toán phân loại đa lớp $V$ chiều thành chuỗi các **bài toán hồi quy Logistic nhị phân (Binary Classification)**:
- Cặp từ thực tế $(w_I, w_O)$ xuất hiện cùng nhau trong văn bản được gán nhãn $y = 1$ (Positive Pair).
- Lấy ngẫu nhiên $K$ từ ngẫu nhiên $w_i$ từ từ điển không xuất hiện trong ngữ cảnh và gán nhãn $y = 0$ (Negative Samples).
- **Hàm mất mát Negative Sampling:**
  $$\mathcal{L}_{\text{NEG}} = \log \sigma({v'_{w_O}}^\top v_{w_I}) + \sum_{i=1}^K \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-{v'_{w_i}}^\top v_{w_I}) \right]$$

Trong đó:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ là hàm Sigmoid.
- $K$ thường chọn từ $5 - 20$ (cho tập dữ liệu nhỏ) hoặc $2 - 5$ (cho tập dữ liệu lớn hàng tỷ từ).
- **Độ phức tạp tính toán giảm từ $O(V)$ xuống chỉ còn $O(K)$**, tăng tốc độ huấn luyện hàng nghìn lần!

### 4.2. Phân phối Lấy Mẫu Mũ $3/4$ (The $3/4$ Power Distribution)
Khi chọn các mẫu âm tính $w_i$, nếu lấy theo tần suất xuất hiện tự nhiên $U(w)$, các từ dừng phổ biến như *"the"*, *"is"* sẽ bị chọn liên tục. Nếu lấy ngẫu nhiên đều (uniform), các từ hiếm sẽ bị chọn quá mức.  
Mikolov phát hiện phân phối thực nghiệm tối ưu là lũy thừa $3/4$:

$$P_n(w) = \frac{U(w)^{3/4}}{\sum_{w'} U(w')^{3/4}}$$

Lũy thừa $0.75$ giúp tăng nhẹ xác suất xuất hiện của các từ hiếm mà không làm mất tính đại diện của các từ phổ thông.

### 4.3. Giảm mẫu các từ quá phổ biến (Subsampling of Frequent Words)
Các từ như *"in"*, *"the"*, *"a"* xuất hiện hàng triệu lần nhưng cung cấp rất ít thông tin ngữ nghĩa. Mỗi từ $w$ trong văn bản có xác suất bị loại bỏ (discard) khi nạp vào cửa sổ huấn luyện:

$$P(\text{discard}(w)) = 1 - \sqrt{\frac{t}{f(w)}}$$

Trong đó $f(w)$ là tần suất của từ và ngưỡng $t \approx 10^{-5}$. Kỹ thuật này giúp tăng tốc độ huấn luyện gấp $2 - 10$ lần và cải thiện đáng kể độ chính xác của các từ hiếm.

---

## 5. Đại Số Ngữ Nghĩa Tuyệt Đẹp (Word Analogy Arithmetic)

Một trong những khám phá gây chấn động nhất của Word2Vec là **tính chất cộng tính tuyến tính (Linear Compositionality)** của không gian vector:

```text
       Ý NIỆM QUỐC GIA - THỦ ĐÔ                     Ý NIỆM GIỚI TÍNH & ĐỊA VỊ
   Vector(Spain)          Vector(Madrid)           Vector(King)          Vector(Queen)
        ○ ─────────────────────► ○                      ○ ────────────────────► ○
        ▲                        ▲                      ▲                       ▲
        │                        │                      │                       │
        │                        │                      │                       │
        ○ ─────────────────────► ○                      ○ ────────────────────► ○
   Vector(France)         Vector(Paris)            Vector(Man)           Vector(Woman)
   [Dịch chuyển = "Thủ đô của"]                    [Dịch chuyển = "Nữ tính hóa"]
```

### 5.1. Phép suy luận tương đồng (Analogy Task)
Để trả lời câu hỏi: *"King đối với Man thì tương tự như từ nào đối với Woman?"*:
$$\vec{d} = \vec{v}_{\text{King}} - \vec{v}_{\text{Man}} + \vec{v}_{\text{Woman}}$$
Tìm từ $w^*$ trong toàn bộ từ điển có độ tương đồng Cosine cao nhất với vector $\vec{d}$:
$$w^* = \arg\max_{w \notin \{\text{King}, \text{Man}, \text{Woman}\}} \frac{\vec{d} \cdot \vec{v}_w}{\|\vec{d}\| \|\vec{v}_w\|}$$
Kết quả tìm được chuẩn xác là: **$\text{Queen}$**!

### 5.2. Các mối quan hệ ngữ nghĩa & ngữ pháp được khám phá tự động:
- **Thủ đô - Quốc gia:** $\vec{v}_{\text{Hanoi}} - \vec{v}_{\text{Vietnam}} + \vec{v}_{\text{Japan}} \approx \vec{v}_{\text{Tokyo}}$
- **So sánh hơn nhất:** $\vec{v}_{\text{biggest}} - \vec{v}_{\text{big}} + \vec{v}_{\text{small}} \approx \vec{v}_{\text{smallest}}$
- **Thì quá khứ:** $\vec{v}_{\text{went}} - \vec{v}_{\text{go}} + \vec{v}_{\text{see}} \approx \vec{v}_{\text{saw}}$
- **Công ty - Sản phẩm:** $\vec{v}_{\text{Apple}} - \vec{v}_{\text{iPhone}} + \vec{v}_{\text{Android}} \approx \vec{v}_{\text{Google}}$

---

## 6. Quy Trình Huấn Luyện & Tối Ưu Hóa

- **Kích thước vector nhúng ($D$):** Thường từ $100$ đến $300$ chiều.
- **Kích thước cửa sổ ngữ cảnh ($C$):** $5$ đối với CBOW, $10$ đối với Skip-Gram.
- **Tốc độ học (Learning Rate $\alpha$):** Khởi tạo $\alpha = 0.025$ cho Skip-Gram ($0.05$ cho CBOW) và giảm tuyến tính dần về $0$ trong suốt quá trình duyệt qua văn bản.
- **Số mẫu âm tính ($K$):** $5$ mẫu âm cho mỗi cặp từ thực tế.
- **Cập nhật trọng số:** Lan truyền ngược cập nhật đồng thời ma trận nhúng đầu vào $W_{\text{in}}$ và ma trận trọng số ngữ cảnh đầu ra $W_{\text{out}}$.

---

## 7. Kết Quả Thực Nghiệm Trong Bài Báo

### 7.1. Độ chính xác trên bộ kiểm thử Semantic-Syntactic Word Relationship
Bộ kiểm thử gồm 19,544 câu hỏi suy luận tương đồng do nhóm Mikolov thiết kế (Semantic: 8,869 câu, Syntactic: 10,675 câu):

| Mô Hình | Chiều Vector ($D$) | Kích Thước Dữ Liệu | Semantic Acc (%) | Syntactic Acc (%) | Tổng Độ Chính Xác |
|---|:---:|:---:|:---:|:---:|:---:|
| NNLM (Bengio 2003) | 100 | 6B words | 34.2% | 64.5% | 53.7% |
| RNNLM (Mikolov 2010) | 100 | 6B words | 49.3% | 55.4% | 53.0% |
| **CBOW (Mikolov 2013)** | **300** | **6B words** | **63.6%** | **67.4%** | **66.1%** |
| **Skip-Gram (Mikolov 2013)**| **300** | **6B words** | **65.2%** | **63.0%** | **65.6%** |
| **Skip-Gram + NEG (Google News)** | **300** | **100B words** | **77.2%** | **69.8%** | **72.8%** |

### 7.2. Tốc độ huấn luyện vượt trội
Nhờ loại bỏ tầng ẩn phi tuyến tính và áp dụng Negative Sampling, Word2Vec có thể xử lý **hơn 1 tỷ từ mỗi ngày** trên một máy chủ đa luồng CPU thông thường, điều mà không bất kỳ kiến trúc NNLM nào trước đó có thể làm được.

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Hạn chế cốt tử của Word2Vec nhìn từ hiện đại
1. **Biểu diễn Tĩnh (Static Embeddings):** Mỗi từ chỉ có duy nhất một vector cố định. Từ đa nghĩa như *"Apple"* (quả táo hay tập đoàn công nghệ) bị gộp chung vào một điểm vector duy nhất.
2. **Vấn đề từ vựng ngoài danh mục (Out-Of-Vocabulary - OOV):** Hoàn toàn không thể tạo vector cho các từ mới hoặc từ viết sai chính tả.

### 8.2. Cây phả hệ tiến hóa từ Word2Vec
```text
WORD2VEC (Mikolov 2013) - Vector tĩnh ở cấp độ từ (Word-level static embeddings)
     │
     ├──► GloVe (Pennington et al. 2014) - Kết hợp ma trận đồng xuất hiện toàn cục với thừa số ma trận
     │
     ├──► FastText (Bojanowski et al. 2016) - Phân tách từ thành các n-gram ký tự con (Giải quyết triệt để OOV)
     │
     ├──► ELMo (Peters et al. 2018) - Biểu diễn theo ngữ cảnh động (Contextualized Embeddings via biLM)
     │
     └──► BERT / GPT / Transformer (2017 - nay) - Tự chú ý sâu (Deep Self-Attention) tạo ra biểu diễn ngữ cảnh hoàn hảo.
```

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục cung cấp hệ thống 4 tệp cài đặt mã nguồn chuẩn mực (From Scratch & PyTorch) cùng báo cáo lý thuyết và PDF gốc:

```text
Day 06 - Word2Vec/
├── README.md                      # Báo cáo lý thuyết chuyên sâu 10 mục chuẩn này
├── word2vec_scratch.py            # Cài đặt thuần Python & NumPy (Skip-gram, Negative Sampling, Analogy Arithmetic & Benchmark)
├── word2vec_scratch.ipynb         # Notebook thực hành tương tác trực quan (From Scratch)
├── word2vec_pytorch.py            # Triển khai Word2Vec Skip-Gram PyTorch, chiếu không gian PCA/t-SNE & Benchmark 3 mô hình
├── word2vec_pytorch.ipynb         # Notebook thực hành tương tác & trực quan hóa 4 bảng đối chiếu
├── outputs/                       # Thư mục lưu biểu đồ đối chiếu học thuật & không gian nhúng
│   ├── word2vec_embedding_space.png    # Phép chiếu không gian vector 2D thể hiện các cụm ngữ nghĩa
│   └── word2vec_academic_comparison.png # Biểu đồ 4 bảng phân tích học thuật chuyên sâu
└── resources/
    └── Day 6 Word2Vec.pdf         # Toàn văn bài báo gốc ICLR 2013
```

### 9.1. Tệp `word2vec_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python chuẩn + NumPy (không dùng PyTorch hay thư viện ngoài).
- **Tính năng nổi bật**:
  - Tự xây dựng bộ mã hóa từ vựng `Vocabulary`: ánh xạ từ $\leftrightarrow$ chỉ số, tính tần suất và xây dựng phân phối lấy mẫu âm $P_n(w) \propto U(w)^{3/4}$.
  - Cài đặt mô hình `Word2VecScratch` (Skip-gram với Negative Sampling).
  - Tự lập trình lan truyền ngược giải tích cập nhật ma trận nhúng đầu vào $W_{\text{in}}$ và đầu ra $W_{\text{out}}$.
  - Bộ giải tương đồng ngữ nghĩa `solve_analogy(w1, w2, w3)`.
  - `run_academic_value_comparison()`: **Module đối chiếu giá trị học thuật** so sánh trực diện One-Hot / Co-occurrence SVD với Không gian liên tục Word2Vec.

### 9.2. Tệp `word2vec_pytorch.py` (Triển khai bằng PyTorch)
- **Công nghệ**: PyTorch, matplotlib, scikit-learn (PCA).
- **Tính năng nổi bật**:
  - Lớp mạng `SkipGramModel` sử dụng hai tầng `nn.Embedding` cho vector từ trung tâm và vector ngữ cảnh.
  - Hàm mất mát tối ưu `NegativeSamplingLoss` với Sigmoid Binary Cross-Entropy.
  - Phép chiếu không gian nhúng 2D trực quan hóa các mối quan hệ ngữ nghĩa.
  - `run_academic_value_benchmark()`: Khối đối chuẩn 3 trường phái mô hình:
    1. *Mô Hình 1 (Tiền-Word2Vec / One-Hot & Co-occurrence):* Vector trực giao, không có tính định hướng ngữ nghĩa.
    2. *Mô Hình 2 (Word2Vec Full Softmax):* Điểm nghẽn tính toán $O(V)$, không thể mở rộng từ điển lớn.
    3. *Mô Hình 3 (Word2Vec Negative Sampling):* Đột phá $O(K)$, đại số vector ngữ nghĩa hoàn hảo.
  - `plot_comprehensive_academic_comparison()`: Tự động lưu biểu đồ 4 bảng phân tích vào `outputs/word2vec_academic_comparison.png`.

### 9.3. Bảng Tổng Kết Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark)

| Tiêu Chí So Sánh | Mô Hình 1: One-Hot / SVD (Tiền-2013) | Mô Hình 2: Word2Vec Full-Softmax | Mô Hình 3: Word2Vec Negative Sampling |
|---|:---:|:---:|:---:|
| **Bản chất biểu diễn** | Thưa thớt, trực giao ($V$ chiều) | Dày đặc ($D \approx 100-300$) | **Dày đặc, liên tục ($D \approx 100-300$)** |
| **Độ phức tạp mỗi bước** | $O(V^2)$ phân rã ma trận SVD | $O(V \times D)$ rất đắt đỏ | **$O(K \times D)$ cực kỳ gọn nhẹ ($K \ll V$)** |
| **Khoảng cách ngữ nghĩa** | Không có ($\text{dot}=0$) | Tốt nhưng quá chậm | **Độ tương đồng Cosine phản ánh đúng ngữ nghĩa** |
| **Đại số vector ($King - Man$)**| Hoàn toàn bất khả thi | Có khả năng | **Xuất sắc ($\approx Queen$, bảo toàn phương hướng)** |
| **Quy mô từ vựng hỗ trợ** | Bị nghẽn ở vài chục nghìn từ | Hạn chế ở $V < 50,000$ | **Khả năng mở rộng hàng triệu từ ngữ vựng** |
| **Thời gian huấn luyện** | Chậm, tốn RAM đột biến | Rất lâu (hàng chục giây/epoch) | **Nhanh gấp hàng chục lần** |

### 9.4. Hướng Dẫn Chạy & Thực Hành

```bash
# 1. Chạy mô phỏng From Scratch (Thuần Python + NumPy)
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 06 - Word2Vec/word2vec_scratch.py"

# 2. Chạy mô hình PyTorch & Xuất biểu đồ không gian nhúng + Đối chiếu học thuật
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 06 - Word2Vec/word2vec_pytorch.py"

# 3. Khởi chạy Jupyter Notebook tương tác trực quan
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 06 - Word2Vec/word2vec_scratch.ipynb"
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 06 - Word2Vec/word2vec_pytorch.ipynb"
```

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Phân tích sự khác biệt căn bản giữa biểu diễn từ One-Hot và biểu diễn phân bố (Distributed Representations) trong Word2Vec. Tại sao biểu diễn One-Hot lại thất bại trong việc nắm bắt quan hệ ngữ nghĩa?**
2. **So sánh cơ chế hoạt động, ưu điểm và nhược điểm giữa hai kiến trúc CBOW và Skip-Gram. Trong trường hợp nào thì nên ưu tiên dùng Skip-Gram hơn CBOW?**
3. **Tại sao hàm Softmax thông thường lại là "điểm nghẽn tính toán" lớn nhất khi huấn luyện mô hình ngôn ngữ nơ-ron? Kỹ thuật Negative Sampling đã giải quyết bài toán này như thế nào?**
4. **Ý nghĩa toán học và tác dụng thực nghiệm của số mũ $3/4$ trong phân phối lấy mẫu âm $P_n(w) \propto U(w)^{3/4}$ là gì?**
5. **Kỹ thuật Subsampling of Frequent Words hoạt động như thế nào và tại sao nó lại giúp tăng tốc độ huấn luyện cũng như cải thiện chất lượng vector của các từ hiếm?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
