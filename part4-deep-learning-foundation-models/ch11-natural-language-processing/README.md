# Chương 11: Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Nắm vững bức tranh tổng quan của NLP:** Phân loại các tác vụ NLP chính (Phân loại văn bản, NER, Dịch máy, Tóm tắt, Mô hình ngôn ngữ) và ứng dụng thực tiễn.
2. **Làm chủ các phương pháp biểu diễn văn bản cổ điển:** Hiểu rõ Bag-of-Words (BoW), công thức toán học của **TF-IDF**, mô hình $n$-gram, phân tích ngữ nghĩa tiềm ẩn (LSA qua SVD) và mô hình chủ đề LDA.
3. **Thấu hiểu NLP thống kê:** Mô hình ngôn ngữ $n$-gram xác suất và Mô hình Markov ẩn (HMM) giải mã từ loại bằng giải thuật Viterbi.
4. **Làm chủ Biểu diễn phân tán (Word Embeddings):** Hiểu Giả thuyết phân phối (Distributional Hypothesis), kiến trúc **Word2Vec** (CBOW và Skip-gram với Negative Sampling), **GloVe**, **FastText** (Subword modeling) và giới hạn của vector nhúng tĩnh trước hiện tượng từ đa nghĩa.
5. **Thành thạo Mạng nơ-ron chuỗi (Sequence Models):** Phân tích kiến trúc RNN cơ bản, thuật toán BPTT, nguyên nhân toán học của hiện tượng Triệt tiêu/Bùng nổ gradient và cơ chế 3 cổng của **LSTM** cùng 2 cổng của **GRU**.
6. **Làm chủ Kiến trúc Bộ mã hóa - Bộ giải mã (Seq2Seq / Encoder-Decoder):** Hiểu cơ chế ánh xạ chuỗi sang chuỗi có độ dài biến thiên, biểu diễn xác suất tự hồi quy (Autoregressive probability) và phân tích "Nút thắt cổ chai thông tin" (Information Bottleneck) – động cơ ra đời của Transformer.

---

## 11.1 Giới thiệu về Xử lý Ngôn ngữ Tự nhiên (Introduction to NLP)

### 11.1.1 Khái niệm và Thách thức
Xử lý ngôn ngữ tự nhiên (NLP) là lĩnh vực giao thoa giữa Khoa học máy tính, Trí tuệ nhân tạo và Ngôn ngữ học, nhằm mục đích giúp máy tính có khả năng đọc, hiểu, suy luận và sinh ra ngôn ngữ con người một cách tự nhiên.
- **Những thách thức bản chất của ngôn ngữ:**
  1. *Tính nhập nhằng (Ambiguity):* Nhập nhằng từ vựng (từ đồng âm khác nghĩa), nhập nhằng cú pháp (cấu trúc câu nhiều cách hiểu), nhập nhằng ngữ dụng (nói mỉa mai, ẩn dụ).
  2. *Tính rời rạc (Discreteness):* Ngôn ngữ cấu tạo từ các ký hiệu rời rạc (từ, ký tự), không có tính liên tục tự nhiên như điểm ảnh hay sóng âm.
  3. *Tính biến thiên và tiến hóa:* Ngữ pháp thay đổi theo ngữ cảnh văn hóa, tiếng lóng, phương ngữ.

### 11.1.2 Các tác vụ cốt lõi trong NLP
- **Phân loại văn bản (Text Classification):** Phân loại chủ đề tin tức, lọc thư rác, phân tích cảm xúc (Sentiment Analysis: Tích cực / Tiêu cực).
- **Gán nhãn thực thể có tên (Named Entity Recognition - NER):** Trích xuất tên người, địa điểm, tổ chức, ngày tháng từ văn bản thô.
- **Dịch máy (Machine Translation):** Chuyển ngữ tự động giữa các ngôn ngữ tự nhiên.
- **Tóm tắt văn bản (Summarization):** Rút gọn văn bản dài thành đoạn tóm lược giữ nguyên ý chính (Trích xuất - Extractive vs. Trừu tượng - Abstractive).
- **Mô hình hóa ngôn ngữ (Language Modeling):** Ước lượng phân phối xác suất của chuỗi từ $P(w_1, w_2, \dots, w_m)$ hoặc dự đoán từ tiếp theo $P(w_t \mid w_{<t})$ – nền tảng cốt lõi của các LLMs hiện đại.

---

## 11.2 Biểu diễn văn bản Cổ điển (Classical Text Representations)

Trước kỷ nguyên Deep Learning, máy tính không thể hiểu trực tiếp các chuỗi ký tự văn bản mà phải chuyển đổi chúng thành các vector số học rời rạc.

### 11.2.1 Túi từ (Bag-of-Words - BoW)
- Xây dựng một từ điển gồm $|V|$ từ vựng duy nhất xuất hiện trong toàn bộ kho ngữ liệu (Corpus).
- Mỗi văn bản được biểu diễn bằng một vector $\mathbf{x} \in \mathbb{R}^{|V|}$, trong đó phần tử $x_i$ là tần số xuất hiện (hoặc biến nhị phân $0/1$) của từ thứ $i$ trong văn bản đó.
- **Nhược điểm cốt tử:** Bỏ qua hoàn toàn thứ tự cú pháp của từ ("Không thích nhưng vẫn xem" có cùng biểu diễn BoW với "Thích nhưng vẫn không xem"); ma trận biểu diễn cực kỳ thưa (Sparse).

### 11.2.2 Tần số từ - Nghịch đảo tần số văn bản (TF-IDF)
BoW có nhược điểm là những từ xuất hiện quá nhiều lần (như "và", "là", "của", "the", "is") sẽ chiếm ưu thế áp đảo dù chúng mang rất ít giá trị phân biệt nội dung. **TF-IDF** giải quyết vấn đề này bằng cách kết hợp:
1. **Tần số từ (Term Frequency - $\text{TF}$):** Đo lường mức độ quan trọng của từ $t$ trong một văn bản $d$:
   $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
2. **Nghịch đảo tần số văn bản (Inverse Document Frequency - $\text{IDF}$):** Phạt các từ xuất hiện phổ biến ở khắp mọi văn bản trong tập ngữ liệu $D$:
   $$\text{IDF}(t, D) = \log\left( \frac{|D|}{|\{d \in D : t \in d\}| + 1} \right)$$
3. **Chỉ số TF-IDF tổng hợp:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
   - Từ có điểm TF-IDF cao nhất là từ xuất hiện nhiều lần trong văn bản khảo sát, nhưng lại hiếm khi xuất hiện ở các văn bản khác trong kho ngữ liệu.

### 11.2.3 Mô hình $n$-gram
Thay vì chỉ đếm từng từ đơn lẻ ($1$-gram / Unigram), mô hình $n$-gram đếm chuỗi gồm $n$ từ liên tiếp:
- Bigram ($n=2$): "trí tuệ", "nhân tạo", "học máy".
- Trigram ($n=3$): "trí tuệ nhân tạo", "mạng nơ ron".
- **Ưu điểm:** Giữ lại được ngữ cảnh cục bộ ngắn; **Nhược điểm:** Kích thước từ điển bùng nổ theo hàm mũ $|V|^n$.

### 11.2.4 Phân tích Ngữ nghĩa Tiềm ẩn (Latent Semantic Analysis - LSA)
Áp dụng phân rã giá trị suy biến (SVD) lên ma trận Từ - Văn bản (Term-Document Matrix) $\mathbf{X} \in \mathbb{R}^{|V| \times |D|}$:
$$\mathbf{X} \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$
- Giảm số chiều từ $|V|$ xuống $k$ khái niệm tiềm ẩn ($k \ll |V|$), giúp gom các từ đồng nghĩa (Synonyms) về gần nhau trong không gian vector liên tục.

### 11.2.5 Mô hình Chủ đề (Topic Modeling - LDA)
**Latent Dirichlet Allocation (Blei et al., 2003):** Mô hình sinh xác suất giả định:
- Mỗi văn bản là một phân phối hỗn hợp trên tập hợp các chủ đề tiềm ẩn: $\boldsymbol{\theta}_d \sim \text{Dirichlet}(\boldsymbol{\alpha})$.
- Mỗi chủ đề là một phân phối trên các từ vựng: $\boldsymbol{\phi}_k \sim \text{Dirichlet}(\boldsymbol{\beta})$.
- Sử dụng thuật toán lấy mẫu Gibbs (Gibbs Sampling) hoặc biến phân Bayes để tự động trích xuất các chủ đề ngầm trong hàng ngàn bài báo.

---

## 11.3 NLP Thống kê (Statistical NLP)

### 11.3.1 Mô hình Ngôn ngữ $n$-gram Thống kê
Ước lượng xác suất của một câu gồm $m$ từ dựa trên giả định Markov bậc $n-1$:
$$P(w_1, w_2, \dots, w_m) = \prod_{i=1}^m P(w_i \mid w_1, \dots, w_{i-1}) \approx \prod_{i=1}^m P(w_i \mid w_{i-n+1}, \dots, w_{i-1})$$
- Ước lượng hợp lý cực đại (MLE):
  $$P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$
- **Vấn đề Zero Probability:** Nếu một cụm từ chưa từng xuất hiện trong tập huấn luyện, $\text{Count} = 0 \implies$ Toàn bộ xác suất của câu bị nhân về 0. Khắc phục bằng các kỹ thuật làm mịn (Smoothing: Laplace add-one, Kneser-Ney Smoothing).

### 11.3.2 Mô hình Markov Ẩn (Hidden Markov Model - HMM) trong Gán nhãn Từ loại
- Trạng thái ẩn $S_t \in \{N, V, Adj, \dots\}$ là nhãn từ loại (POS Tag).
- Quan sát $O_t \in V$ là từ thực tế phát ra.
- Sử dụng **Thuật toán Viterbi** (quy hoạch động) để tìm chuỗi nhãn ẩn có xác suất sinh ra chuỗi từ cao nhất:
  $$S^* = \arg\max_{S_1, \dots, S_T} P(S_1) \prod_{t=1}^T P(O_t \mid S_t) \prod_{t=2}^T P(S_t \mid S_{t-1})$$

---

## 11.4 Biểu diễn phân tán (Word Embeddings)

```
        One-Hot Encoding                   Word Embeddings (Phân tán)
     (Rời rạc, trực giao, thưa)               (Dày đặc, mang ngữ nghĩa)
   King  : [1, 0, 0, 0, 0, 0]                King  : [ 0.85,  0.92, -0.12,  0.45]
   Queen : [0, 1, 0, 0, 0, 0]                Queen : [ 0.82, -0.88, -0.10,  0.43]
   Man   : [0, 0, 1, 0, 0, 0]                Man   : [ 0.70,  0.89, -0.35, -0.10]
   Woman : [0, 0, 0, 1, 0, 0]                Woman : [ 0.68, -0.85, -0.32, -0.12]
   => d(King, Queen) = d(King, Apple)        => v(King) - v(Man) + v(Woman) ≈ v(Queen)
```

### 11.4.1 Giả thuyết Phân phối (Distributional Hypothesis)
Được nhà ngôn ngữ học J.R. Firth phát biểu năm 1957:
> *"You shall know a word by the company it keeps"* (Bạn sẽ biết nghĩa của một từ thông qua các từ láng giềng đi kèm với nó).

### 11.4.2 Word2Vec (Mikolov et al., 2013 - Google)
Chuyển hóa biểu diễn từ One-hot thành các vector thực có số chiều thấp và mật độ cao ($\mathbf{v}_w \in \mathbb{R}^{d}$, thường $d \in [100, 300]$).
1. **Continuous Bag of Words (CBOW):** Dự đoán từ đích ở giữa $w_t$ dựa trên các từ ngữ cảnh xung quanh $w_{t-c}, \dots, w_{t+c}$. Thích hợp cho tập dữ liệu lớn, tốc độ nhanh.
2. **Skip-gram:** Dự đoán các từ ngữ cảnh xung quanh $w_{t-c}, \dots, w_{t+c}$ từ một từ đích $w_t$ duy nhất ở giữa. Nắm bắt các từ hiếm tốt hơn nhiều.
3. **Kỹ thuật Lấy mẫu Âm tính (Negative Sampling):** Thay vì tính mẫu số Softmax trên toàn bộ từ vựng $|V|$ (tốn kém $\mathcal{O}(|V|)$), ta xấp xỉ bằng bài toán phân loại nhị phân giữa từ ngữ cảnh thực ($y=1$) và $k$ từ ngữ cảnh ngẫu nhiên được rút ngẫu nhiên ($y=0$).

### 11.4.3 GloVe và FastText
- **GloVe (Pennington et al., 2014 - Stanford):** Tận dụng trực tiếp tỷ số xác suất đồng xuất hiện từ trong ma trận toàn cục (Global Co-occurrence Matrix) để học vector nhúng thông qua hàm mất mát bình phương trọng số.
- **FastText (Bojanowski et al., 2017 - Facebook):** Coi mỗi từ là tập hợp các $n$-gram ký tự (Subwords). Ví dụ từ "apple" gồm: `<ap`, `app`, `ppl`, `ple`, `le>`. Vector của từ là tổng vector của các subwords.
  - *Đột phá:* Xử lý xuất sắc các từ chưa từng thấy trong từ điển (**Out-Of-Vocabulary - OOV**) và các ngôn ngữ biến đổi hình thái phong phú (Morphologically rich languages).

### 11.4.4 Giới hạn cốt tử của Vector Nhúng Tĩnh
Các mô hình nhúng từ tĩnh (Word2Vec, GloVe, FastText) gán **duy nhất một vector cố định** cho mỗi từ trong suốt thời gian tồn tại. Chúng hoàn toàn thất bại trước các **từ đa nghĩa (Polysemy)**:
- Từ "ngân hàng" trong câu *"Tôi đi rút tiền ở ngân hàng"* và *"Ngồi câu cá bên ngân hàng bờ sông"* đều nhận chung một vector! Sự bất cập này đòi hỏi sự ra đời của **Vector nhúng theo ngữ cảnh động (Contextualized Embeddings)** trong ELMo, BERT và Transformer.

---

## 11.5 Mạng Nơ-ron Chuỗi (Recurrent Neural Networks - RNN)

### 11.5.1 Kiến trúc RNN cơ bản
Mạng RNN duy trì một **trạng thái ẩn (Hidden State)** $\mathbf{h}_t \in \mathbb{R}^d$ đóng vai trò như một bộ nhớ lưu trữ thông tin của toàn bộ các bước thời gian từ quá khứ đến hiện tại:
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
$$\hat{\mathbf{y}}_t = \text{Softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$

```
          y1             y2             yt
           ▲              ▲              ▲
           │              │              │
        ┌──┴──┐        ┌──┴──┐        ┌──┴──┐
 h0 ──► │ RNN │ ─────► │ RNN │ ─────► │ RNN │ ──► ht
        └──┬──┘   h1   └──┬──┘   h2   └──┬──┘
           ▲              ▲              ▲
           │              │              │
          x1             x2             xt
```

### 11.5.2 Lan truyền ngược qua Thời gian (BPTT) và Vấn đề Triệt tiêu Gradient
- Để tính đạo hàm tại bước thời gian $t$ ngược về bước thời gian đầu tiên $1$, ta phải nhân liên tục các đạo hàm chuyển trạng thái:
  $$\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \prod_{k=2}^t \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \prod_{k=2}^t \text{diag}(1 - \tanh^2(\mathbf{z}_k)) \mathbf{W}_{hh}^T$$
- **Hậu quả:**
  1. Nếu các trị riêng của $\mathbf{W}_{hh}$ nhỏ hơn 1 hoặc do đạo hàm của $\tanh < 1$, tích lũy chuỗi làm gradient bị suy giảm theo hàm mũ $\implies$ **Triệt tiêu Gradient (Vanishing Gradient)**. Mạng hoàn toàn "quên" mất thông tin ở đầu câu khi câu dài quá 10 từ.
  2. Nếu các trị riêng của $\mathbf{W}_{hh}$ lớn hơn 1, gradient tăng vọt theo hàm mũ $\implies$ **Bùng nổ Gradient (Exploding Gradient)**, khiến tham số bị biến dạng thành giá trị `NaN`. (Khắc phục bằng kỹ thuật cắt tỉa gradient - Gradient Clipping).

---

## 11.6 Mạng Hồi quy có Cổng (LSTM & GRU)

### 11.6.1 LSTM (Long Short-Term Memory - Hochreiter & Schmidhuber, 1997)
LSTM hóa giải triệt tiêu gradient bằng cách đưa vào **Trạng thái ô (Cell State $\mathbf{c}_t$)** đóng vai trò như một "đại lộ thông tin" tuyến tính chạy dọc theo thời gian, được điều khiển bởi 3 cổng logic đóng mở bằng hàm Sigmoid:

```
                      ct-1 ──────────────────────[+]────────► ct
                                                  ▲
                                                  │ (it * c~t)
                            ┌─── [x] ◄────────────┤
                            │     ▲               │
                       (ft) │     │               │ (c~t)
                            │   ┌─┴─┐   ┌───┐   ┌─┴─┐
                            σ   │ σ │   │ σ │   │tanh
                            │   └───┘   └───┘   └───┘
                            ▲     ▲       ▲       ▲
                            │     │       │       │
             ht-1 ──────────┴─────┴───────┴───────┴──────────► ht
              xt  ───────────────────────────────────────────►
```

#### Bốn phương trình cốt lõi của LSTM:
1. **Cổng quên (Forget Gate $\mathbf{f}_t$):** Quyết định bao nhiêu phần trăm thông tin cũ trong $\mathbf{c}_{t-1}$ sẽ bị vứt bỏ:
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
2. **Cổng vào (Input Gate $\mathbf{i}_t$) & Ứng viên nhớ mới ($\tilde{\mathbf{c}}_t$):** Quyết định thông tin mới nào sẽ được nạp vào bộ nhớ:
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i), \qquad \tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
3. **Cập nhật Cell State:** Cộng dồn tuyến tính (không bị bão hòa gradient):
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
4. **Cổng ra (Output Gate $\mathbf{o}_t$) & Trạng thái ẩn $\mathbf{h}_t$:**
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o), \qquad \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

### 11.6.2 GRU (Gated Recurrent Unit - Cho et al., 2014)
- Gộp chung Cell State và Hidden State thành $\mathbf{h}_t$ duy nhất.
- Tinh giản chỉ còn **2 cổng**: Cổng cập nhật (Update Gate $\mathbf{z}_t$) và Cổng đặt lại (Reset Gate $\mathbf{r}_t$):
  $$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t]), \qquad \mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$
  $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]), \qquad \mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$
- **Ưu điểm:** Ít hơn $25\%$ tham số so với LSTM, tốc độ huấn luyện nhanh hơn và hiệu năng tương đương trên nhiều bài toán chuỗi.

---

## 11.7 Kiến trúc Bộ mã hóa - Bộ giải mã (Encoder-Decoder / Seq2Seq)

Được đề xuất độc lập bởi Cho et al. (2014) và Sutskever et al. (2014) để giải quyết bài toán ánh xạ chuỗi đầu vào $\mathbf{x} = (x_1, \dots, x_{T_x})$ sang chuỗi đầu ra $\mathbf{y} = (y_1, \dots, y_{T_y})$ có độ dài **hoàn toàn khác nhau** (như Dịch máy, Tóm tắt văn bản).

```
  ENCODER                                     DECODER
  "Tôi"     "yêu"     "AI"                    "I"       "love"     "AI"      <EOS>
    │         │         │                      ▲          ▲          ▲         ▲
    ▼         ▼         ▼                      │          │          │         │
 ┌─────┐   ┌─────┐   ┌─────┐  Vector Ngữ cảnh┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
 │ RNN │──►│ RNN │──►│ RNN │────── v ───────►│ RNN │──►│ RNN │──►│ RNN │──►│ RNN │
 └─────┘   └─────┘   └─────┘                 └─────┘   └─────┘   └─────┘   └─────┘
                                                ▲         ▲         ▲
                                                │         │         │
                                              <SOS>      "I"      "love"
```

1. **Bộ mã hóa (Encoder):** Đọc tuần tự chuỗi nguồn và nén toàn bộ thông tin ngữ nghĩa vào một **Vector ngữ cảnh (Context Vector $\mathbf{v} = \mathbf{h}_{T_x}$)** duy nhất.
2. **Bộ giải mã (Decoder):** Khởi tạo trạng thái ẩn từ $\mathbf{v}$ và tự hồi quy (Autoregressive) sinh ra từng từ đích $\hat{y}_t$:
   $$P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{T_y} P(y_t \mid y_1, \dots, y_{t-1}, \mathbf{v})$$
3. **Nút thắt cổ chai Thông tin (Information Bottleneck):**
   - Bắt buộc toàn bộ ý nghĩa của một câu dài 50 từ phải bị nén chặt vào một vector có kích thước cố định (ví dụ 512 chiều). Khi câu nguồn dài hơn 20 từ, hiệu năng dịch thuật của Seq2Seq sụp đổ nghiêm trọng vì Decoder không thể giải mã chính xác các chi tiết ở đầu câu.
   - **Lối thoát lịch sử:** Sự bất lực của Vector Ngữ cảnh cố định đã trực tiếp thúc đẩy Dzmitry Bahdanau (2014) phát minh ra **Cơ chế Chú ý (Attention Mechanism)** – tiền đề dẫn tới cuộc cách mạng **Transformer** năm 2017.

---

## 11.8 Tài liệu tham khảo (References)

1. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo: `resources/Deep Learning.pdf`] (Chương 10: Sequence Modeling: Recurrent and Recursive Nets).
2. **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).** Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*. (Word2Vec).
3. **Pennington, J., Socher, R., & Manning, C. D. (2014).** GloVe: Global vectors for word representation. *EMNLP*, 1532-1543. (GloVe).
4. **Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory. *Neural Computation*, 9(8), 1735-1780. (LSTM).
5. **Cho, K., et al. (2014).** Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*, 1724-1734. (GRU & Seq2Seq).
6. **Sutskever, I., Vinyals, O., & Le, Q. V. (2014).** Sequence to sequence learning with neural networks. *NeurIPS*, 27. (Seq2Seq).
7. **Bahdanau, D., Cho, K., & Bengio, Y. (2014).** Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*. (Attention đầu tiên cho Seq2Seq).

---

## 11.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng bài thực hành xử lý ngôn ngữ tự nhiên từ kinh điển đến hiện đại:

1. **Thành phần 1 - Tiền xử lý văn bản và TF-IDF từ đầu (from scratch):**
   - Viết hàm Tokenizer, xây dựng bảng từ điển (Vocabulary), tính toán tần số $\text{TF}$, $\text{IDF}$ và ma trận $\text{TF-IDF}$ chỉ bằng `NumPy`.
2. **Thành phần 2 - Mô hình Word2Vec Skip-gram rút gọn từ đầu:**
   - Xây dựng mạng nơ-ron 2 tầng trọng số $\mathbf{W}_{\text{in}}$ và $\mathbf{W}_{\text{out}}$ học vector nhúng từ cho một tập câu nhỏ.
   - Minh họa không gian vector nhúng sau huấn luyện và kiểm chứng khoảng cách ngữ nghĩa giữa các từ tương đồng.
3. **Thành phần 3 - Cài đặt Tầng RNN tuần hoàn cơ bản (Vanilla RNN from scratch):**
   - Cài đặt công thức $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$ trên một chuỗi thời gian.
   - Minh họa hiện tượng triệt tiêu gradient qua nhiều bước thời gian.
4. **Thành phần 4 - Cài đặt Tầng LSTM hoàn chỉnh:**
   - Lập trình đầy đủ các cổng Forget, Input, Output và cập nhật Cell State $\mathbf{c}_t$.
   - Chứng minh dòng chảy Cell State bảo toàn thông tin nhớ dài hạn vượt trội so với RNN truyền thống.
