# Chương 12: Mô hình Bộ biến đổi (Transformer Models)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Hiểu rõ giới hạn của RNN/CNN và động lực ra đời của Transformer:** Phân tích sự vượt trội của độ dài đường dẫn tín hiệu $\mathcal{O}(1)$ và khả năng song song hóa hoàn toàn trên GPU so với tính tuần tự $\mathcal{O}(N)$ của RNN.
2. **Làm chủ Cơ chế Chú ý (Self-Attention & Multi-Head Attention):** Dẫn xuất trọn vẹn công thức toán học Scaled Dot-Product Attention, giải thích vai trò của hệ số tỷ lệ $\frac{1}{\sqrt{d_k}}$ và cơ chế trích xuất đa không gian của Multi-Head Attention.
3. **Hiểu sâu cơ chế Mã hóa Vị trí (Positional Encoding):** Nắm vững công thức hình sin/cosin tuần hoàn của Vaswani et al. (2017) và các cải tiến hiện đại như Rotary Position Embedding (RoPE).
4. **Hệ thống hóa toàn bộ khối kiến trúc Transformer:** Phân tích Layer Normalization (Pre-LN vs. Post-LN, RMSNorm), mạng FFN hai tầng (hoặc SwiGLU), và phân biệt 3 nhánh kiến trúc: Encoder-only (BERT), Decoder-only (GPT) và Encoder-Decoder (T5).
5. **Nắm vững Vision Transformer (ViT):** Hiểu cách thức "biến ảnh thành từ ngữ" qua cơ chế chia Patch, Token phân loại `[CLS]`, và đối chiếu thiên kiến quy nạp (Inductive Bias) giữa ViT và CNN.

---

## 12.1 Hạn chế của RNN/CNN và Sự cần thiết của Transformer

Trước năm 2017, các mô hình chuỗi (RNN, LSTM, GRU) là chuẩn mực thống trị trong NLP. Tuy nhiên, chúng vấp phải hai rào cản kỹ thuật nghiêm trọng:
1. **Nút thắt cổ chai tính toán tuần tự (Sequential Bottleneck):** Để tính toán trạng thái ẩn $\mathbf{h}_t$, mạng RNN bắt buộc phải đợi bước $\mathbf{h}_{t-1}$ tính xong. Quá trình này hoàn toàn không thể song song hóa trên các phần cứng hiện đại như GPU/TPU, làm chậm tiến độ huấn luyện trên dữ liệu lớn.
2. **Suy hao ngữ cảnh dài (Long-range Dependency Decay):** Mặc dù LSTM có Cell State, tín hiệu ngữ nghĩa giữa hai token cách xa nhau $N$ bước vẫn phải đi qua $N$ phép biến đổi phi tuyến liên tiếp.

| Kiến trúc | Độ phức tạp mỗi tầng | Chiều dài đường dẫn tối đa (Path Length) | Khả năng song song hóa |
| :--- | :---: | :---: | :---: |
| **RNN / LSTM** | $\mathcal{O}(N \cdot d^2)$ | $\mathcal{O}(N)$ (Tuần tự từng bước) | **Không** (Tuần tự bắt buộc) |
| **CNN (Dilated Conv)** | $\mathcal{O}(K \cdot N \cdot d^2)$ | $\mathcal{O}(\log_K N)$ (Cây phân cấp) | **Có** |
| **Self-Attention** | $\mathcal{O}(N^2 \cdot d)$ | $\mathbf{\mathcal{O}(1)}$ **(Trực tiếp tức thì)** | **Hoàn hảo (Toàn bộ chuỗi song song)** |

> Trong cơ chế Self-Attention, bất kỳ token nào cũng có thể "nói chuyện trực tiếp" với bất kỳ token nào khác trong câu chỉ trong **duy nhất 1 bước tính toán** ($\mathcal{O}(1)$), bất chấp khoảng cách giữa chúng là 2 từ hay 2000 từ!

---

## 12.2 Cơ chế Tự chú ý (Self-Attention Mechanism)

### 12.2.1 Ý tưởng cốt lõi: Bộ ba Query, Key, Value
Cơ chế Chú ý lấy cảm hứng từ hệ thống truy xuất thông tin (Retrieval System):
- **Query ($\mathbf{Q}$):** Đại diện cho từ hiện tại đang muốn tìm kiếm thông tin ngữ cảnh liên quan.
- **Key ($\mathbf{K}$):** Đại diện cho "nhãn danh mục" hoặc đặc tính của tất cả các từ trong câu.
- **Value ($\mathbf{V}$):** Chứa nội dung thông tin thực sự của các từ.

Cho chuỗi đầu vào gồm $N$ vector nhúng $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}}$, ta chiếu tuyến tính bằng 3 ma trận trọng số học được:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \qquad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \qquad \mathbf{V} = \mathbf{X} \mathbf{W}^V$$
trong đó $\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ và $\mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

```
             Q               K
             │               │
             └───────┬───────┘
                     ▼
                 Tích vô hướng: Q * K^T
                     │
                     ▼
                 Chia cho: √d_k  (Scale)
                     │
                     ▼
                 Mặt nạ (Masking - Tùy chọn)
                     │
                     ▼
                 Softmax (Trọng số chú ý α)
                     │
                     ├──────────────► V
                     ▼                │
               Nhân có trọng số: α * V
                     │
                     ▼
                  Output
```

### 12.2.2 Công thức Toán học Scaled Dot-Product Attention
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

#### Tại sao bắt buộc phải chia cho hệ số tỷ lệ $\sqrt{d_k}$?
Giả sử các thành phần của $\mathbf{q}$ và $\mathbf{k}$ là các biến ngẫu nhiên độc lập có kỳ vọng bằng 0 và phương sai bằng 1: $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \text{Var}(q_i) = \text{Var}(k_i) = 1$.  
Tích vô hướng $z = \mathbf{q}^T \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ sẽ có:
$$\mathbb{E}[z] = 0, \qquad \text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
- Khi số chiều $d_k$ lớn (ví dụ $d_k = 64$ hoặc $128$), độ lớn của tích vô hướng sẽ dao động rất mạnh trong khoảng $[-\sqrt{d_k}, \sqrt{d_k}]$.
- Khi $z$ quá lớn, hàm Softmax bị đẩy sâu vào các vùng bão hòa cực đoan (xác suất tiệm cận 0 hoặc 1), nơi mà **đạo hàm của Softmax triệt tiêu về 0**, làm đình trệ hoàn toàn quá trình cập nhật gradient!
- Việc chia cho $\sqrt{d_k}$ kéo phương sai của giá trị đầu vào Softmax trở về đúng bằng $1$, duy trì dòng chảy gradient ổn định.

### 12.2.3 Cơ chế Chú ý Đa đầu (Multi-Head Attention - MHA)
Thay vì chỉ tính một hàm Attention duy nhất với chiều $d_{\text{model}}$, Multi-Head Attention chia nhỏ không gian biểu diễn thành $h$ "đầu chú ý" (Attention Heads) độc lập (thường $h = 8$ hoặc $16$, với $d_k = d_v = d_{\text{model}} / h$):
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$
$$\text{với } \text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V)$$
- **Ý nghĩa ngôn ngữ học:** Mỗi Attention Head sẽ chuyên môn hóa vào một loại quan hệ ngữ nghĩa khác nhau: Head 1 bắt quan hệ Cú pháp (Chủ ngữ - Vị ngữ); Head 2 bắt quan hệ Đại từ thay thế ("nó" trỏ tới danh từ nào); Head 3 bắt quan hệ Thời gian / Không gian.

---

## 12.3 Mã hóa Vị trí (Positional Encoding)

Do phép toán Self-Attention chỉ tính toán dựa trên tích vô hướng giữa các vector mà không hề có cấu trúc vòng lặp hay tích chập, nó hoàn toàn **bất biến với phép hoán vị (Permutation Invariant)**. Một câu bị xáo trộn trật tự từ lộn xộn sẽ cho ra tập biểu diễn y hệt câu gốc! Để mô hình nhận biết được trật tự trước sau, ta bắt buộc phải cộng thêm thông tin vị trí vào từng vector từ:
$$\mathbf{X}_{\text{input}} = \text{Embedding}(\mathbf{X}) + \mathbf{PE}$$

### 12.3.1 Mã hóa Vị trí Hình sin (Sinusoidal Positional Encoding)
Vaswani et al. (2017) đề xuất công thức giải tích sử dụng các tần số sóng sin và cosin khác nhau:
$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$
$$PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$
Trong đó:
- $pos \in [0, N-1]$ là vị trí của token trong câu.
- $i \in [0, d_{\text{model}}/2 - 1]$ là chỉ số chiều vector.
- **Tính chất hình học ưu việt:** Với mọi khoảng cách cố định $k$, vector vị trí $PE_{pos + k}$ có thể biểu diễn như một phép quay tuyến tính từ $PE_{pos}$:
  $$PE_{pos + k} = \mathbf{R}_k \cdot PE_{pos}$$
  giúp mô hình dễ dàng học cách chú ý tới các vị trí tương đối một cách tổng quát, kể cả với các độ dài câu chưa từng thấy trong tập huấn luyện.

### 12.3.2 Các cải tiến hiện đại: RoPE (Rotary Position Embedding)
Trong các mô hình LLM tiên tiến hàng đầu (LLaMA 1/2/3, Mistral, Gemini), mã hóa hình sin cộng trực tiếp được thay thế bằng **RoPE (Su et al., 2021)**: RoPE nhân trực tiếp ma trận xoay góc $\mathbf{R}_{\Theta, m}$ vào vector Query và Key, giúp tích vô hướng $\mathbf{q}_m^T \mathbf{k}_n$ tự nhiên phụ thuộc vào khoảng cách tương đối $m - n$.

---

## 12.4 Kiến trúc Toàn diện của Transformer

```
   ┌────────────────────────────────┐       ┌────────────────────────────────┐
   │        ENCODER BLOCK           │       │        DECODER BLOCK           │
   │                                │       │                                │
   │   Input                        │       │   Output (Shifted right)       │
   │     │                          │       │     │                          │
   │     ▼                          │       │     ▼                          │
   │   Positional Encoding          │       │   Positional Encoding          │
   │     │                          │       │     │                          │
   │     ▼                          │       │     ▼                          │
   │ ┌────────────────────────────┐ │       │ ┌────────────────────────────┐ │
   │ │ Multi-Head Self-Attention  │ │       │ │ Masked Self-Attention      │ │
   │ └─────────────┬──────────────┘ │       │ └─────────────┬──────────────┘ │
   │               ▼                │       │               ▼                │
   │     Add & LayerNorm            │       │     Add & LayerNorm            │
   │               │                │       │               │                │
   │               ├────────────────┼───────┼──────────────►│ (Cross-Attn)   │
   │               ▼                │       │ ┌─────────────┴──────────────┐ │
   │ ┌────────────────────────────┐ │       │ │ Multi-Head Cross-Attention │ │
   │ │ Feed-Forward Network (FFN) │ │       │ └─────────────┬──────────────┘ │
   │ └─────────────┬──────────────┘ │       │               ▼                │
   │               ▼                │       │     Add & LayerNorm            │
   │     Add & LayerNorm            │       │               │                │
   │                                │       │               ▼                │
   │                                │       │ ┌────────────────────────────┐ │
   │                                │       │ │ Feed-Forward Network (FFN) │ │
   │                                │       │ └─────────────┬──────────────┘ │
   │                                │       │               ▼                │
   │                                │       │     Add & LayerNorm            │
   └────────────────────────────────┘       └────────────────────────────────┘
```

### 12.4.1 Chuẩn hóa tầng (Layer Normalization - LayerNorm)
Khác với BatchNorm (chuẩn hóa dọc theo chiều Batch), **LayerNorm** chuẩn hóa ngang qua toàn bộ các đặc trưng của duy nhất một token:
$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}, \qquad \mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$
- **Pre-LN vs. Post-LN:** Transformer gốc dùng Post-LN ($\mathbf{x}_{l} = \text{LN}(\mathbf{x}_{l-1} + \text{SubLayer}(\mathbf{x}_{l-1}))$), đòi hỏi warmup tốc độ học rất cẩn thận. Các LLM hiện đại đều chuyển sang **Pre-LN** ($\mathbf{x}_l = \mathbf{x}_{l-1} + \text{SubLayer}(\text{LN}(\mathbf{x}_{l-1}))$) hoặc **RMSNorm** (loại bỏ việc trừ trung bình $\mu$) giúp ổn định huấn luyện mạng sâu hàng trăm tầng.

### 12.4.2 Mạng Truyền thẳng Từng Vị trí (Position-Wise FFN)
Nằm sau khối Attention, biến đổi độc lập và giống nhau trên từng token:
$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$
Thường mở rộng số chiều lên 4 lần ($d_{\text{ff}} = 4 \times d_{\text{model}}$) rồi co trở lại. Trong các LLM hiện đại (như LLaMA), FFN được nâng cấp thành biến thể **SwiGLU**:
$$\text{SwiGLU}(\mathbf{x}) = (\text{Swish}(\mathbf{x}\mathbf{W}) \odot (\mathbf{x}\mathbf{V})) \mathbf{W}_2$$

### 12.4.3 Ba họ Kiến trúc Transformer chính:
1. **Chỉ gồm Bộ mã hóa (Encoder-Only / Autoencoding):** Tiêu biểu: **BERT**, **RoBERTa**. Nhìn thấy ngữ cảnh hai chiều (Bidirectional), chuyên dụng cho trích xuất đặc trưng, phân loại văn bản, NER.
2. **Chỉ gồm Bộ giải mã (Decoder-Only / Autoregressive):** Tiêu biểu: **GPT-1/2/3/4**, **LLaMA**, **Claude**. Sử dụng Mặt nạ nhân quả (**Causal Masking**) che đi các token tương lai, chuyên dụng cho sinh ngôn ngữ tự nhiên.
3. **Cả Bộ mã hóa và Bộ giải mã (Encoder-Decoder / Seq2Seq):** Tiêu biểu: **T5**, **BART**. Chuyên dụng cho dịch máy và tóm tắt văn bản.

---

## 12.5 Vision Transformer (ViT - Dosovitskiy et al., 2020)

Bài báo lịch sử *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* của Google Research đã chứng minh: Kiến trúc Transformer thuần túy có thể áp dụng trực tiếp lên hình ảnh và đánh bại hoàn toàn các mạng CNN tối tân.

```
 Ảnh đầu vào (H x W x C)
       │
       ▼
 Chia thành các Patch (P x P)  ──►  Trải phẳng thành Vector
 [Patch 1] [Patch 2] ... [Patch N]      [x1]   [x2]  ...  [xN]
       │                                  │      │          │
       ▼                                  ▼      ▼          ▼
 Chiếu tuyến tính (Linear Projection)   [E1]   [E2]  ...  [EN]
       │
       ▼
 Ghép thêm Token [CLS] và Positional Embeddings
 [CLS] + [E1 + PE1] + [E2 + PE2] + ... + [EN + PEN]
       │
       ▼
 [ Chuỗi Transformer Encoder Blocks ]
       │
       ▼
 Đầu ra của [CLS] ──► MLP Head ──► Phân loại lớp ảnh
```

1. **Biểu diễn Chuỗi mảnh ghép (Patch Embeddings):** Bức ảnh $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ được chia thành $N = \frac{HW}{P^2}$ mảnh nhỏ không chồng lấn kích thước $P \times P$ (thường $P = 16$). Mỗi patch được trải phẳng thành vector chiều dài $P^2 C$ và chiếu tuyến tính qua ma trận $\mathbf{E}$ thành vector nhúng có chiều $d_{\text{model}}$.
2. **Token Phân loại (`[CLS]` Token):** Tương tự BERT, một vector học được `[CLS]` được chèn vào đầu chuỗi. Sau khi đi qua toàn bộ các tầng Transformer, trạng thái đầu ra tại vị trí `[CLS]` được đưa vào bộ phân loại MLP để đưa ra nhãn của bức ảnh.
3. **Đối chiếu Thiên kiến Quy nạp (Inductive Bias): ViT vs. CNN:**
   - **CNN:** Có thiên kiến quy nạp hình học rất mạnh (Tính cục bộ 2D và Tính bất biến dịch chuyển). CNN học rất nhanh từ lượng dữ liệu nhỏ.
   - **ViT:** Gần như **không có thiên kiến quy nạp**; mạng phải tự học cấu trúc không gian 2D từ đầu thông qua dữ liệu. Do đó, khi huấn luyện trên tập dữ liệu nhỏ (ImageNet-1k), ViT dễ bị quá khớp và kém hơn ResNet. Nhưng khi được tiền huấn luyện trên dữ liệu khổng lồ (JFT-300M với 300 triệu ảnh), ViT bứt phá vượt xa CNN nhờ khả năng thiết lập chú ý toàn cục không bị giới hạn.

---

## 12.6 Tài liệu tham khảo (References)

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998-6008. (Bài báo gốc khai sinh Transformer).
2. **Dosovitskiy, A., et al. (2020).** An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*. (Vision Transformer - ViT).
3. **Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024).** RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568, 127063. (RoPE).
4. **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).** Layer normalization. *arXiv preprint arXiv:1607.06450*.
5. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).** BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.

---

## 12.7 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng bài thực nghiệm cài đặt các cơ chế cốt lõi của Transformer từ đầu bằng `NumPy`:

1. **Thành phần 1 - Scaled Dot-Product Attention từ đầu:**
   - Viết hàm `scaled_dot_product_attention(Q, K, V, mask=None)`.
   - Kiểm chứng thực nghiệm vì sao nếu không chia cho $\sqrt{d_k}$, phân phối Softmax sẽ bị bão hòa cực đoan.
2. **Thành phần 2 - Tự cài đặt Multi-Head Attention (MHA from scratch):**
   - Phân tách ma trận $Q, K, V$ thành $h=4$ đầu chú ý, tính Attention song song và ghép nối (Concat) chiếu qua $\mathbf{W}^O$.
3. **Thành phần 3 - Mã hóa Vị trí Hình sin (Sinusoidal Positional Encoding):**
   - Tự viết hàm sinh ma trận $PE$ theo công thức sóng sin/cosin và vẽ bản đồ nhiệt trực quan hóa các dải tần số.
   - Tính tích vô hướng giữa các vector vị trí để chứng minh tính phụ thuộc khoảng cách tương đối.
4. **Thành phần 4 - Causal Masking (Mặt nạ Nhân quả trong GPT Decoder):**
   - Tạo ma trận tam giác dưới `mask` để ngăn chặn token nhìn trước tương lai trong bài toán sinh ngôn ngữ.
5. **Thành phần 5 - Mô phỏng cơ chế Patch Embedding của Vision Transformer (ViT):**
   - Cắt một bức ảnh $32 \times 32$ thành các patch $8 \times 8$, chiếu thành vector và thêm token `[CLS]`.
