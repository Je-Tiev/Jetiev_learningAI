# Chương 13: Học biểu diễn quy mô lớn và Mô hình nền tảng (Large-Scale Representation Learning & Foundation Models)

## Mục tiêu chương học
1. **Thấu hiểu động cơ cốt lõi**: Nắm vững bước chuyển dịch từ học có giám sát truyền thống (supervised learning) sang học tự giám sát quy mô lớn (self-supervised pre-training) và mô hình nền tảng (Foundation Models).
2. **Làm chủ các phương pháp tiền huấn luyện**: Hiểu sâu cơ chế toán học của Masked Language Modeling (MLM), Causal Language Modeling (CLM / Autoregressive), và Contrastive Learning (SimCLR, InfoNCE).
3. **Thành thạo các chiến lược tinh chỉnh**: Nắm vững Full Fine-Tuning, Linear Probing, Parameter-Efficient Fine-Tuning (PEFT) tiêu biểu là LoRA (Low-Rank Adaptation), cùng kỹ thuật In-Context Learning (Prompting).
4. **Phân tích các mô hình nền tảng tiêu biểu**: Khảo sát kiến trúc và nguyên lý của BERT, GPT series, wav2vec (âm thanh), và CLIP (đa phương thức hình ảnh - văn bản).

---

## 1. Động cơ ra đời của học biểu diễn quy mô lớn

### 1.1. Nghịch lý dữ liệu gán nhãn và điểm nghẽn của Supervised Learning
Trong kỷ nguyên trước mô hình nền tảng, học sâu phụ thuộc nặng nề vào các tập dữ liệu có giám sát được gán nhãn thủ công (như ImageNet với hơn 1 triệu ảnh, Stanford Sentiment Treebank, SQuAD). Tuy nhiên:
- **Chi phí gán nhãn khổng lồ**: Việc tạo nhãn đòi hỏi chuyên gia trong các lĩnh vực đặc thù (y tế, pháp lý, phân tích mã nguồn).
- **Học cục bộ và nghèo nàn biểu diễn**: Một mô hình chỉ học dự đoán nhãn $y$ từ đầu vào $x$ thường chỉ trích xuất các đặc trưng bề mặt (spurious correlations) tối ưu hóa cho hàm mất mát của tác vụ hẹp đó, thiếu tính tổng quát hóa ra thế giới thực.
- **Biển dữ liệu không gán nhãn**: Nhân loại tạo ra hàng ngàn Petabyte văn bản, hình ảnh, âm thanh tự do trên Internet mỗi ngày. Nhu cầu cấp thiết là khai thác nguồn tài nguyên vô tận này mà không cần gắn bất kỳ nhãn thủ công nào.

### 1.2. Khái niệm Mô hình nền tảng (Foundation Models)
Khái niệm được đề xuất bởi Viện AI Stanford (Bommasani et al., 2021):
$$\text{Dữ liệu khổng lồ không nhãn} \xrightarrow{\text{Self-Supervised Pre-training}} \text{Mô hình nền tảng} \xrightarrow{\text{Adaptation / Fine-Tuning}} \begin{cases} \text{Hỏi đáp (QA)} \\ \text{Dịch thuật} \\ \text{Thị giác máy tính} \\ \text{Lập trình code} \\ \text{Robot điều khiển} \end{cases}$$

Hai đặc tính cốt lõi:
1. **Sự xuất hiện (Emergence)**: Khả năng đột sinh ngầm định của mô hình khi đạt quy mô tham số và dữ liệu đủ lớn mà không hề được lập trình trực tiếp (vd: suy luận logic từng bước - Chain-of-Thought, dịch ngôn ngữ ít tài nguyên).
2. **Sự đồng nhất (Homogenization)**: Một kiến trúc mô hình duy nhất (thường là Transformer) đóng vai trò nền móng cho vô số ứng dụng hạ nguồn.

---

## 2. Các phương pháp tiền huấn luyện (Pre-training Objectives)

### 2.1. Dự đoán che khuất (Masked Prediction)

#### 2.1.1. Masked Language Modeling (MLM - Tiêu biểu: BERT)
Cho chuỗi đầu vào $X = (x_1, x_2, \dots, x_n)$. Ta chọn ngẫu nhiên một tập chỉ số con $M \subset \{1, \dots, n\}$ (thường là $15\%$ số token) để che giấu (thay bằng token đặc biệt `[MASK]`, hoặc giữ nguyên, hoặc thay bằng từ ngẫu nhiên theo tỷ lệ 80-10-10). Ký hiệu chuỗi bị che là $\tilde{X}$.

Hàm mất mát là cực tiểu hóa hàm đối ngẫu Cross-Entropy trên các token bị che:
$$\mathcal{L}_{\text{MLM}}(\theta) = - \sum_{i \in M} \log P_\theta(x_i \mid \tilde{X})$$
Trong đó xác suất được tính qua phân phối Softmax trên toàn bộ từ vựng $|\mathcal{V}|$:
$$P(x_i \mid \tilde{X}) = \frac{\exp(\mathbf{w}_{x_i}^T \mathbf{h}_i)}{\sum_{w' \in \mathcal{V}} \exp(\mathbf{w}_{w'}^T \mathbf{h}_i)}$$
với $\mathbf{h}_i$ là vector ẩn tại vị trí $i$ ở tầng cuối cùng của bộ Encoder Transformer, có khả năng chú ý hai chiều (bidirectional context).

#### 2.1.2. Masked Image Modeling (MIM - Tiêu biểu: MAE - Masked Autoencoders)
Chia ảnh $I$ thành các mảnh (patches). Che giấu tỷ lệ rất cao (ví dụ $75\%$ số patches). Bộ Encoder chỉ xử lý $25\%$ số patch nhìn thấy được, sau đó Decoder nhỏ dự đoán giá trị pixel gốc của các patch bị che:
$$\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \| p_i - \hat{p}_i \|_2^2$$

---

### 2.2. Ngôn ngữ nhân quả (Causal Language Modeling - CLM / Autoregressive)
Mô hình hóa xác suất đồng thời của một chuỗi $X = (x_1, x_2, \dots, x_n)$ thông qua quy tắc chuỗi xác suất có điều kiện:
$$P(X) = \prod_{t=1}^n P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

Kiến trúc sử dụng là Transformer Decoder với cơ chế **Causal Attention Mask** (ma trận tam giác dưới), ngăn chặn thông tin rò rỉ từ tương lai về quá khứ:
$$\mathbf{M}_{ij} = \begin{cases} 0, & \text{nếu } i \ge j \\ -\infty, & \text{nếu } i < j \end{cases}$$

Hàm mất mát âm log-likelihood cực tiểu hóa:
$$\mathcal{L}_{\text{CLM}}(\theta) = - \frac{1}{n} \sum_{t=1}^n \log P_\theta(x_t \mid x_{<t})$$

Ưu điểm vượt trội của CLM là khả năng sinh văn bản tự nhiên theo cơ chế tự hồi quy từng token (autoregressive generation), là nền móng của toàn bộ dòng mô hình GPT.

---

### 2.3. Học biểu diễn tương phản (Contrastive Learning)

#### 2.3.1. Nguyên lý cơ bản (Positive vs Negative pairs)
Cho một mẫu dữ liệu $x$. Ta áp dụng hai phép biến đổi dữ liệu ngẫu nhiên (data augmentations) $t_1, t_2 \sim \mathcal{T}$ để tạo ra cặp dương (positive pair) $(x_i, x_j)$. Các mẫu khác trong batch đóng vai trò là mẫu âm (negative samples).
Mục tiêu: Kéo gần biểu diễn của cặp dương trong không gian vector nhúng, đồng thời đẩy xa biểu diễn của các mẫu âm.

#### 2.3.2. Hàm mất mát InfoNCE (Information Noise-Contrastive Estimation)
Xét vector nhúng chuẩn hóa $\mathbf{z}_i = \frac{f(x_i)}{\|f(x_i)\|_2}$ và $\mathbf{z}_j = \frac{f(x_j)}{\|f(x_j)\|_2}$:
$$\mathcal{L}_{\text{InfoNCE}}^{(i, j)} = - \log \frac{\exp(\mathbf{z}_i^T \mathbf{z}_j / \tau)}{\exp(\mathbf{z}_i^T \mathbf{z}_j / \tau) + \sum_{k=1, k \ne i, j}^{2N} \exp(\mathbf{z}_i^T \mathbf{z}_k / \tau)}$$
Trong đó:
- $\mathbf{z}_i^T \mathbf{z}_j$ là Cosine Similarity giữa hai vector đơn vị.
- $\tau > 0$ là siêu tham số nhiệt độ (temperature scaling), kiểm soát độ sắc nhọn của phân phối xác suất và độ nhạy trước các mẫu âm khó (hard negatives).

Bản chất lý thuyết thông tin: Tối thiểu hóa $\mathcal{L}_{\text{InfoNCE}}$ tương đương với việc tối đa hóa cận dưới của thông tin tương hỗ $I(z_i; z_j)$ giữa hai góc nhìn khác nhau của cùng một đối tượng (Oord et al., 2018).

---

## 3. Các chiến lược tinh chỉnh và chuyển giao (Adaptation & Fine-Tuning)

| Chiến lược | Số tham số cập nhật | Chi phí bộ nhớ GPU | Nguy cơ Quên thảm họa (Catastrophic Forgetting) | Hiệu năng tác vụ chuyên biệt |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | $100\%$ tham số | Rất cao ($16 \times$ bộ nhớ trọng số cho optimizer states) | Cao | Rất cao |
| **Linear Probing** | $< 1\%$ (chỉ tầng phân loại cuối) | Rất thấp | Không | Trung bình - Khá |
| **LoRA (PEFT)** | $0.01\% - 0.5\%$ | Thấp (chỉ lưu gradient của ma trận rank thấp) | Rất thấp | Tương đương hoặc vượt Full Fine-Tuning |
| **Prompting (Zero/Few-Shot)** | $0\%$ (đóng băng hoàn toàn) | Không tốn thêm bộ nhớ lưu trữ trọng số mới | Không | Phụ thuộc vào độ dài ngữ cảnh và kích thước mô hình |

### 3.1. Tinh chỉnh toàn phần (Full Fine-Tuning)
Mô hình khởi tạo từ trọng số pre-trained $\theta_{\text{base}}$. Khi huấn luyện trên tập dữ liệu tác vụ hạ nguồn $\mathcal{D}_{\text{downstream}}$, tất cả ma trận trọng số $\theta$ đều nhận gradient và được cập nhật qua AdamW.
- Nhược điểm: Với mô hình 70 tỷ tham số, việc lưu trữ optimizer states (mô-men bậc 1, bậc 2 dạng FP32) đòi hỏi hàng trăm GB VRAM cho mỗi tác vụ riêng biệt.

### 3.2. Thăm dò tuyến tính (Linear Probing)
Đóng băng (freeze) toàn bộ mạng nơ-ron nền tảng làm bộ trích xuất đặc trưng cố định $\mathbf{h} = f_{\theta_{\text{frozen}}}(x)$. Ta chỉ huấn luyện một bộ phân loại tuyến tính đơn giản:
$$\hat{y} = \text{Softmax}(\mathbf{W}_{\text{head}} \mathbf{h} + \mathbf{b})$$
Thường dùng làm thước đo chuẩn để đánh giá chất lượng của không gian biểu diễn học được trong giai đoạn Pre-training.

### 3.3. LoRA: Low-Rank Adaptation (Hu et al., 2021)
#### 3.3.1. Giả thuyết về Số chiều nội tại thấp (Intrinsic Low Rank Hypothesis)
Các nghiên cứu (Aghajanyan et al., 2020) chỉ ra rằng sự biến thiên trọng số $\Delta \mathbf{W}$ khi thích nghi với tác vụ mới thực chất nằm trên một đa tạp con có số chiều nội tại thấp hơn rất nhiều so với không gian tham số ban đầu $d \times k$.

#### 3.3.2. Cơ chế phân rã ma trận
Thay vì tối ưu trực tiếp ma trận đầy đủ $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$, LoRA đóng băng hoàn toàn $\mathbf{W}_0$ và phân rã phần biến thiên trọng số thành tích của hai ma trận hạng thấp (low-rank):
$$\Delta \mathbf{W} = \mathbf{B} \cdot \mathbf{A}$$
với $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$ và thứ hạng $r \ll \min(d, k)$ (thường chọn $r \in \{4, 8, 16\}$).

Phép biến đổi thuận cho đầu vào $\mathbf{x}$:
$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \Delta \mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} (\mathbf{B} \mathbf{A} \mathbf{x})$$
Trong đó:
- $\mathbf{A}$ được khởi tạo ngẫu nhiên theo phân phối chuẩn Gaussian $\mathcal{N}(0, \sigma^2)$.
- $\mathbf{B}$ được khởi tạo bằng ma trận $0$. Do đó tại thời điểm bắt đầu huấn luyện, $\Delta \mathbf{W} = \mathbf{0}$, hành vi mô hình không hề bị thay đổi đột ngột!
- $\alpha$ là một hằng số tỷ lệ scaling cố định giúp việc tinh chỉnh không bị phụ thuộc quá lớn vào lựa chọn hạng $r$.

**Triển khai suy luận không có độ trễ (Zero Inference Latency)**: Khi đưa vào sản phẩm thực tế, ta có thể cộng trực tiếp trọng số thích nghi vào mô hình gốc: $\mathbf{W}_{\text{deployed}} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}$, khôi phục đúng cấu trúc mạng ban đầu mà không làm tăng bất kỳ phép tính nào trong lúc inference!

### 3.4. Lời nhắc (Prompting) & Học trong ngữ cảnh (In-Context Learning - ICL)
Khám phá đột phá từ GPT-3 (Brown et al., 2020): Mô hình không cần cập nhật bất kỳ trọng số nào ($\nabla_\theta = \mathbf{0}$) mà vẫn giải quyết được bài toán mới thông qua cấu trúc prompt:
- **Zero-Shot Prompting**: Cung cấp mô tả nhiệm vụ trực tiếp: `Dịch câu sau sang tiếng Anh: "Trí tuệ nhân tạo đang thay đổi thế giới" -> `
- **Few-Shot Prompting**: Cung cấp một vài cặp ví dụ (demonstrations) mẫu trước câu hỏi mục tiêu.
- **Chain-of-Thought (CoT)**: Hướng dẫn mô hình tạo chuỗi suy luận từng bước (`Let's think step by step`), giúp tăng vọt độ chính xác trong các bài toán toán học và logic biểu tượng.

---

## 4. Các mô hình nền tảng nổi tiếng

### 4.1. BERT (Devlin et al., 2018 - Bidirectional Encoder Representations from Transformers)
- **Kiến trúc**: Transformer Encoder thuần túy.
- **Mục tiêu tiền huấn luyện**: Masked LM ($15\%$) + Next Sentence Prediction (NSP).
- **Điểm mạnh**: Học biểu diễn ngữ cảnh hai chiều sâu sắc, tạo chuẩn mực cho bài toán hiểu ngôn ngữ tự nhiên (NLU), trích xuất thực thể (NER), và phân loại văn bản.

### 4.2. GPT Series (OpenAI - Generative Pre-trained Transformer: GPT-1, 2, 3, 4)
- **Kiến trúc**: Transformer Decoder thuần túy với Masked Multi-Head Attention.
- **Mục tiêu tiền huấn luyện**: Autoregressive Causal Language Modeling.
- **Quy luật mở rộng (Scaling Laws - Kaplan et al., 2020)**: Hiệu năng mô hình (Loss) giảm theo luật hàm mũ (power-law) khi tăng kích thước mô hình $N$, kích thước dữ liệu $D$, và tổng năng lực tính toán $C$:
  $$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}$$

### 4.3. wav2vec 2.0 (Baevski et al., 2020 - Biểu diễn âm thanh tự giám sát)
- **Kiến trúc**: Mạng tích chập CNN trích xuất đặc trưng thô từ dạng sóng âm thanh (raw audio waveform) $\rightarrow$ Lượng tử hóa vector (Vector Quantization - VQ) gán nhãn biểu diễn rời rạc $\rightarrow$ Transformer Encoder áp dụng kỹ thuật Masking tương tự BERT.
- **Ứng dụng**: Nhận dạng giọng nói tự động (ASR) với lượng dữ liệu gắn nhãn cực nhỏ (chỉ cần 10 phút âm thanh gắn nhãn để đạt độ chính xác cao).

### 4.4. CLIP (Radford et al., 2021 - Contrastive Language-Image Pre-training)
- **Kiến trúc**: Hai tháp (Dual-Tower) gồm Text Encoder (Transformer) và Vision Encoder (ViT / ResNet).
- **Cơ chế**: Nhận một batch gồm $N$ cặp ảnh - văn bản $(I_i, T_i)$. Mô hình chiếu biểu diễn của cả hai về một không gian vector nhúng chung $d$ chiều:
  $$\mathbf{v}_i = \frac{f_{\text{img}}(I_i)}{\|f_{\text{img}}(I_i)\|_2}, \quad \mathbf{u}_i = \frac{f_{\text{txt}}(T_i)}{\|f_{\text{txt}}(T_i)\|_2}$$
- **Hàm mất mát**: Tối đa hóa Cosine Similarity của $N$ cặp đúng $(I_i, T_i)$ và cực tiểu hóa độ tương đồng của $N^2 - N$ cặp sai thông qua mất mát đối xứng Cross-Entropy:
  $$\mathcal{L} = \frac{1}{2} (\mathcal{L}_{\text{image-to-text}} + \mathcal{L}_{\text{text-to-image}})$$
- **Khả năng Zero-Shot Transfer vượt trội**: Phân loại ảnh bất kỳ bằng cách so khớp vector ảnh với vector của các câu mô tả nhãn (`"a photo of a {class}"`).

---

## 5. Tài liệu tham khảo chính thống
1. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. *Deep Learning*, MIT Press. (Chương 15: Representation Learning).
2. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018)**. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
3. **Brown, T. B., et al. (2020)**. *Language Models are Few-Shot Learners* (GPT-3). NeurIPS 2020.
4. **Radford, A., et al. (2021)**. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
5. **Hu, E. J., et al. (2021)**. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
6. **Oord, A. v. d., Li, Y., & Vinyals, O. (2018)**. *Representation Learning with Contrastive Predictive Coding* (InfoNCE). arXiv:1807.03748.

---

## 6. Định hướng thực hành (`demo.ipynb`)
Trong tệp `demo.ipynb`, chúng ta sẽ trực tiếp xây dựng hai trụ cột kỹ thuật của học biểu diễn quy mô lớn từ con số không:
1. **Cài đặt LoRA (Low-Rank Adaptation) từ đầu**: Xây dựng lớp nơ-ron tuyến tính mở rộng $\mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}$, đo lường số lượng tham số có thể huấn luyện (giảm $95\%-99\%$), thực hiện quá trình lan truyền ngược cập nhật riêng hai ma trận $\mathbf{A}, \mathbf{B}$, và kiểm chứng kỹ thuật gộp trọng số (weight merging) cho suy luận tốc độ cao.
2. **Hàm mất mát tương phản InfoNCE & Chiếu không gian nhúng**: Mô phỏng bài toán học biểu diễn tương phản trên các vector dữ liệu nhân tạo, theo dõi ma trận tương đồng (similarity matrix) trước và sau khi tối ưu hóa InfoNCE loss.
