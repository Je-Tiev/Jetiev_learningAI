# Day 10: Adam — Adam: A Method for Stochastic Optimization

> **Bài báo gốc:** *Adam: A Method for Stochastic Optimization*  
> **Tác giả:** Diederik P. Kingma (University of Amsterdam / OpenAI), Jimmy Lei Ba (University of Toronto)  
> **Xuất bản:** International Conference on Learning Representations (ICLR), 2015 (arXiv:1412.6980)  
> **Tài liệu PDF gốc:** [`resources/Day 10 Adam.pdf`](./resources/Day%2010%20Adam.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Nút Thắt Của Các Thuật Toán Tối Ưu Tiền Bối](#2-bối-cảnh-lịch-sử--nút-thắt-của-các-thuật-toán-tối-ưu-tiền-bối)
3. [Kiến Trúc & Bản Chất Toán Học Của Adam](#3-kiến-trúc--bản-chất-toán-học-của-adam)
4. [Đột Phá Kỹ Thuật: Hiệu Chỉnh Độ Chệch (Bias Correction)](#4-đột-phá-kỹ-thuật-hiệu-chỉnh-độ-chệch-bias-correction)
5. [Cơ Chế Hoạt Động & Phân Tích Giải Thuật](#5-cơ-chế-hoạt-động--phân-tích-giải-thuật)
6. [Quy Trình Triển Khai & Khối Đối Chuẩn Học Thuật](#6-quy-trình-triển-khai--khối-đối-chuẩn-học-thuật)
7. [Kết Quả Thực Nghiệm Trong Bài Báo Gốc](#7-kết-quả-thực-nghiệm-trong-bài-báo-gốc)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến AdamW](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-adamw)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Trong lĩnh vực Học sâu (Deep Learning), **bộ tối ưu hóa (Optimizer)** là trái tim điều khiển quá trình cập nhật hàng triệu đến hàng trăm tỷ tham số của mạng nơ-ron nhằm cực tiểu hóa hàm mất mát $\mathcal{L}(\theta)$. 

Trước năm 2014, việc huấn luyện các mạng nơ-ron sâu là một nghệ thuật thử-sai đầy trắc trở:
- Người kỹ sư phải tinh chỉnh thủ công tốc độ học (*learning rate*), thiết lập lịch trình giảm tốc độ học phức tạp (*learning rate schedules*), hoặc chấp nhận thuật toán dao động dữ dội trong các thung lũng dốc hẹp (*ravines*).
- Thuật toán **Adam** (*Adaptive Moment Estimation*) do **Diederik P. Kingma** và **Jimmy Ba** công bố năm 2014 đã tạo nên một cuộc cách mạng: trở thành **bộ tối ưu hóa mặc định số một** của toàn bộ ngành công nghiệp AI trong suốt hơn một thập kỷ.

```
+--------------------------------------------------------------------------------+
|                         CỘT MỐC TIẾN HÓA BỘ TỐI ƯU HÓA                         |
|                                                                                |
|  SGD (Robbins-Monro 1951)                                                      |
|       |                                                                        |
|       +---> Momentum (Polyak 1964) -----> Tăng tốc theo hướng nhất quán        |
|       |                                                                        |
|       +---> AdaGrad (Duchi 2011) --------> Tốc độ học thích ứng theo tọa độ   |
|                 |                                                              |
|                 v                                                              |
|             RMSprop (Hinton 2012) -------> Trung bình động bậc 2 (EMA)         |
|                 |                                                              |
|                 +------------------+                                           |
|                                    |                                           |
|                                    v                                           |
|                        ADAM (Kingma & Ba 2014)                                 |
|            (Mô-men bậc 1 + Mô-men bậc 2 + Bias Correction)                     |
|                                    |                                           |
|                                    v                                           |
|                        ADAMW (Loshchilov & Hutter 2017)                        |
|            (Decoupled Weight Decay - Chuẩn mực cho Transformers & LLMs)        |
+--------------------------------------------------------------------------------+
```

### Ý Nghĩa Cốt Lõi:
1. **Kết hợp ưu điểm vượt trội của hai trường phái:**
   - Trường phái **Momentum**: Giữ đà vận tốc bằng trung bình động hàm mũ của gradient (mô-men bậc 1), dập tắt dao động vuông góc và gia tốc theo hướng đáy thung lũng.
   - Trường phái **RMSprop**: Điều chỉnh bước nhảy theo từng tọa độ dựa vào độ lớn biến thiên bình phương gradient (mô-men bậc 2), giúp các tham số có gradient thưa (*sparse*) hoặc nhỏ vẫn được cập nhật thích đáng.
2. **Hiệu chỉnh độ chệch khởi tạo giải tích (Analytical Bias Correction):** Giải quyết triệt để vấn đề "bước đi chệch hướng về 0" ở những bước lặp đầu tiên do khởi tạo $m_0 = 0, v_0 = 0$.
3. **Bất biến với tỷ lệ gradient (Scale Invariance):** Độ lớn bước nhảy cập nhật tỷ lệ thuận với tỷ số Tín hiệu trên Nhiễu (*Signal-to-Noise Ratio - SNR*) thay vì phụ thuộc hoàn toàn vào độ lớn thuần túy của gradient.

---

## 2. Bối Cảnh Lịch Sử & Nút Thắt Của Các Thuật Toán Tối Ưu Tiền Bối

Để thấu hiểu giá trị học thuật của Adam, ta phải phân tích những hạn chế cố hữu của các phương pháp tiền bối trước năm 2014:

### 2.1. Vanilla Stochastic Gradient Descent (SGD)
$$\theta_t = \theta_{t-1} - \alpha \cdot g_t$$
- **Hạn chế 1 (Dao động trong địa hình cong lệch - Ravines):** Ở các thung lũng mà độ dốc theo chiều này gấp hàng nghìn lần chiều kia (ma trận Hessian có số điều kiện lớn $\kappa \gg 1$), SGD bị dao động zíc-zắc qua lại giữa hai bờ dốc mà hầu như không tiến được về phía trước dọc theo đáy thung lũng.
- **Hạn chế 2 (Tốc độ học đồng nhất):** Áp dụng cùng một hằng số $\alpha$ cho tất cả các chiều tham số, khiến các đặc trưng hiếm gặp (sparse features) gần như không được cập nhật.

### 2.2. SGD có Quán Tính (Momentum)
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad \theta_t = \theta_{t-1} - \alpha \cdot m_t$$
- **Ưu điểm:** Tích lũy vận tốc, triệt tiêu các thành phần dao động ngược chiều và đẩy nhanh quá trình lăn xuống dốc.
- **Hạn chế:** Vẫn dùng chung một tốc độ học toàn cục cho mọi tham số. Không tự động thích ứng khi các tọa độ có phương sai gradient chênh lệch nhau nhiều bậc độ lớn.

### 2.3. AdaGrad (Adaptive Gradient Algorithm - Duchi et al., 2011)
$$v_t = v_{t-1} + g_t^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot g_t$$
- **Ưu điểm:** Mỗi tham số có một tốc độ học riêng. Tham số nào có gradient xuất hiện thường xuyên sẽ bị giảm bước nhảy, tham số nào hiếm gặp sẽ nhận bước nhảy lớn hơn.
- **Nút thắt chí mạng:** Vì $v_t$ là tổng dồn tăng đơn điệu ($v_t \ge v_{t-1}$), mẫu số $\sqrt{v_t}$ tăng liên tục khiến tốc độ học hiệu dụng $\frac{\alpha}{\sqrt{v_t}}$ triệt tiêu dần về 0. Mô hình bị "đóng băng" ngừng học từ rất sớm trước khi chạm tới cực tiểu.

### 2.4. RMSprop (Titchman & Hinton, 2012)
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot g_t$$
- **Ưu điểm:** Thay tổng dồn bằng Trung bình động hàm mũ (*Exponential Moving Average - EMA*), giúp "quên đi" quá khứ xa xôi và duy trì tốc độ học thích ứng liên tục.
- **Hạn chế:** Không có quán tính mô-men bậc 1 (vẫn dùng $g_t$ tức thời gây nhiễu) và **không có cơ chế hiệu chỉnh độ chệch khởi tạo**, dẫn tới bước cập nhật ban đầu rất nhạy cảm và dễ bất ổn định.

---

## 3. Kiến Trúc & Bản Chất Toán Học Của Adam

Adam định nghĩa hai thống kê chuyển động hàm mũ:
1. **Ước lượng Mô-men bậc 1 (First Raw Moment - Kỳ vọng của Gradient):**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. **Ước lượng Mô-men bậc 2 (Second Raw Moment - Kỳ vọng của Bình phương Gradient):**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Trong đó:
- $g_t = \nabla_\theta \mathcal{L}_t(\theta_{t-1})$ là gradient ngẫu nhiên tại bước $t$.
- $\beta_1 \in [0, 1)$ là hệ số phân rã mô-men bậc 1 (mặc định $\beta_1 = 0.9$).
- $\beta_2 \in [0, 1)$ là hệ số phân rã mô-men bậc 2 (mặc định $\beta_2 = 0.999$).
- Phép toán bình phương $g_t^2$ và nhân chia là phép tính theo từng phần tử (element-wise).

---

## 4. Đột Phá Kỹ Thuật: Hiệu Chỉnh Độ Chệch (Bias Correction)

Đây là đóng góp lý thuyết tinh tế và quan trọng nhất của bài báo. 

### 4.1. Vấn Đề Chệch Về Số Không (Zero-bias Problem)
Khi bắt đầu huấn luyện tại bước $t = 0$, ta buộc phải khởi tạo các vector trạng thái bằng 0: $m_0 = \mathbf{0}, v_0 = \mathbf{0}$.
Khai triển chuỗi hồi quy của $m_t$:
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

Lấy kỳ vọng toán học cả hai vế (giả thiết gradient $g_i$ đến từ phân phối dừng xấp xỉ có kỳ vọng $\mathbb{E}[g_i] \approx \mathbb{E}[g_t]$):
$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] \approx \mathbb{E}[g_t] \cdot (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$

Áp dụng công thức tổng cấp số nhân $\sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$:
$$\mathbb{E}[m_t] \approx \mathbb{E}[g_t] \cdot (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g_t] \cdot (1 - \beta_1^t)$$

> ⚠️ **Hệ quả khi chưa hiệu chỉnh:**  
> Ở những bước lặp đầu tiên ($t = 1, 2, ...$), thừa số $(1 - \beta_1^t) < 1$.  
> Đặc biệt với mô-men bậc 2 khi $\beta_2 = 0.999$, tại $t = 1$ ta có:
> $$1 - \beta_2^1 = 1 - 0.999 = 0.001$$
> Giá trị $v_1 = (1 - 0.999) g_1^2 = 0.001 g_1^2$ bị thu nhỏ gấp 1000 lần so với giá trị thực $\mathbb{E}[g_1^2]$!  
> Nếu không có hiệu chỉnh, phép chia cho $\sqrt{v_t}$ sẽ tạo ra bước cập nhật khổng lồ làm nổ mô hình ngay epoch đầu tiên.

### 4.2. Công Thức Hiệu Chỉnh Giải Tích (Unbiased Estimators)
Để ước lượng trở thành không chệch ($\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$ và $\mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]$), ta chia cho hệ số suy giảm:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Khi $t \to \infty$, $1 - \beta_1^t \to 1$ và $1 - \beta_2^t \to 1$, hiệu chỉnh tự động biến mất một cách mượt mà.

---

## 5. Cơ Chế Hoạt Động & Phân Tích Giải Thuật

### 5.1. Thuật Toán Adam Hoàn Chỉnh (Pseudocode)

```python
# Khởi tạo tham số
theta_0: Vector tham số ban đầu
alpha: Tốc độ học cơ sở (mặc định: 0.001)
beta_1: Hệ số suy giảm mô-men 1 (mặc định: 0.9)
beta_2: Hệ số suy giảm mô-men 2 (mặc định: 0.999)
epsilon: Hằng số chống chia cho 0 (mặc định: 1e-8)

m_0 = 0 # Vector mô-men bậc 1
v_0 = 0 # Vector mô-men bậc 2
t = 0   # Bộ đếm bước lặp thời gian

while chưa hội tụ:
    t = t + 1
    g_t = compute_gradient(theta_{t-1}) # Tính gradient mini-batch
    
    # 1. Cập nhật mô-men có trọng số
    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
    v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ** 2)
    
    # 2. Hiệu chỉnh độ chệch khởi tạo
    m_hat_t = m_t / (1 - beta_1 ** t)
    v_hat_t = v_t / (1 - beta_2 ** t)
    
    # 3. Cập nhật tham số
    theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

### 5.2. Góc Nhìn Trực Giác: Tỷ Số Tín Hiệu Trên Nhiễu (SNR)
Bước cập nhật hiệu dụng tại mỗi tọa độ $j$ có dạng:
$$\Delta \theta_t^{(j)} \approx -\alpha \cdot \frac{\hat{m}_t^{(j)}}{\sqrt{\hat{v}_t^{(j)}}}$$
Đại lượng $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \approx \frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}}$ đóng vai trò như **Signal-to-Noise Ratio (SNR)**:
- **Tín hiệu đồng thuận cao (Certainty cao):** Gradient luôn cùng dấu và ổn định qua các bước lặp $\implies \mathbb{E}[g]^2 \approx \mathbb{E}[g^2] \implies \frac{\hat{m}}{\sqrt{\hat{v}}} \approx \pm 1$. Bước nhảy đạt giá trị tối đa $\approx \pm \alpha$.
- **Nhiễu ngẫu nhiên cao (Uncertainty cao):** Gradient liên tục đổi dấu hoặc dao động thất thường $\implies \mathbb{E}[g] \approx 0 \implies \frac{\hat{m}}{\sqrt{\hat{v}}} \to 0$. Bước nhảy tự động co hẹp về 0 để tránh phá vỡ hội tụ!

---

## 6. Quy Trình Triển Khai & Khối Đối Chuẩn Học Thuật

Nhằm làm sáng tỏ giá trị học thuật của bài báo theo đúng quy chuẩn, mã nguồn đi kèm tiến hành đối chứng thực nghiệm 5 phương pháp tối ưu hóa trên **Địa hình hàm Beale / Rosenbrock (Ill-conditioned Valley)** và bài toán huấn luyện Mạng nơ-ron:

1. **Vanilla SGD:** Không có thích ứng, dập dềnh zíc-zắc trong thung lũng.
2. **SGD + Momentum:** Thoát khỏi bẫy cục bộ nhưng văng quán tính rộng.
3. **AdaGrad:** Dừng bước quá sớm do $v_t$ dồn tích vô hạn.
4. **RMSprop:** Thích ứng nhưng dao động bước đi vì thiếu mô-men bậc 1 và bias correction.
5. **Adam (Kingma & Ba 2014):** Hội tụ nhanh nhất, quỹ đạo mượt mà, tự động cân bằng tốc độ học từng chiều.

---

## 7. Kết Quả Thực Nghiệm Trong Bài Báo Gốc

Trong bài báo tại ICLR 2015, Kingma & Ba đã chứng minh tính ưu việt của Adam trên nhiều mô hình và tập dữ liệu kinh điển:

| Bài toán Thực nghiệm | Mô hình Kiến trúc | Đối thủ cạnh tranh | Kết quả Thực nghiệm Adam |
|---|---|---|---|
| **Logistic Regression** | Tuyến tính trên MNIST | SGD, AdaGrad, RMSprop | Adam hội tụ về cực tiểu mất mát nhanh gấp 1.5x - 2x |
| **Multi-layer Perceptron** | MLP 2 tầng ẩn (ReLU, Dropout) | SGD + Nesterov, RMSprop | Tốc độ suy giảm loss vượt trội, không cần schedule thủ công |
| **Convolutional Net (CNN)** | CNN 3 tầng trên CIFAR-10 | RMSprop, AdaDelta | Đạt loss thấp nhất trong 45 epochs đầu tiên |
| **Recurrent Net (RNN)** | LSTM trên Penn Treebank | RMSprop, SGD Momentum | Xử lý hoàn hảo hiện tượng vanishing/exploding gradients |

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến AdamW

Mặc dù cực kỳ thành công, Adam nguyên bản vẫn có những nhược điểm được giới nghiên cứu phân tích sau đó:

### 8.1. Hạn Chế: Khả Năng Khái Quát Hóa Kém Hơn SGD Trong Thị Giác Máy Tính
Khoảng 2016-2017, các nhà nghiên cứu nhận thấy trên các mô hình ResNet huấn luyện cho ImageNet, SGD + Momentum thường đạt độ chính xác kiểm tra (*Test Accuracy*) cao hơn Adam từ 1% đến 2%. Nguyên nhân đến từ việc tốc độ học thích ứng của Adam có xu hướng "nhảy" vào các cực tiểu cục bộ hẹp (*sharp minima*) thay vì cực tiểu phẳng (*flat minima*).

### 8.2. Đột Phá Kế Thừa: AdamW (Loshchilov & Hutter, ICLR 2019)
Nhóm tác giả phát hiện ra một lỗ hổng trong cách Adam kết hợp với kỹ thuật **$L_2$ Regularization** (Weight Decay):
- Trong SGD chuẩn, $L_2$ regularization tương đương với Weight Decay.
- Nhưng trong Adam, số hạng $L_2$ ($\lambda \theta$) được cộng trực tiếp vào gradient: $\tilde{g}_t = g_t + \lambda \theta_{t-1}$. Khi đó, $\lambda \theta$ bị chia cho $\sqrt{v_t} + \epsilon$!
- **Hậu quả:** Những trọng số có gradient lớn sẽ bị phạt $L_2$ rất ít, còn trọng số có gradient nhỏ lại bị phạt quá nặng.
- **Giải pháp AdamW (Decoupled Weight Decay):** Tách biệt hoàn toàn việc phân rã trọng số ra khỏi cập nhật gradient:
  $$\theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
Ngày nay, **AdamW** là bộ tối ưu hóa tiêu chuẩn vàng được sử dụng để huấn luyện hầu hết các mô hình Ngôn ngữ Lớn (LLMs) như LLaMA, GPT-4, Claude và DeepSeek.

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Cấu trúc 4 thành phần quy chuẩn của chuyên đề Day 10:

```text
Day 10 - Adam/
├── README.md              # Báo cáo lý thuyết toàn diện (Tài liệu này)
├── adam_scratch.py        # Tự cài đặt SGD, AdaGrad, RMSprop, Adam, AdamW bằng thuần NumPy
├── adam_scratch.ipynb     # Notebook tương tác từng ô kèm giải thích số học cho người mới
├── adam_pytorch.py        # Cài đặt PyTorch, bộ theo dõi quỹ đạo 2D & xuất outputs/
├── adam_pytorch.ipynb     # Notebook thực hành PyTorch kèm trực quan hóa biểu đồ inline
└── resources/
    └── Day 10 Adam.pdf    # Bài báo nghiên cứu ICLR 2015 gốc
```

### Hướng Dẫn Chạy Mã Nguồn:
```powershell
# 1. Chạy mã nguồn from-scratch thuần NumPy:
python adam_scratch.py

# 2. Chạy mã nguồn PyTorch hiện đại & xuất biểu đồ đối chuẩn:
python adam_pytorch.py
```

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Tại sao Adam cần cả mô-men bậc 1 ($m_t$) lẫn mô-men bậc 2 ($v_t$)? Nếu chỉ dùng $v_t$ mà không có $m_t$ thì đó là thuật toán nào?**
   - *Trả lời:* Mô-men bậc 1 $m_t$ giúp tích lũy quán tính và làm mượt hướng di chuyển (giảm rung lắc). Mô-men bậc 2 $v_t$ giúp đo lường phương sai để thích ứng tốc độ học theo từng chiều. Nếu chỉ dùng $v_t$ mà không dùng $m_t$, thuật toán suy biến thành RMSprop.

2. **Nếu ta bỏ qua bước hiệu chỉnh độ chệch ($m_t / (1 - \beta_1^t)$ và $v_t / (1 - \beta_2^t)$), điều gì sẽ xảy ra ở những bước lặp đầu tiên?**
   - *Trả lời:* Do $m_0 = 0, v_0 = 0$, ở các bước đầu $m_t$ và $v_t$ bị kéo mạnh về 0. Đặc biệt vì $\beta_2 = 0.999$, $v_1 = 0.001 g_1^2$, mẫu số $\sqrt{v_1}$ rất bé làm bước cập nhật $\frac{\alpha}{\sqrt{v_1}} m_1$ bị bùng nổ quá mức (nổ gradient / phân kỳ mô hình).

3. **Ý nghĩa trực giác của đại lượng $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}$ là gì?**
   - *Trả lời:* Đại lượng này đại diện cho tỷ số Tín hiệu trên Nhiễu (SNR). Nếu các gradient cùng hướng, tỷ số này tiến tới $\pm 1$ và bước cập nhật là $\pm \alpha$. Nếu các gradient ngẫu nhiên và trái chiều, tỷ số này tiệm cận 0, tự động giảm tốc độ học ở vùng nhiễu.

4. **Sự khác biệt cốt lõi giữa Adam và AdamW là gì?**
   - *Trả lời:* Adam cộng số hạng phạt $L_2$ vào gradient trước khi đưa vào tính toán mô-men, làm suy giảm tác dụng điều hòa của các trọng số lớn. AdamW tách riêng (decouple) việc phân rã trọng số ra khỏi gradient, trừ trực tiếp $\eta \lambda \theta_{t-1}$ ở bước cập nhật cuối.

5. **Tại sao hệ số $\beta_2$ thường được chọn rất gần 1 ($\beta_2 = 0.999$) trong khi $\beta_1$ chỉ là $0.9$?**
   - *Trả lời:* Mô-men bậc 2 ước lượng kỳ vọng bình phương gradient (phương sai năng lượng). Việc tính phương sai đòi hỏi một cửa sổ quan sát thời gian dài hơn ($\frac{1}{1 - \beta_2} = 1000$ bước lặp) để ổn định thống kê, trong khi hướng gradient (mô-men bậc 1) chỉ cần quan sát khoảng $\frac{1}{1 - \beta_1} = 10$ bước lặp gần nhất để bắt kịp các khúc cua của địa hình hàm mất mát.
