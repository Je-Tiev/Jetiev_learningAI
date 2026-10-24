# Day 05: AlexNet — ImageNet Classification with Deep Convolutional Neural Networks

> **Bài báo gốc:** *ImageNet Classification with Deep Convolutional Neural Networks*  
> **Tác giả:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
> **Xuất bản:** Hội nghị *Advances in Neural Information Processing Systems (NeurIPS / NIPS)*, Lake Tahoe, Nevada, 2012, Trang 1097–1105.  
> **Tài liệu PDF gốc:** [`resources/Day 5 AlexNet.pdf`](./resources/Day%205%20AlexNet.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Sự Hội Tụ Của "Tam Giác Vàng AI"](#2-bối-cảnh-lịch-sử--sự-hội-tụ-của-tam-giác-vàng-ai)
3. [Kiến Trúc Chi Tiết Của Mạng AlexNet](#3-kiến-trúc-chi-tiết-của-mạng-alexnet)
4. [Các Đột Phá Kỹ Thuật Then Chốt](#4-các-đột-phá-kỹ-thuật-then-chốt)
5. [Kỹ Thuật Chống Quá Khớp (Overfitting) Đỉnh Cao](#5-kỹ-thuật-chống-quá-khớp-overfitting-đỉnh-cao)
6. [Quy Trình Huấn Luyện & Tối Ưu Hóa](#6-quy-trình-huấn-luyện--tối-ưu-hóa)
7. [Kết Quả Thực Nghiệm & Cú Sốc ILSVRC 2012](#7-kết-quả-thực-nghiệm--cú-sốc-ilsvrc-2012)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Vào mùa thu năm 2012, tại Hội nghị về Hệ thống Xử lý Thông tin Nơ-ron (NIPS), một bài báo khoa học đã vĩnh viễn thay đổi quỹ đạo phát triển của toàn bộ ngành Trí tuệ Nhân tạo: **ImageNet Classification with Deep Convolutional Neural Networks**.

Tác giả chính là **Alex Krizhevsky** (nghiên cứu sinh tiến sĩ tại Đại học Toronto), cùng với cộng sự thân thiết **Ilya Sutskever** (người sau này đồng sáng lập và là Giám đốc Khoa học của OpenAI) dưới sự hướng dẫn của Giáo sư **Geoffrey Hinton** (Giải Turing 2018, Giải Nobel Vật lý 2024).

Tại cuộc thi thị giác máy tính danh giá nhất hành tinh **ILSVRC 2012**, mô hình của nhóm nghiên cứu mang tên **AlexNet** đã tạo nên một cơn địa chấn chưa từng có:
- Đạt tỷ lệ lỗi **Top-5 là 15.3%**, trong khi giải nhì (sử dụng các phương pháp thị giác truyền thống tinh vi nhất như SIFT + Fisher Vectors + Support Vector Machines) chỉ đạt **26.2%**.
- Cách biệt áp đảo **hơn 10.9%** đã đánh sập hoàn toàn kỷ nguyên của kỹ thuật trích xuất đặc trưng thủ công (Handcrafted Feature Engineering), đưa mạng nơ-ron tích chập sâu (Deep CNN) và phần cứng tính toán GPU lên vũ đài thống trị duy nhất của Trí tuệ Nhân tạo hiện đại.

```text
               CUỘC THI ILSVRC 2012 (1,000 LỚP VẬT THỂ - 1.2 TRIỆU ẢNH)
┌───────────────────────────────────────┬──────────────┬───────────────────────────────┐
│ Phương Pháp Thử Nghiệm                │ Lỗi Top-5    │ Công Nghệ Cốt Lõi             │
├───────────────────────────────────────┼──────────────┼───────────────────────────────┤
│ Giải nhì (ISI Tokyo)                  │ 26.17%       │ Handcrafted SIFT + Fisher SVM │
│ Giải ba (Oxford VGG baseline)         │ 26.97%       │ Sparse Coding + Kernel SVM    │
│ 🏆 ALEXNET (Krizhevsky, Hinton 2012)  │ 15.31%       │ Deep CNN (8 layers, 2x GPUs)  │
└───────────────────────────────────────┴──────────────┴───────────────────────────────┘
```

---

## 2. Bối Cảnh Lịch Sử & Sự Hội Tụ Của "Tam Giác Vàng AI"

Ý tưởng về mạng nơ-ron tích chập không phải mới xuất hiện vào năm 2012. Yann LeCun đã giới thiệu mạng **LeNet-5** từ năm 1998 để nhận diện chữ số séc ngân hàng MNIST. Tuy nhiên, trong suốt 14 năm sau đó, mạng CNN bị lãng quên vì:
- Mạng sâu không thể hội tụ khi dùng hàm kích hoạt Sigmoid hoặc Tanh (Hiện tượng triệt tiêu gradient - Vanishing Gradient).
- Số lượng tham số lên đến hàng chục triệu khiến mô hình bị quá khớp (overfit) ngay lập tức trên các bộ dữ liệu nhỏ vài nghìn ảnh.
- Máy tính cá nhân (CPU) thời đó mất hàng tháng trời mới huấn luyện xong vài nghìn mẫu dữ liệu.

Sự thành công vang dội của AlexNet năm 2012 bắt nguồn từ sự hội tụ kỳ diệu của **"Tam Giác Vàng AI"**:

```text
                           DỮ LIỆU KHỔNG LỒ
                      (ImageNet: 1.2M ảnh, 1000 lớp)
                                    ▲
                                   ╱ ╲
                                  ╱   ╲
                                 ╱     ╲
                                ╱       ╲
                               ╱         ╲
           SỨC MẠNH GPU tính toán ◄─────────► THUẬT TOÁN ĐỘT PHÁ
        (2x NVIDIA GTX 580 - CUDA)             (ReLU, Dropout, LRN)
```

1. **Dữ liệu lớn (Big Data)**: Tập dữ liệu ImageNet của GS. Lý Phi Phi cung cấp 1.2 triệu ảnh gán nhãn, đủ lớn để nuôi dưỡng mô hình 60 triệu tham số mà không bị overfitting chết người.
2. **Sức mạnh tính toán phần cứng (GPU Computing)**: Alex Krizhevsky đã tự tay viết mã lệnh hợp ngữ CUDA thuần để ghép nối song song **2 card đồ họa chơi game phổ thông NVIDIA GeForce GTX 580 (3GB VRAM mỗi card)**, rút ngắn thời gian huấn luyện từ vài tháng xuống chỉ còn **5–6 ngày**.
3. **Đột phá về thuật toán**: Sử dụng hàm kích hoạt **ReLU** chống bão hòa gradient, kỹ thuật **Dropout** chống co cụm trọng số, và **Data Augmentation** tăng cường dữ liệu.

---

## 3. Kiến Trúc Chi Tiết Của Mạng AlexNet

AlexNet gồm **8 tầng có trọng số học được (learnable layers)**: 5 tầng Tích chập (Convolutional Layers - Conv1 đến Conv5) và 3 tầng Kết nối đầy đủ (Fully Connected Layers - FC6 đến FC8).

```text
Đầu vào           Conv 1           Conv 2           Conv 3      Conv 4      Conv 5          FC 6      FC 7      FC 8
[224x224x3] ──> [55x55x96]  ──> [27x27x256] ──> [13x13x384] ──> [13x13] ──> [13x13x256] ──> [4096] ──> [4096] ──> [1000]
                 11x11, s=4        5x5, pad=2        3x3, pad=1   3x3, pad=1  3x3, pad=1
                 ReLU + LRN        ReLU + LRN        ReLU         ReLU        ReLU + Pool
                 + MaxPool 3x3     + MaxPool 3x3                              (6x6x256)
```

### Bảng Thông Số Từng Tầng:

| Tầng | Loại Tầng | Kích Thước Đầu Vào | Kích Thước Bộ Lọc / Kernel | Stride & Pad | Kích Thước Đầu Ra | Số Tham Số Trọng Số |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Input** | Dữ liệu ảnh | $224 \times 224 \times 3$ | - | - | $224 \times 224 \times 3$ | 0 |
| **Conv 1** | Tích chập | $224 \times 224 \times 3$ | $96$ filters, $11 \times 11 \times 3$ | $s=4, p=0$ | $55 \times 55 \times 96$ | $\approx 35\text{K}$ |
| **Pool 1** | MaxPool Overlap | $55 \times 55 \times 96$ | $3 \times 3$ | $s=2, p=0$ | $27 \times 27 \times 96$ | 0 |
| **Conv 2** | Tích chập | $27 \times 27 \times 96$ | $256$ filters, $5 \times 5 \times 48$ | $s=1, p=2$ | $27 \times 27 \times 256$ | $\approx 307\text{K}$ |
| **Pool 2** | MaxPool Overlap | $27 \times 27 \times 256$ | $3 \times 3$ | $s=2, p=0$ | $13 \times 13 \times 256$ | 0 |
| **Conv 3** | Tích chập | $13 \times 13 \times 256$ | $384$ filters, $3 \times 3 \times 256$ | $s=1, p=1$ | $13 \times 13 \times 384$ | $\approx 885\text{K}$ |
| **Conv 4** | Tích chập | $13 \times 13 \times 384$ | $384$ filters, $3 \times 3 \times 192$ | $s=1, p=1$ | $13 \times 13 \times 384$ | $\approx 663\text{K}$ |
| **Conv 5** | Tích chập | $13 \times 13 \times 384$ | $256$ filters, $3 \times 3 \times 192$ | $s=1, p=1$ | $13 \times 13 \times 256$ | $\approx 442\text{K}$ |
| **Pool 5** | MaxPool Overlap | $13 \times 13 \times 256$ | $3 \times 3$ | $s=2, p=0$ | $6 \times 6 \times 256$ | 0 |
| **FC 6** | Dense + Dropout | $6 \times 6 \times 256 = 9216$ | Ma trận trọng số | - | $4,096$ | $\approx 37.7\text{M}$ |
| **FC 7** | Dense + Dropout | $4,096$ | Ma trận trọng số | - | $4,096$ | $\approx 16.8\text{M}$ |
| **FC 8** | Dense + Softmax | $4,096$ | Ma trận trọng số | - | $1,000$ | $\approx 4.1\text{M}$ |

> 💡 **Quan sát then chốt:**  
> - Tổng số tham số của AlexNet là **khoảng 60 triệu tham số** và **650,000 nơ-ron**.
> - Các tầng Convolutional chỉ chiếm **khoảng 5%** số tham số nhưng chiếm **gần 95%** khối lượng tính toán (phép nhân-cộng FLOPs).
> - Tầng Fully Connected (đặc biệt là FC6) chiếm tới **hơn 90%** dung lượng bộ nhớ tham số của mô hình!

---

## 4. Các Đột Phá Kỹ Thuật Then Chốt

### 4.1. Hàm kích hoạt phi tuyến ReLU (Rectified Linear Unit)
Trước AlexNet, các mạng nơ-ron chủ yếu sử dụng hàm Sigmoid $f(x) = \frac{1}{1 + e^{-x}}$ hoặc Tanh $f(x) = \tanh(x)$. Cả hai hàm này đều bị hiện tượng **bão hòa (saturating)**: khi giá trị $|x|$ lớn, đạo hàm suy giảm tiệm cận về 0, làm triệt tiêu hoàn toàn gradient qua các tầng sâu.

AlexNet sử dụng hàm **ReLU không bão hòa**:

$$f(x) = \max(0, x)$$

- **Tốc độ hội tụ vượt trội:** Đạo hàm của ReLU bằng $1$ với mọi $x > 0$. Tín hiệu gradient lan truyền nguyên vẹn qua hàng chục tầng mà không bị suy hao.
- **Thực nghiệm bài báo chứng minh:** Mạng CNN sử dụng ReLU đạt tỷ lệ lỗi huấn luyện 25% **nhanh gấp 6 lần** so với mạng tương đương sử dụng hàm Tanh!

```text
Hàm kích hoạt Tanh (Bão hòa 2 đầu)          Hàm kích hoạt ReLU (Tuyến tính miền dương)
          f(x)                                        f(x)
           ▲                                           ▲
       1 ──┼───╭─────────                          ───┼─────── / (Độ dốc = 1)
           │  ╭╯                                       │      /
           │ ╭╯                                        │     /
      ───┼─┼─┼─────────► x                       ─────┴────/────────► x
         -1│╰╯                                        0
           ▼                                           ▼
```

### 4.2. Huấn luyện phân tán trên 2 GPU (Multi-GPU Parallelism)
Năm 2012, card đồ họa mạnh nhất NVIDIA GTX 580 chỉ có 3GB VRAM, không đủ để chứa toàn bộ mạng 60 triệu tham số cùng các bản đồ đặc trưng (feature maps) trong quá trình lan truyền ngược.

Krizhevsky đã đưa ra kiến trúc **phân đôi mạng**:
- Chia nửa số bộ lọc của mỗi tầng tích chập lên **GPU 1** và **GPU 2** (ví dụ: tầng Conv1 có 96 bộ lọc thì mỗi GPU chạy 48 bộ lọc).
- **Giao tiếp chọn lọc giữa 2 GPU (Cross-GPU Communication):**
  - Các tầng Conv1, Conv2, Conv4 chỉ kết nối nội bộ trong cùng GPU để tiết kiệm băng thông bus PCIe.
  - Tầng Conv3 và các tầng Fully Connected (FC6, FC7, FC8) mới thực hiện giao tiếp chéo lấy dữ liệu từ cả 2 GPU.
- Kỹ thuật này giảm 1.7% lỗi Top-1 và 1.2% lỗi Top-5 so với việc chỉ dùng 1 GPU.

### 4.3. Chuẩn hóa Phản hồi Cục bộ (Local Response Normalization - LRN)
Lấy cảm hứng từ hiện tượng **ức chế bên (Lateral Inhibition)** trong các tế bào nơ-ron thần kinh thị giác sinh học (khi một nơ-ron bị kích thích cực mạnh, nó sẽ phát tín hiệu ức chế các nơ-ron lân cận để làm sắc nét hình ảnh):

$$b_{x, y}^i = \frac{a_{x, y}^i}{\left(k + \frac{\alpha}{n} \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x, y}^j)^2\right)^\beta}$$

Trong đó:
- $a_{x, y}^i$: Giá trị kích hoạt của nơ-ron áp dụng bộ lọc $i$ tại tọa độ $(x, y)$.
- $n$: Số lượng kênh lân cận được chuẩn hóa (AlexNet đặt $n = 5$).
- Các siêu tham số: $k = 2$, $\alpha = 10^{-4}$, $\beta = 0.75$.
- LRN giúp giảm tỷ lệ lỗi Top-1 thêm 1.4% và Top-5 thêm 1.2%. *(Mặc dù ngày nay LRN đã được thay thế hoàn toàn bởi Batch Normalization, nó vẫn là một cột mốc quan trọng).*

### 4.4. Gộp Cực Đại Chồng Lấn (Overlapping Pooling)
Các tầng gộp thông thường có kích cỡ cửa sổ $s \times s$ bằng đúng bước trượt $z$ ($s = z$, không chồng lấn).  
AlexNet thiết lập **bước trượt nhỏ hơn kích cỡ cửa sổ** ($s = 3, z = 2$):
- Các vùng gộp đè lên nhau một phần.
- Giúp giảm 0.4% lỗi Top-1 và 0.3% lỗi Top-5, đồng thời khiến mô hình khó bị quá khớp hơn.

---

## 5. Kỹ Thuật Chống Quá Khớp (Overfitting) Đỉnh Cao

Với 60 triệu tham số, việc AlexNet không bị "học vẹt" 1.2 triệu ảnh là nhờ 2 kỹ thuật điều hòa (regularization) mang tính bước ngoặt:

### 5.1. Kỹ Thuật Tắt Nơ-ron Ngẫu Nhiên (Dropout)
Lần đầu tiên được áp dụng thực chiến ở quy mô lớn (dựa trên nghiên cứu của Hinton et al., 2012):
- Tại mỗi vòng lặp huấn luyện, mỗi nơ-ron trong các tầng FC6 và FC7 có xác suất **$p = 0.5$ bị "tắt" (gán bằng 0)**.
- Các nơ-ron bị tắt không tham gia vào lan truyền xuôi và không được cập nhật trọng số trong lan truyền ngược.
- **Ý nghĩa bản chất:** Ép mạng nơ-ron không được phép phụ thuộc vào sự xuất hiện của bất kỳ nơ-ron cụ thể nào khác. Mỗi nơ-ron buộc phải tự học các đặc trưng độc lập và mạnh mẽ. Dropout tương đương với việc **huấn luyện trung bình trọng số của cấp số mũ $2^N$ mạng nơ-ron khác nhau**!

```text
Mạng Nơ-ron Chuẩn (Dense)                  Mạng Nơ-ron Áp Dụng Dropout (p=0.5)
      ○ ─── ○ ─── ○                              ○ ─── ✕ ─── ○
      │ ╲ ╱ │ ╲ ╱ │                              │     │   ╱ │
      ○ ─── ○ ─── ○                              ✕ ─── ○ ─── ✕
      │ ╱ ╲ │ ╱ ╲ │                                ╲   │ ╱
      ○ ─── ○ ─── ○                              ○ ─── ✕ ─── ○
```

### 5.2. Tăng Cường Dữ Liệu Thực Thể (Data Augmentation)
Tự động nhân bản dữ liệu ngay trên CPU trong lúc GPU đang tính toán:
1. **Cắt và Lật Ngẫu Nhiên (Random Cropping & Horizontal Reflections):**
   - Từ ảnh gốc $256 \times 256$, trích xuất ngẫu nhiên các khung ảnh con kích thước $224 \times 224$ và lật gương nằm ngang.
   - Giúp tăng kích thước tập huấn luyện lên **gấp 2048 lần**, chống biến dạng tỷ lệ và dịch chuyển góc nhìn.
2. **Nhiễu Màu Sắc Bằng Phân Tích Thành Phần Chính (PCA Color Jittering):**
   - Thực hiện phân rã PCA trên toàn bộ giá trị điểm ảnh RGB của tập ImageNet.
   - Với mỗi ảnh, thêm một đại lượng ngẫu nhiên tỷ lệ với các trị riêng (eigenvalues) và vector riêng (eigenvectors):
     $$I_{xy} = I_{xy} + [p_1, p_2, p_3][\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T$$
   - Mô phỏng chính xác sự thay đổi về cường độ và sắc thái ánh sáng tự nhiên ngoài đời thực mà không làm đổi nhãn vật thể.

---

## 6. Quy Trình Huấn Luyện & Tối Ưu Hóa

AlexNet sử dụng thuật toán Giảm Gradient Ngẫu nhiên theo mẻ (Stochastic Gradient Descent - SGD):

- **Kích thước mẻ (Batch size):** 128 mẫu.
- **Động lượng (Momentum):** $\gamma = 0.9$.
- **Suy giảm trọng số (Weight Decay / L2 Regularization):** $\lambda = 0.0005$.
  $$v_{t+1} = 0.9 \cdot v_t - 0.0005 \cdot \epsilon \cdot w_t - \epsilon \cdot \left\langle \left. \frac{\partial L}{\partial w} \right|_{w_t} \right\rangle_{D_i}$$
  $$w_{t+1} = w_t + v_{t+1}$$
- **Tốc độ học (Learning Rate):** Khởi tạo $\epsilon = 0.01$, chia cho 10 khi tỷ lệ lỗi kiểm định ngừng cải thiện (giảm 3 lần trong suốt 90 epoch).
- **Khởi tạo trọng số:** Phân phối chuẩn $w \sim \mathcal{N}(0, 0.01)$, bias các tầng Conv2, Conv4, Conv5 và các tầng FC được khởi tạo bằng hằng số $1$ để tăng tốc độ kích hoạt ReLU trong những vòng đầu.

---

## 7. Kết Quả Thực Nghiệm & Cú Sốc ILSVRC 2012

### 7.1. Bảng Xếp Hạng ILSVRC 2012

| Mô Hình / Phương Pháp | Tỷ Lệ Lỗi Top-1 | Tỷ Lệ Lỗi Top-5 | Ghi Chú |
|---|:---:|:---:|---|
| Phương pháp truyền thống tốt nhất (2010) | 47.1% | 28.2% | SIFT + LBP + SVM |
| Giải nhì ILSVRC 2012 (ISI) | 45.7% | 26.2% | Hợp thể 7 mô hình Fisher Vector |
| **AlexNet (1 CNN duy nhất)** | **40.7%** | **18.2%** | Đánh bại giải nhì 8% |
| **AlexNet (Hợp thể 5 CNNs)** | **37.5%** | **15.3%** | **Vô địch tuyệt đối, cách biệt 10.9%** |

### 7.2. Đặc trưng thị giác tầng đầu tiên (Feature Visualization)
Khi trực quan hóa 96 bộ lọc của tầng Conv1 kích thước $11 \times 11 \times 3$ (Figure 3 trong bài báo):
- **GPU 1:** Chuyên học các bộ lọc tần số cao định hướng không màu (bộ lọc phát hiện cạnh Gabor ở nhiều góc độ: dọc, ngang, nghiêng $45^\circ$).
- **GPU 2:** Chuyên học các bộ lọc đốm màu (color blobs) và gradient màu sắc diện rộng.
- **Hiện tượng đáng kinh ngạc:** Mạng nơ-ron sâu tự động tái hiện lại chính xác cơ chế xử lý hình ảnh của vỏ não thị giác sơ cấp (V1 Cortex) của động vật có vú mà hoàn toàn không cần con người lập trình trước!

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Hạn chế của AlexNet nhìn từ hiện đại
1. **Bộ lọc Conv1 quá lớn ($11 \times 11$, stride 4)**: Làm mất quá nhiều chi tiết không gian ngay từ bước đầu tiên.
2. **LRN cồng kềnh**: Kém hiệu quả hơn nhiều so với Batch Normalization (Ioffe & Szegedy 2015).
3. **Tầng Fully Connected chiếm quá nhiều bộ nhớ**: FC6 và FC7 ngốn gần 55 triệu tham số, gây lãng phí tính toán và dễ gây overfitting.

### 8.2. Cây phả hệ tiến hóa từ AlexNet
```text
ALEXNET (2012) - 8 layers, 11x11 convs, Top-5: 16.4%
     │
     ├──► ZFNet (2013) - Tinh chỉnh Conv1 thành 7x7, stride 2, Top-5: 11.7%
     │
     ├──► VGGNet (2014) - Thay thế 11x11/5x5 bằng chuỗi các bộ lọc nhỏ 3x3, Top-5: 7.3%
     │
     ├──► GoogLeNet / Inception (2014) - Khối đa nhánh, Global Average Pooling loại bỏ FC, Top-5: 6.7%
     │
     ├──► ResNet (2015) - Kết nối tắt Residual Connections, sâu 152 tầng, Top-5: 3.57% (Vượt mắt người!)
     │
     └──► Vision Transformer / ViT (2020) - Đưa Attention thay thế hoàn toàn Convolution.
```

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục cung cấp 2 tệp cài đặt mã nguồn hoàn chỉnh:

```text
Day 05 - AlexNet/
├── README.md               # Báo cáo lý thuyết chuyên sâu 10 mục chuẩn này
├── alexnet_scratch.py      # Cài đặt thuần Python & NumPy (Conv2D im2col, Overlapping MaxPool, ReLU, LRN, Dropout)
├── alexnet_pytorch.py      # Triển khai AlexNet chuẩn PyTorch, trực quan hóa 96 bộ lọc Conv1 & xuất biểu đồ
└── resources/
    └── Day 5 AlexNet.pdf
```

### 9.1. Tệp `alexnet_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python chuẩn + NumPy (không dùng PyTorch hay thư viện ngoài).
- **Tính năng nổi bật**:
  - Tự cài đặt phép **Tích chập 2D (Convolution Forward & Backward)**.
  - Tự cài đặt **Overlapping Max Pooling ($3 \times 3$, stride 2)** lưu vết chỉ số cực đại để truyền gradient.
  - Cài đặt chuẩn hóa **Local Response Normalization (LRN)** theo đúng phương trình (1) trong bài báo.
  - Tự cài đặt tầng **Inverted Dropout ($p = 0.5$)** và hàm mất mát Softmax Cross-Entropy.
  - Chạy mô phỏng mini-AlexNet end-to-end, kiểm chứng lan truyền ngược và giảm thiểu hàm mất mát.

### 9.2. Tệp `alexnet_pytorch.py` (Triển khai bằng PyTorch)
- **Công nghệ**: PyTorch, torchvision, matplotlib.
- **Tính năng nổi bật**:
  - Tái tạo kiến trúc nguyên bản AlexNet cổ điển kèm lớp `LocalResponseNorm`.
  - Hỗ trợ nạp dữ liệu CIFAR-10 hoặc Fallback an toàn tự sinh ảnh nhiều lớp khi chạy offline.
  - **Tái hiện Figure 3 của bài báo:** Trích xuất và vẽ lưới $96$ bộ lọc tích chập tầng Conv1 ($11 \times 11 \times 3$) hiển thị các bộ dò cạnh và màu sắc.
  - Tự động xuất ảnh phân tích bộ lọc vào `outputs/alexnet_filters.png` và đồ thị huấn luyện vào `outputs/alexnet_training.png`.

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Vì sao việc chuyển đổi từ hàm kích hoạt Sigmoid/Tanh sang ReLU lại giúp AlexNet giải quyết được vấn đề triệt tiêu gradient và tăng tốc huấn luyện gấp 6 lần?**
2. **Kỹ thuật Dropout ($p=0.5$) hoạt động như thế nào trong quá trình huấn luyện và suy luận (inference)? Tại sao Dropout lại được ví như việc huấn luyện trung bình một tập hợp khổng lồ các mô hình con?**
3. **Phân tích sự khác biệt về mặt phân bổ tham số và khối lượng tính toán giữa các tầng tích chập (Convolutional) và các tầng kết nối đầy đủ (Fully Connected) trong AlexNet?**
4. **Cơ chế hoạt động và vai trò sinh học của chuẩn hóa phản hồi cục bộ (Local Response Normalization - LRN) là gì? Vì sao trong các kiến trúc hiện đại ngày nay, LRN lại bị thay thế bởi Batch Normalization?**
5. **Kỹ thuật Overlapping Pooling ($s=3, z=2$) khác với Non-overlapping Pooling ($s=2, z=2$) như thế nào và nó đem lại lợi ích gì cho việc chống quá khớp (overfitting)?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
