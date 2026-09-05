# Chương 10: Thị giác máy tính (Computer Vision)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Nắm vững bản chất dữ liệu hình ảnh:** Hiểu biểu diễn ảnh dưới dạng tensor $H \times W \times C$, hàm cường độ $I(x, y)$ và các bộ lọc truyền thống (Point & Neighborhood operators, Sobel, Gaussian).
2. **Làm chủ nguyên lý thiết kế Mạng nơ-ron tích chập (CNN):** Hiểu sâu sắc 3 nguyên lý trụ cột: Tính kết nối cục bộ (Local Connectivity), Chia sẻ trọng số (Weight Sharing) và Tính tương đương theo dịch chuyển (Translation Equivariance).
3. **Thành thạo toán học lớp tích chập và gộp:** Tính toán kích thước đầu ra qua Stride và Padding; tính toán Miền cảm thụ (Receptive Field) phân cấp; phân biệt Max Pooling và Global Average Pooling (GAP).
4. **Hệ thống hóa các kiến trúc CNN kinh điển phân loại ảnh:** Phân tích bước chuyển dịch từ LeNet-5, AlexNet, VGGNet (triết lý $3 \times 3$), Inception ($1 \times 1$ bottleneck) đến ResNet (Residual / Skip Connections hóa giải suy thoái mạng sâu).
5. **Nắm vững các mô hình Nhận diện đối tượng (Object Detection):** Phân biệt mô hình Two-Stage (R-CNN, Fast R-CNN, Faster R-CNN với RPN) và mô hình One-Stage thời gian thực (YOLO).
6. **Làm chủ các mô hình Phân đoạn ảnh (Image Segmentation):** Phân tích kiến trúc FCN và mạng U-Net với kết nối tắt Encoder-Decoder trong chẩn đoán y tế.

---

## 10.1 Giới thiệu về Thị giác máy tính (Introduction to Computer Vision)

### 10.1.1 Khái niệm
Thị giác máy tính (Computer Vision - CV) là ngành khoa học liên ngành nghiên cứu các thuật toán và hệ thống giúp máy tính có khả năng tự động thu nhận, xử lý, phân tích và "hiểu" thông tin ngữ nghĩa từ dữ liệu hình ảnh và video kỹ thuật số.

### 10.1.2 Tiến trình lịch sử nghiên cứu
- **Thập niên 1950 - 1960:** Thí nghiệm sinh học của David Hubel và Torsten Wiesel (giải Nobel 1981) trên vỏ não thị giác của mèo, phát hiện ra các tế bào thần kinh đơn giản (Simple cells) phản ứng với các cạnh viền theo hướng nhất định, và tế bào phức tạp (Complex cells) phản ứng bất biến với vị trí.
- **Thập niên 1980:** Kunihiko Fukushima phát minh ra **Neocognitron**, kiến trúc nơ-ron nhân tạo đầu tiên mô phỏng cơ chế tế bào đơn giản/phức tạp bằng việc xếp chồng các tầng tích chập và tầng gộp.
- **1998:** Yann LeCun công bố mạng **LeNet-5**, áp dụng học sâu và thuật toán Backpropagation để đọc chữ số viết tay trên séc ngân hàng.
- **2012 - Nay:** Sự kiện **AlexNet** giành chiến thắng cuộc thi ImageNet, mở ra kỷ nguyên bùng nổ của Deep Learning trong thị giác máy tính nhờ GPU và tập dữ liệu lớn.

### 10.1.3 Các tác vụ cốt lõi trong Thị giác máy tính

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Phân loại ảnh │   │   Định vị &     │   │   Nhận diện     │   │    Phân đoạn    │
│ (Classification)│   │ Nhận diện (Det) │   │ đối tượng (YOLO)│   │ (Segmentation)  │
├─────────────────┤   ├─────────────────┤   ├─────────────────┤   ├─────────────────┤
│ "Có 1 con mèo"  │   │ "Con mèo ở đâu" │   │ Hộp Bounding Box│   │ Gán nhãn từng   │
│ (Cat: 0.98)     │   │ [x, y, w, h]    │   │ cho mọi đối     │   │ pixel chính     │
│                 │   │                 │   │ tượng trong ảnh │   │ xác (Mask)      │
└─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘
```

---

## 10.2 Thị giác máy tính truyền thống (Traditional Computer Vision)

### 10.2.1 Cách máy tính "nhìn" một bức ảnh
1. **Dưới dạng Tensor:** Một bức ảnh màu kỹ thuật số được biểu diễn bằng tensor 3 chiều $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$, trong đó:
   - $H$: Chiều cao (Height - số hàng pixel).
   - $W$: Chiều rộng (Width - số cột pixel).
   - $C$: Số kênh màu (Channels: $C=1$ cho ảnh xám, $C=3$ cho ảnh màu RGB, $C=4$ cho ảnh RGBA).
   - Mỗi giá trị pixel thường là một số nguyên $I(i, j, c) \in [0, 255]$ (8-bit integer) hoặc số thực $[0.0, 1.0]$.
2. **Dưới dạng Hàm cường độ liên tục:** Xem ảnh như hàm 2 biến $I(x, y)$ thể hiện độ sáng tại tọa độ $(x, y)$. Các vùng ảnh biến thiên nhanh (đạo hàm bậc nhất $\nabla I = [\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}]^T$ lớn) chính là các **đường biên cạnh (Edges)**.
3. **Dưới dạng Miền tần số (Frequency Domain):** Biến đổi Fourier biến ảnh thành phổ tần số. Tần số thấp biểu diễn các vùng phẳng đồng nhất; tần số cao biểu diễn các chi tiết nhỏ, vân bề mặt (Texture) và cạnh viền.

### 10.2.2 Các Bộ lọc ảnh truyền thống (Image Filters)
Một bộ lọc (Filter / Kernel) là một ma trận nhỏ $\mathbf{K} \in \mathbb{R}^{k \times k}$ trượt trên ảnh để biến đổi giá trị pixel:

#### 1. Bộ lọc điểm (Point Operators):
Chỉ phụ thuộc vào giá trị của chính pixel đó:
- **Biến đổi tương phản:** $I'(x, y) = a \cdot I(x, y) + b$.
- **Gamma Correction:** $I'(x, y) = c \cdot I(x, y)^\gamma$.
- **Cân bằng biểu đồ sắc độ (Histogram Equalization):** Phân bố lại mật độ xác suất mức xám để tối đa hóa độ tương phản.

#### 2. Bộ lọc lân cận tuyến tính (Neighborhood Linear Filters):
- **Bộ lọc làm mờ trung bình (Mean Blur):** $\mathbf{K} = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$ (giảm nhiễu hạt).
- **Bộ lọc Gaussian Blur:** Sử dụng hàm mật độ Gaussian 2D làm trọng số, làm mịn ảnh tự nhiên mà không làm nhòe cạnh quá mức:
  $$K_{ij} = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{(i - c_x)^2 + (j - c_y)^2}{2\sigma^2} \right)$$
- **Bộ lọc phát hiện cạnh viền Sobel (Sobel Edge Detector):** Tính xấp xỉ đạo hàm theo phương ngang và dọc:
  $$\mathbf{K}_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \qquad \mathbf{K}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$
  Độ lớn gradient biên: $G = \sqrt{G_x^2 + G_y^2}$.

---

## 10.3 Mạng Nơ-ron Tích chập (Convolutional Neural Networks - CNN)

### 10.3.1 Giới hạn của MLP và Động lực ra đời của CNN
Nếu dùng mạng MLP thông thường cho một bức ảnh màu độ phân giải $1000 \times 1000 \times 3$, tầng đầu vào có $3 \times 10^6$ nơ-ron. Một tầng ẩn nối đầy đủ (Dense) có 1000 nơ-ron sẽ đòi hỏi **$3 \times 10^9$ tham số**! Điều này gây bùng nổ bộ nhớ, quá khớp trầm trọng và phá hủy cấu trúc không gian 2D cục bộ khi kéo phẳng ảnh (Flatten).

### 10.3.2 Ba nguyên lý trụ cột của kiến trúc CNN
1. **Tính kết nối cục bộ (Local Connectivity):** Mỗi nơ-ron chỉ kết nối với một vùng không gian nhỏ của ảnh đầu vào gọi là **Miền cảm thụ (Receptive Field)**.
2. **Chia sẻ trọng số (Weight Sharing / Parameter Sharing):** Một bộ lọc kernel được dùng chung (trượt qua) toàn bộ các vị trí trên ảnh. Nếu một đặc trưng (ví dụ: góc nhọn) hữu ích ở góc trên bên trái, nó cũng hữu ích ở góc dưới bên phải. Giúp giảm hàng triệu tham số xuống chỉ còn vài trăm tham số!
3. **Tính tương đương theo dịch chuyển (Translation Equivariance):** Nếu dịch chuyển đối tượng trong ảnh một đoạn $\Delta \mathbf{x}$, biểu diễn đặc trưng ở đầu ra của lớp tích chập cũng sẽ bị dịch chuyển tương ứng một đoạn $\Delta \mathbf{x}$:
   $$f(T_{\mathbf{v}}(I)) = T_{\mathbf{v}}(f(I))$$

### 10.3.3 Toán học của Lớp Tích chập (Convolutional Layer)
Cho ảnh đầu vào $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{\text{in}}}$ và một tập gồm $C_{\text{out}}$ bộ lọc, mỗi bộ lọc là một tensor trọng số $\mathbf{W}^{(k)} \in \mathbb{R}^{K_h \times K_w \times C_{\text{in}}}$ cùng độ lệch $b_k$. Phép toán tích chập sinh ra Feature Map đầu ra:
$$S_k(i, j) = \sum_{c=1}^{C_{\text{in}}} \sum_{m=-k_h/2}^{k_h/2} \sum_{n=-k_w/2}^{k_w/2} X(i + m, j + n, c) \cdot W^{(k)}(m, n, c) + b_k$$

#### Công thức tính kích thước Feature Map đầu ra:
$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K_h + 2P}{S} \right\rfloor + 1, \qquad W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K_w + 2P}{S} \right\rfloor + 1$$
Trong đó:
- $K$: Kích thước cạnh của Kernel.
- $P$: Độ dày đệm số 0 (Zero Padding). *Same Padding* chọn $P = \frac{K - 1}{2}$ để giữ nguyên kích thước không gian khi $S=1$.
- $S$: Bước trượt của Kernel (Stride).

### 10.3.4 Lớp Gộp (Pooling Layer) và Miền Cảm thụ (Receptive Field)
- **Max Pooling:** Chọn giá trị lớn nhất trong cửa sổ cục bộ (thường kích thước $2 \times 2$, Stride $2$). Giảm kích thước không gian đi một nửa, giảm số lượng tính toán và tạo ra tính bất biến dịch chuyển nhỏ (small translation invariance).
- **Average Pooling / Global Average Pooling (GAP):** Tính trung bình giá trị. GAP nằm ở cuối mạng, tính trung bình toàn bộ một feature map thành một số vô hướng duy nhất, thay thế cho tầng Fully Connected đồ sộ để tránh quá khớp.
- **Miền cảm thụ (Receptive Field - RF):** Vùng kích thước trên ảnh gốc đầu vào mà một nơ-ron ở tầng thứ $l$ có thể "nhìn thấy":
  $$RF_l = RF_{l-1} + (K_l - 1) \cdot J_{l-1}$$
  trong đó $J_{l-1} = \prod_{i=1}^{l-1} S_i$ là bước nhảy tích lũy. Các tầng càng sâu có Receptive Field càng lớn, giúp mạng nắm bắt được ngữ cảnh toàn cục (người, ô tô) từ các chi tiết cục bộ (cạnh viền, góc).

---

## 10.4 Các Kiến trúc CNN Kinh điển cho Phân loại ảnh (Image Classification)

```
  LeNet-5 (1998) ──► AlexNet (2012) ──► VGGNet (2014) ──► ResNet (2015)
   [5x5 Conv]         [11x11, 5x5]        [Xếp chồng 3x3]   [Skip Connections]
   [Chữ số séc]       [GPU, ReLU]         [Rất sâu: 16-19]  [Cực sâu: 50-152]
```

### 10.4.1 LeNet-5 (Yann LeCun, 1998)
- Thiết kế: `Input(32x32) -> Conv(5x5) -> AvgPool -> Conv(5x5) -> AvgPool -> FC -> FC -> Output(10)`.
- Sử dụng hàm kích hoạt Sigmoid/Tanh; đặt nền móng cho toàn bộ cấu trúc tích chập sau này.

### 10.4.2 AlexNet (Krizhevsky, Sutskever, Hinton, 2012)
- Giành chiến thắng ImageNet 2012 với tỷ lệ lỗi top-5 giảm từ $26\%$ xuống $15.3\%$.
- Các đột phá then chốt: Sử dụng hàm kích hoạt **ReLU** thay cho Sigmoid để tránh triệt tiêu gradient; áp dụng **Dropout ($0.5$)** và **Data Augmentation** chống quá khớp; huấn luyện song song trên 2 GPU NVIDIA GTX 580.

### 10.4.3 VGGNet (Simonyan & Zisserman, 2014 - Oxford)
- **Triết lý thiết kế $3 \times 3$ đồng nhất:** Thay vì dùng kernel lớn ($11 \times 11$ hay $7 \times 7$), VGG chỉ sử dụng liên tiếp các kernel nhỏ $3 \times 3$ với Stride 1.
- **Ý nghĩa toán học:**
  - Hai lớp $3 \times 3$ liên tiếp có Miền cảm thụ tương đương một lớp $5 \times 5$, nhưng số tham số giảm từ $5^2 C^2 = 25C^2$ xuống $2 \times (3^2 C^2) = 18C^2$ (giảm $28\%$) và bổ sung thêm một hàm phi tuyến ở giữa!
  - Ba lớp $3 \times 3$ có Miền cảm thụ tương đương một lớp $7 \times 7$, nhưng giảm đến $45\%$ số tham số.
- VGG-16 và VGG-19 chứng minh rằng mạng càng sâu thì năng lực biểu diễn đặc trưng càng mạnh.

### 10.4.4 ResNet (He et al., 2015 - Microsoft Research)
Khi làm mạng sâu hơn (ví dụ 56 tầng so với 20 tầng), các nhà nghiên cứu phát hiện ra **Vấn đề suy thoái (Degradation Problem)**: Sai số huấn luyện của mạng 56 tầng lại tệ hơn mạng 20 tầng (không phải do overfitting mà do mạng quá sâu khiến gradient không thể tối ưu được).
- **Đột phá Residual Learning (Khối phần dư & Skip Connection):** Thay vì bắt các tầng nơ-ron học trực tiếp hàm ánh xạ mong muốn $\mathcal{H}(\mathbf{x})$, ResNet chia tách:
  $$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x} \implies \mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$$
- **Skip Connection (Đường truyền tắt):** Tín hiệu đầu vào $\mathbf{x}$ được cộng trực tiếp vào đầu ra sau biến đổi:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x}$$
- **Bản chất toán học:** Gradient lan truyền ngược có dạng:
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$$
  Số hạng đồng nhất $+ \mathbf{I}$ đóng vai trò như một **"đường cao tốc gradient" (Gradient Superhighway)**, đảm bảo dòng chảy gradient không bao giờ bị triệt tiêu về 0 ngay cả khi mạng sâu tới 152 tầng hay 1000 tầng!

### 10.4.5 Inception / GoogLeNet (Szegedy et al., 2014)
- **Kiến trúc đa tỷ lệ (Multi-scale processing):** Áp dụng song song các bộ lọc kích thước khác nhau ($1 \times 1, 3 \times 3, 5 \times 5$) cùng Max Pooling trong cùng một khối Inception để chụp các đặc trưng ở nhiều quy mô.
- **Tích chập $1 \times 1$ (Bottleneck Layer):** Giảm số lượng kênh màu (Channel reduction) trước khi tính toán các bộ lọc lớn, giúp giảm hàng triệu phép tính FLOPs.

---

## 10.5 CNN cho Nhận diện Đối tượng (Object Detection)

Nhận diện đối tượng là bài toán kết hợp: Xác định lớp của đối tượng VÀ định vị vị trí bằng hộp chữ nhật giới hạn (Bounding Box: $[x, y, w, h]$).

### 10.5.1 Dòng mô hình Two-Stage (Dựa trên vùng đề xuất - Region Proposals)
1. **R-CNN (2014):** Sử dụng thuật toán Selective Search để cắt ra khoảng 2000 vùng ứng viên (Region Proposals), nạp từng vùng vào mạng CNN riêng biệt để trích xuất đặc trưng $\implies$ Cực kỳ chậm (mất 47 giây/ảnh).
2. **Fast R-CNN (2015):** Nạp toàn bộ ảnh qua CNN một lần duy nhất để tạo Feature Map; sử dụng tầng **RoI Pooling (Region of Interest Pooling)** để trích xuất đặc trưng từng vùng ứng viên từ Feature Map chung $\implies$ Nhanh hơn R-CNN gấp 25 lần.
3. **Faster R-CNN (2015):** Thay thế Selective Search bằng **Mạng đề xuất vùng nơ-ron (Region Proposal Network - RPN)** với các hộp neo (Anchor Boxes) đa tỉ lệ; toàn bộ hệ thống trở thành mô hình nội suy đầu cuối (End-to-End Deep Learning).

### 10.5.2 Dòng mô hình One-Stage: YOLO (You Only Look Once - Redmon et al., 2016)
Thay vì chia thành hai giai đoạn riêng biệt, YOLO biến bài toán nhận diện đối tượng thành một **bài toán hồi quy duy nhất từ ảnh đầu vào ra tọa độ và xác suất lớp**:
- **Cơ chế hoạt động:**
  1. Chia bức ảnh thành lưới $S \times S$ ô (Grid cells).
  2. Nếu tâm của một đối tượng rơi vào ô nào, ô đó chịu trách nhiệm dự đoán đối tượng.
  3. Mỗi ô dự đoán $B$ hộp bao gồm: Tọa độ tâm $(x, y)$, kích thước $(w, h)$, điểm tin cậy (Confidence Score: $P(\text{Object}) \times \text{IoU}$), và phân phối xác suất lớp $C$.
- **Ưu điểm vượt trội:** Tốc độ suy luận siêu tốc (45 – 150 FPS), hoạt động thời gian thực (Real-time) trên camera hành trình và robot tự hành.

---

## 10.6 CNN cho Phân đoạn ảnh (Image Segmentation)

Phân đoạn ảnh là bài toán gán nhãn ngữ nghĩa cho **từng pixel riêng biệt** của bức ảnh.

```
       [ Encoder: Downsampling ]              [ Decoder: Upsampling ]
             (Trích xuất ngữ nghĩa)                  (Phục hồi độ phân giải)
                     ╲                              ╱
                       ╲                          ╱
                         ╲  ── Skip Connection ─►╱
                           ╲                    ╱
                             └── Bottleneck ───┘
```

### 10.6.1 Mạng tích chập hoàn toàn (Fully Convolutional Networks - FCN, 2015)
- Thay thế toàn bộ các tầng Fully Connected ở cuối mạng bằng các tầng Tích chập $1 \times 1$.
- Sử dụng phép tích chập chuyển vị (**Transposed Convolution / Deconvolution**) để phóng to (Upsampling) Feature Map trở lại độ phân giải gốc của ảnh ban đầu.

### 10.6.2 Mạng U-Net (Ronneberger et al., 2015)
- **Cấu trúc hình chữ U đối xứng:**
  - **Nhánh co (Contracting / Encoder path):** Các tầng Conv và Max Pooling liên tiếp để trích xuất ngữ nghĩa ở mức trừu tượng cao (giảm độ phân giải không gian, tăng số kênh đặc trưng).
  - **Nhánh giãn (Expansive / Decoder path):** Các tầng Transposed Conv để khôi phục độ phân giải không gian.
  - **Kết nối tắt tầm xa (Long Skip Connections):** Sao chép trực tiếp các Feature Map độ phân giải cao từ Encoder sang Decoder và ghép nối (Concatenate) với Feature Map được phóng to.
- **Ý nghĩa sống còn:** Giúp Decoder vừa sở hữu thông tin ngữ nghĩa toàn cục mức cao, vừa giữ lại được các chi tiết biên độ nét cao ở mức pixel. U-Net là tiêu chuẩn vàng trong phân đoạn ảnh y tế (khối u não, tế bào máu) và là xương sống kiến trúc của các mô hình khuếch tán tạo sinh (Diffusion Models như Stable Diffusion).

---

## 10.7 Tài liệu tham khảo (References)

1. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo: `resources/Deep Learning.pdf`] (Chương 9: Convolutional Networks).
2. **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** Imagenet classification with deep convolutional neural networks. *NeurIPS*, 25, 1097-1105. (AlexNet).
3. **Simonyan, K., & Zisserman, A. (2014).** Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*. (VGGNet).
4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. *CVPR*, 770-778. (ResNet).
5. **Ren, S., He, K., Girshick, R., & Sun, J. (2015).** Faster R-CNN: Towards real-time object detection with region proposal networks. *NeurIPS*, 28. (Faster R-CNN).
6. **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).** You only look once: Unified, real-time object detection. *CVPR*, 779-788. (YOLO).
7. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 234-241. (U-Net).

---

## 10.8 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng bài thực hành thị giác máy tính toàn diện:

1. **Thành phần 1 - Cài đặt phép toán Tích chập 2D (Conv2D) từ đầu bằng NumPy:**
   - Viết hàm `conv2d(image, kernel, stride, padding)` chuẩn hóa toán học.
   - Thử nghiệm các bộ lọc truyền thống: Làm mịn Gaussian, phát hiện biên cạnh Sobel ngang và dọc.
2. **Thành phần 2 - Khối phần dư ResNet (Residual Block from scratch):**
   - Cài đặt khối phần dư với Skip Connection: $\mathbf{y} = \text{ReLU}(\text{Conv}_2(\text{ReLU}(\text{Conv}_1(\mathbf{x}))) + \mathbf{x})$.
   - Chứng minh gradient truyền ngược nguyên vẹn qua đường truyền tắt.
3. **Thành phần 3 - Xây dựng mô hình CNN hoàn chỉnh phân loại chữ số:**
   - Cấu trúc: `Conv2D(3x3) -> ReLU -> MaxPool2D(2x2) -> Conv2D(3x3) -> ReLU -> GAP -> Dense(10) -> Softmax`.
   - Huấn luyện và đánh giá trên tập dữ liệu ảnh mẫu $8 \times 8$ hoặc $28 \times 28$.
4. **Thành phần 4 - Trực quan hóa Feature Maps (Visualizing CNN Activations):**
   - Trích xuất và trực quan hóa các bản đồ đặc trưng (Feature Maps) qua từng tầng tích chập để quan sát cách mạng "nhìn" từ cạnh viền sơ khai đến các hình thái ngữ nghĩa cao cấp.
5. **Thành phần 5 - Mô phỏng cơ chế IoU (Intersection over Union) và Hộp bao trong Nhận diện đối tượng:**
   - Cài đặt hàm tính chỉ số $\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$ và thuật toán triệt tiêu phi cực đại Non-Maximum Suppression (NMS).
