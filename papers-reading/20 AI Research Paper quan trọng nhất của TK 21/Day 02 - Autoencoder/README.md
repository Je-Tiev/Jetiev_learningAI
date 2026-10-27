# Day 02: Autoencoder — Reducing the Dimensionality of Data with Neural Networks

> **Bài báo gốc:** *Reducing the Dimensionality of Data with Neural Networks*  
> **Tác giả:** G. E. Hinton & R. R. Salakhutdinov  
> **Xuất bản:** Tạp chí *Science*, Tập 313, Số 5786, Ngày 28 tháng 7 năm 2006, Trang 504–507.  
> **Tài liệu PDF gốc:** [`resources/Day 2 Autoencoder.pdf`](./resources/Day%202%20Autoencoder.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Bài Toán Giảm Chiều Dữ Liệu](#2-bối-cảnh-lịch-sử--bài-toán-giảm-chiều-dữ-liệu)
3. [Kiến Trúc Autoencoder Truyền Thống](#3-kiến-trúc-autoencoder-truyền-thống)
4. [Đột Phá Của Hinton (2006): Pre-training Từng Tầng Bằng RBM](#4-đột-phá-của-hinton-2006-pre-training-từng-tầng-bằng-rbm)
5. [Thuật Toán Contrastive Divergence (CD-1)](#5-thuật-toán-contrastive-divergence-cd-1)
6. [Quy Trình 3 Bước: Pre-train, Unroll & Fine-tune](#6-quy-trình-3-bước-pre-train-unroll--fine-tune)
7. [Kết Quả Thực Nghiệm Trong Bài Báo](#7-kết-quả-thực-nghiệm-trong-bài-báo)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Vào năm 2006, mạng nơ-ron nhân tạo gần như bị cộng đồng nghiên cứu AI lãng quên và ruồng bỏ sau nhiều năm "Mùa đông AI" (AI Winter). Phần lớn các nhà khoa học máy tính lúc bấy giờ ưa chuộng các thuật toán có cơ sở toán học lồi rõ ràng như **Support Vector Machines (SVM)** hay **Random Forests**. Việc huấn luyện các mạng nơ-ron sâu (nhiều hơn 2-3 tầng ẩn) bằng giải thuật lan truyền ngược thông thường (Backpropagation) gần như **bất khả thi**: mạng không hội tụ, bị kẹt trong các cực tiểu địa phương nghèo nàn, hoặc chỉ tái tạo ra trung bình mờ nhạt của tập dữ liệu.

Bài báo năm 2006 của **Geoffrey Hinton** (cha đẻ Deep Learning, giải Turing 2018, giải Nobel Vật lý 2024) và nghiên cứu sinh **Ruslan Salakhutdinov** trên tạp chí danh giá *Science* đã đánh dấu **phát súng mở màn cho sự phục hưng của Deep Learning**.

### Đóng góp cốt lõi của nghiên cứu:
1. **Chứng minh mạng nơ-ron sâu có thể được huấn luyện thành công**: Thông qua kỹ thuật **học không giám sát từng tầng (Greedy Layer-wise Unsupervised Pre-training)** bằng Cỗ máy Boltzmann Giới hạn (Restricted Boltzmann Machine - RBM).
2. **Khởi tạo trọng số thông minh (Smart Weight Initialization)**: Giải quyết triệt để vấn đề sa lầy vào cực tiểu địa phương và triệt tiêu gradient khi khởi tạo ngẫu nhiên.
3. **Vượt trội hoàn toàn so với PCA**: Chứng minh Autoencoder phi tuyến tính sâu nén và bảo toàn cấu trúc dữ liệu tốt hơn rất nhiều so với phương pháp phân tích thành phần chính (PCA) cổ điển.

---

## 2. Bối Cảnh Lịch Sử & Bài Toán Giảm Chiều Dữ Liệu

### 2.1. Lời nguyền số chiều (Curse of Dimensionality)
Dữ liệu trong thế giới thực thường có số chiều cực lớn:
- Một bức ảnh $28 \times 28$ điểm ảnh tương ứng với một vector trong không gian $\mathbb{R}^{784}$.
- Một văn bản biểu diễn qua túi từ (Bag-of-Words) có thể lên đến hàng chục nghìn chiều.

Khi số chiều tăng cao, không gian trở nên cực kỳ thưa thớt, khoảng cách Euclidean giữa các điểm dữ liệu mất dần ý nghĩa phân biệt, đòi hỏi lượng dữ liệu cấp số mũ để mô hình hóa và chi phí tính toán tăng vọt.

### 2.2. Giả thuyết Đa tạp (Manifold Hypothesis)
Mặc dù dữ liệu nằm trong không gian nhiều chiều, nhưng chúng **không phân bố ngẫu nhiên trên toàn bộ không gian**, mà nằm trên hoặc rất gần một **đa tạp phi tuyến tính có số chiều nội tại (intrinsic dimension) nhỏ hơn rất nhiều**. 
> *Ví dụ:* Một bức ảnh chữ số viết tay kích thước $28 \times 28$ (784 chiều) thực chất chỉ được điều khiển bởi một vài bậc tự do: độ nghiêng của nét bút, độ dày nét, hình dáng vòng cong cong,... (chỉ khoảng 6–30 chiều thực).

### 2.3. Hạn chế của PCA (Principal Component Analysis)
Phương pháp giảm chiều phổ biến nhất trước đó là PCA. PCA tìm kiếm các trục trực giao có phương sai lớn nhất. Tuy nhiên:
- **PCA hoàn toàn là tuyến tính (Linear)**: PCA chỉ có thể chiếu dữ liệu lên một siêu phẳng phẳng.
- **Thất bại với các cấu trúc cong**: Nếu dữ liệu nằm trên một đa tạp xoắn ốc (Swiss roll) hoặc đường cong phức tạp hình chữ S, PCA sẽ "đè bẹp" các vòng xoắn lên nhau, làm mất hoàn toàn cấu trúc hình học nội tại.

```text
Dữ liệu phi tuyến (Đa tạp cong hình chữ S):
   ╭──────╮
   │  ●   ╰──────╮
   │   ●      ●  │   <--- PCA (chiếu thẳng đứng) sẽ đè bẹp các điểm thuộc
   ╰──────╮      │        2 nhánh khác nhau vào cùng một vị trí!
          ╰──────╯
```

### 2.4. Khó khăn khi huấn luyện Deep Autoencoder trước 2006
Ý tưởng về Autoencoder đã tồn tại từ những năm 1980 (Rumelhart, Hinton, Williams, 1986). Tuy nhiên:
- **Nếu khởi tạo trọng số lớn**: Mạng bị kẹt ở các cực tiểu địa phương tồi (poor local minima).
- **Nếu khởi tạo trọng số nhỏ**: Đạo hàm lan truyền ngược qua nhiều tầng kích hoạt phi tuyến (như Sigmoid) suy giảm cấp số nhân về 0 (**Vanishing Gradient**). Các tầng đầu tiên gần như không được cập nhật trọng số.
- Kết quả: Mạng Autoencoder sâu chỉ có thể học cách xuất ra **trung bình cộng của tất cả ảnh huấn luyện**!

---

## 3. Kiến Trúc Autoencoder Truyền Thống

Autoencoder là một mạng nơ-ron học không giám sát (unsupervised), được thiết kế để **tự tái tạo lại đầu vào của chính nó** thông qua một nút cổ chai (bottleneck).

```text
Đầu vào x           Tầng ẩn 1       Bottleneck (z)      Tầng ẩn 2          Tái tạo x̂
[784 chiều]  ───>  [1000 chiều] ───> [30 chiều]  ───> [1000 chiều] ───> [784 chiều]
  ─────── ENCODER (Mã hóa) ───────┘              └─────── DECODER (Giải mã) ───────
```

### 3.1. Thành phần mạng
1. **Bộ mã hóa (Encoder) $f_\theta$:**
   Nén đầu vào $x \in \mathbb{R}^D$ thành vector tiềm ẩn (latent code) $z \in \mathbb{R}^d$ với $d \ll D$:
   $$z = f_\theta(x) = \sigma(W_e x + b_e)$$
2. **Nút cổ chai (Bottleneck / Code Layer):**
   Vùng thắt lại của mạng, giới hạn dung lượng thông tin ép mô hình không thể "học vẹt" (identity copy) mà buộc phải chắt lọc các đặc trưng cốt lõi nhất.
3. **Bộ giải mã (Decoder) $g_\phi$:**
   Giải nén vector tiềm ẩn $z$ để tái tạo lại dữ liệu gốc $\hat{x} \in \mathbb{R}^D$:
   $$\hat{x} = g_\phi(z) = \sigma(W_d z + b_d)$$

### 3.2. Trọng số đối xứng (Tied Weights) vs Tự do (Untied Weights)
- **Tied weights**: Đặt $W_d = W_e^T$. Giúp giảm một nửa số tham số, chống overfitting và hoạt động như một cơ chế điều hòa (regularization).
- **Untied weights**: $W_d$ và $W_e$ là các ma trận độc lập được cập nhật riêng biệt.

### 3.3. Hàm mất mát tái tạo (Reconstruction Loss)
- **Mean Squared Error (MSE)** (dùng cho dữ liệu liên tục hoặc xấp xỉ Gaussian):
  $$\mathcal{L}_{\text{MSE}}(x, \hat{x}) = \frac{1}{2} \sum_{i=1}^D (x_i - \hat{x}_i)^2$$
- **Binary Cross-Entropy (BCE)** (khi dữ liệu được chuẩn hóa về $[0, 1]$ như cường độ điểm ảnh pixel):
  $$\mathcal{L}_{\text{BCE}}(x, \hat{x}) = - \sum_{i=1}^D \left[ x_i \log \hat{x}_i + (1 - x_i) \log (1 - \hat{x}_i) \right]$$

> **Mối liên hệ kinh điển giữa Autoencoder và PCA (Bourlard & Kamp, 1988):**  
> Nếu các hàm kích hoạt ở tất cả các tầng là **tuyến tính** và hàm mất mát là MSE, thì không gian con $d$ chiều được học bởi Autoencoder chính là **không gian con sinh bởi $d$ thành phần chính của PCA**! Thêm các hàm kích hoạt phi tuyến (như Sigmoid, ReLU) cho phép Autoencoder vượt ra khỏi giới hạn siêu phẳng của PCA để học các đa tạp uốn lượn.

---

## 4. Đột Phá Của Hinton (2006): Pre-training Từng Tầng Bằng RBM

Để giải quyết bài toán khởi tạo trọng số cho mạng sâu, Hinton & Salakhutdinov đã đề xuất sử dụng **Restricted Boltzmann Machine (RBM)** để huấn luyện sơ bộ (pre-train) từng tầng một theo phương pháp tham lam (Greedy Layer-wise).

```text
                h_1        h_2        h_3       (Code)
                 ○          ○          ○          ○
               ↗   ↖      ↗   ↖      ↗   ↖      ↗   ↖
              ● ─── ●    ● ─── ●    ● ─── ●    ● ─── ●
               Visible    Visible    Visible    Visible
               (Data)     (h_1)      (h_2)      (h_3)
              [RBM 1]    [RBM 2]    [RBM 3]    [RBM 4]
```

### 4.1. Cấu trúc của RBM
RBM là một mạng nơ-ron hai tầng gồm:
- **Tầng nhìn thấy (Visible layer - $v$)**: Đại diện cho dữ liệu quan sát được ($v_i \in \{0, 1\}$).
- **Tầng ẩn (Hidden layer - $h$)**: Đại diện cho các bộ dò đặc trưng ($h_j \in \{0, 1\}$).
- **Đặc điểm then chốt ("Restricted")**: Không có bất kỳ kết nối nào giữa các nơ-ron trong cùng một tầng. Đây là một **đồ thị hai phía (Bipartite Graph)**.

### 4.2. Hàm năng lượng (Energy Function)
Cấu hình đồng thời $(v, h)$ được gán một mức năng lượng $E(v, h)$:
$$E(v, h) = - \sum_{i \in \text{pixels}} b_i v_i - \sum_{j \in \text{features}} c_j h_j - \sum_{i, j} v_i h_j w_{ij}$$
Trong đó:
- $w_{ij}$ là trọng số liên kết giữa nút nhìn thấy $i$ và nút ẩn $j$.
- $b_i$ là độ lệch (bias) của nút nhìn thấy $i$.
- $c_j$ là độ lệch (bias) của nút ẩn $j$.

Xác suất đồng thời của một cấu hình được xác định theo phân phối Boltzmann:
$$P(v, h) = \frac{1}{Z} e^{-E(v, h)}, \quad Z = \sum_{v, h} e^{-E(v, h)} \quad (\text{Hằng số chuẩn hóa Partition Function})$$

### 4.3. Tính độc lập có điều kiện (Conditional Independence)
Nhờ việc không có liên kết nội tầng, khi cố định tầng này, các nơ-ron của tầng kia hoàn toàn **độc lập có điều kiện** với nhau:
$$P(h_j = 1 | v) = \sigma\left(c_j + \sum_i v_i w_{ij}\right)$$
$$P(v_i = 1 | h) = \sigma\left(b_i + \sum_j h_j w_{ij}\right)$$
Trong đó $\sigma(x) = \frac{1}{1 + e^{-x}}$ là hàm sigmoid. Điều này cho phép **lấy mẫu song song (parallel Gibbs sampling)** cực kỳ nhanh chóng trên toàn bộ tầng thay vì cập nhật tuần tự từng nơ-ron.

---

## 5. Thuật Toán Contrastive Divergence (CD-1)

Mục tiêu huấn luyện RBM là cực đại hóa log-likelihood của dữ liệu huấn luyện:
$$\frac{\partial \log P(v)}{\partial w_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$
- $\langle v_i h_j \rangle_{\text{data}}$: Kỳ vọng dưới phân phối dữ liệu thật (dễ tính).
- $\langle v_i h_j \rangle_{\text{model}}$: Kỳ vọng dưới phân phối của mô hình. Để tính chính xác đòi hỏi chuỗi Markov Chain Monte Carlo (MCMC) chạy vô hạn bước đến khi hội tụ (bất khả thi về mặt tính toán).

### Thuật toán CD-1 (1 bước Contrastive Divergence) của Hinton:
Hinton đề xuất một xấp xỉ thiên tài: chỉ cần chạy **1 chuỗi lấy mẫu Gibbs duy nhất** bắt đầu từ dữ liệu thực:

1. **Pha Dương (Positive Phase - Dữ liệu thật):**
   - Đặt $v^{(0)} = x_{\text{data}}$.
   - Tính xác suất kích hoạt và lấy mẫu nhị phân cho tầng ẩn:
     $$h_j^{(0)} \sim P(h_j = 1 | v^{(0)}) = \sigma(c_j + \sum_i v_i^{(0)} w_{ij})$$
2. **Pha Âm (Negative Phase / Confabulation - Tái tạo/Ảo giác):**
   - Từ $h^{(0)}$, tái tạo lại dữ liệu nhìn thấy:
     $$v_i^{(1)} \sim P(v_i = 1 | h^{(0)}) = \sigma(b_i + \sum_j h_j^{(0)} w_{ij})$$
   - Từ $v^{(1)}$, lấy mẫu lại tầng ẩn một lần nữa:
     $$h_j^{(1)} \sim P(h_j = 1 | v^{(1)}) = \sigma(c_j + \sum_i v_i^{(1)} w_{ij})$$
3. **Cập nhật tham số:**
   $$\Delta w_{ij} = \epsilon \left( v_i^{(0)} h_j^{(0)} - v_i^{(1)} h_j^{(1)} \right)$$
   $$\Delta b_i = \epsilon \left( v_i^{(0)} - v_i^{(1)} \right)$$
   $$\Delta c_j = \epsilon \left( h_j^{(0)} - h_j^{(1)} \right)$$
   *(trong đó $\epsilon$ là tốc độ học learning rate).*

> **Ý nghĩa trực quan:** Thuật toán kéo năng lượng $E(v, h)$ của dữ liệu thật xuống thấp (làm dữ liệu thật có xác suất cao hơn) và đẩy năng lượng của dữ liệu do mạng tự tưởng tượng ($v^{(1)}$) lên cao.

---

## 6. Quy Trình 3 Bước: Pre-train, Unroll & Fine-tune

Toàn bộ quy trình huấn luyện một Deep Autoencoder trong bài báo diễn ra qua 3 bước chặt chẽ:

```text
    BƯỚC 1: PRE-TRAIN (RBM)          BƯỚC 2: UNROLL             BƯỚC 3: FINE-TUNE
                                                                (Backpropagation)
         [RBM 3: Top]                ┌─ W_4 ─────────┐
         30 units (Linear)           │   [30 units]  │          Toàn bộ mạng được
               ▲                     │   Code Layer  │          tinh chỉnh đồng thời
               │ W_3                 │       ▲       │          bằng Gradient Descent:
         [RBM 2]                     │   W_3 │   W_3ᵀ│          L(x, x̂) -> min
         500 units                   │       ▼       │
               ▲               ==>   │  [500 units]  │   ==>    Decoder ban đầu dùng
               │ W_2                 │       ▲       │          trọng số chuyển vị
         [RBM 1]                     │   W_2 │   W_2ᵀ│          của Encoder:
         1000 units                  │       ▼       │          W_decoder = W_encoderᵀ
               ▲                     │ [1000 units]  │
               │ W_1                 │       ▲       │
          Input Image                │   W_1 │   W_1ᵀ│
          (784 pixels)               └───────┴───────┘
```

1. **Bước 1: Greedy Layer-Wise Pre-training:**
   - Huấn luyện RBM 1 trên ảnh gốc $784 \to 1000$.
   - Cố định trọng số $W_1$, dùng vector kích hoạt $h_1$ làm đầu vào huấn luyện RBM 2 ($1000 \to 500$).
   - Tiếp tục huấn luyện RBM 3 ($500 \to 250$) và RBM 4 ($250 \to 30$).
   - *Lưu ý*: Tầng 30 chiều ở đỉnh được thiết kế dùng các đơn vị tuyến tính Gaussian (Linear units with Gaussian noise) để tạo ra các giá trị liên tục thích hợp cho việc giảm chiều và trực quan hóa.
2. **Bước 2: Unrolling (Mở bung mạng):**
   - Ghép các tầng RBM đã học lại với nhau: các tầng xuôi tạo thành **Encoder**, các tầng lộn ngược đối xứng tạo thành **Decoder**.
   - Trọng số ban đầu của Decoder được lấy chính là ma trận chuyển vị của Encoder ($W_d = W_e^T$).
3. **Bước 3: Global Fine-tuning (Tinh chỉnh toàn cục):**
   - Thay thế các phép lấy mẫu ngẫu nhiên bằng giá trị xác suất thực xác định (deterministic real-valued probabilities).
   - Dùng thuật toán lan truyền ngược (Backpropagation) tiêu chuẩn để tinh chỉnh tất cả các trọng số end-to-end nhằm tối thiểu hóa sai số tái tạo.
   - **Vì sao bước này thành công?** Nhờ có Pre-training, trọng số ban đầu đã nằm trong "vùng lân cận" của nghiệm tối ưu (basin of attraction). Gradient descent chỉ cần tinh chỉnh nhẹ là đạt được kết quả phi thường mà không bị kẹt.

---

## 7. Kết Quả Thực Nghiệm Trong Bài Báo

Bài báo kiểm chứng Autoencoder trên 4 bộ dữ liệu với các kết quả áp đảo:

### 7.1. Dữ liệu đường cong tổng hợp (Synthetic Curves)
- Bộ dữ liệu gồm các ảnh $28 \times 28$ sinh ra từ 3 điểm ngẫu nhiên trong không gian 2D (chiều nội tại thực sự là 6).
- Mạng nơ-ron sâu $784 - 400 - 200 - 100 - 50 - 25 - 6$ tái tạo gần như hoàn hảo các đường cong (MSE: **1.44**).
- PCA tuyến tính 6 chiều thất bại nặng nề (MSE: **7.64**). Thậm chí PCA dùng tới 18 chiều vẫn có sai số lớn hơn Autoencoder 6 chiều (MSE: **2.45** so với 1.44).

### 7.2. Chữ số viết tay MNIST
- Kiến trúc mạng: $784 - 1000 - 500 - 250 - 30$ (nén về 30 chiều).
- Sai số tái tạo bình phương trung bình:
  - Deep Autoencoder 30D: **3.00**
  - Logistic PCA 30D: **8.01**
  - Standard PCA 30D: **13.87**
- **Trực quan hóa 2 chiều (Figure 3 trong bài báo)**:
  - Khi nén về 2 chiều ($784 \to \dots \to 2$): Autoencoder tách biệt rõ ràng các cụm chữ số $0, 1, \dots, 9$ thành các dải uốn lượn tuyệt đẹp trên mặt phẳng 2D.
  - Ngược lại, 2 thành phần chính đầu tiên của PCA đè các chữ số lẫn lộn vào một đám đông không thể phân tách.

### 7.3. Khuôn mặt Olivetti Faces
- Ảnh mặt người xám $25 \times 25$ ($625$ chiều), mạng nén về 30 chiều: $625 - 2000 - 1000 - 500 - 30$.
- Autoencoder khôi phục được góc nghiêng, biểu cảm và chi tiết khuôn mặt với độ sắc nét vượt trội so với ảnh mờ của PCA.

### 7.4. Truy xuất tài liệu văn bản (Reuters Corpus)
- 804,414 bản tin Reuters, mỗi bài biểu diễn bằng tần suất của 2000 gốc từ phổ biến nhất.
- Mạng $2000 - 500 - 250 - 125 - 10$ nén văn bản thành mã 10 chiều.
- Dùng độ tương đồng Cosine trên mã 10 chiều này để truy xuất tài liệu: Autoencoder vượt trội hoàn toàn so với **LSA (Latent Semantic Analysis)** - phương pháp trứ danh dựa trên PCA/SVD thời bấy giờ.

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Vì sao ngày nay chúng ta ít dùng RBM Pre-training?
Khoảng năm 2010–2015, cộng đồng AI đã phát hiện ra các kỹ thuật mới cho phép huấn luyện mạng rất sâu trực tiếp từ đầu bằng Backpropagation mà **không cần pre-training bằng RBM**:
1. **Hàm kích hoạt ReLU (Nair & Hinton, 2010)**: Không bị bão hòa ở miền dương, giải quyết căn nguyên triệt tiêu gradient.
2. **Kỹ thuật khởi tạo trọng số Xavier / He (2010, 2015)**: Giữ cho phương sai tín hiệu ổn định qua các tầng.
3. **Batch Normalization (Ioffe & Szegedy, 2015)**: Ổn định phân phối phân lớp trung gian.
4. **Kết nối tắt Residual Connections (ResNet - He et al., 2016)**: Cho phép huấn luyện mạng hàng trăm tầng trực tiếp.
5. **GPU & Dữ liệu khổng lồ**: Tốc độ tính toán tăng hàng nghìn lần.

### 8.2. Các biến thể hiện đại của Autoencoder
| Mô hình | Cơ chế & Mục đích | Ứng dụng tiêu biểu |
|---|---|---|
| **Denoising Autoencoder (DAE)** (2008) | Thêm nhiễu ngẫu nhiên vào đầu vào $\tilde{x}$, ép mạng phục hồi ảnh gốc $x$ | Khử nhiễu, học biểu diễn bất biến |
| **Sparse Autoencoder** | Áp dụng ràng buộc thưa (KL-divergence penalty) lên tầng ẩn | Trích chọn đặc trưng thưa, giống vỏ não |
| **Variational Autoencoder (VAE)** (Kingma 2013) | Ép không gian tiềm ẩn tuân theo phân phối xác suất chuẩn $\mathcal{N}(0, I)$ | Mô hình tạo sinh (Generative AI), sinh ảnh |
| **Convolutional Autoencoder (CAE)** | Thay tầng Dense bằng tầng tích chập Conv2D/ConvTranspose2D | Xử lý ảnh, nén ảnh, phân đoạn ảnh |
| **Masked Autoencoder (MAE)** (He et al. 2022) | Che (mask) 75% ảnh và bắt Vision Transformer tái tạo lại | Self-supervised learning đỉnh cao cho thị giác máy tính |

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục này cung cấp 2 tệp cài đặt mã nguồn hoàn chỉnh:

```text
Day 02 - Autoencoder/
├── README.md                  # Tài liệu lý thuyết chi tiết này
├── autoencoder_scratch.py     # Cài đặt thuần Python & NumPy (RBM, CD-1, Backprop from scratch)
├── autoencoder_scratch.ipynb  # Notebook thực hành tương tác (From Scratch)
├── autoencoder_pytorch.py     # Cài đặt hiện đại bằng PyTorch (Deep AE, 2D Latent viz, So sánh PCA)
├── autoencoder_pytorch.ipynb  # Notebook thực hành tương tác & trực quan hóa PyTorch
└── resources/
    └── Day 2 Autoencoder.pdf
```

### 9.1. Tệp `autoencoder_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python + NumPy.
- **Tính năng nổi bật**:
  - Tự định nghĩa các lớp toán học: `DenseLayer`, `Sigmoid`, `ReLU`, `MSELoss`, `BCELoss`.
  - Cài đặt đầy đủ lớp `RBM` với giải thuật **Contrastive Divergence (CD-1)** theo đúng công thức (2) trong bài báo Hinton 2006.
  - Cài đặt quy trình **Pre-training $\to$ Unrolling $\to$ Fine-tuning bằng Backpropagation thuần**.
  - Tự động sinh dữ liệu đa tạp Synthetic Curves để kiểm thử và in kết quả trực quan ra terminal.

### 9.2. Tệp `autoencoder_pytorch.py` (Cài đặt bằng PyTorch)
- **Công nghệ**: PyTorch, torchvision, matplotlib, scikit-learn.
- **Tính năng nổi bật**:
  - `DeepAutoencoder`: Tái hiện cấu trúc mạng sâu của bài báo ($784 \to 1000 \to 500 \to 250 \to 30/2 \to \dots$).
  - Huấn luyện trên MNIST và so sánh trực quan với **PCA Baseline**.
  - Xuất biểu đồ phân cụm không gian tiềm ẩn 2D (`latent_space_2d.png`) tái hiện Figure 3 của Hinton 2006.
  - Tự động fallback an toàn sang dữ liệu nhân tạo nếu chạy offline không có kết nối internet.

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Tại sao PCA chỉ tìm được biểu diễn tuyến tính, trong khi Autoencoder tìm được biểu diễn phi tuyến tính? Yếu tố kiến trúc nào mang lại khả năng này?**
2. **Trong thuật toán CD-1, vì sao Hinton chỉ cần chạy 1 chuỗi lấy mẫu Gibbs (k=1) thay vì chạy tới khi hội tụ? Sai số xấp xỉ này có làm hỏng quá trình học không?**
3. **Tại sao việc khởi tạo trọng số bằng RBM lại giúp mạng sâu tránh được hiện tượng triệt tiêu gradient trong giai đoạn Fine-tuning?**
4. **Nếu ta đặt số chiều tầng ẩn ở nút cổ chai (bottleneck) lớn hơn số chiều đầu vào (overcomplete autoencoder) mà không có ràng buộc gì, điều gì sẽ xảy ra? Làm sao để khắc phục?**
5. **So sánh sự khác biệt cốt lõi giữa Autoencoder truyền thống (trong bài báo này) và Variational Autoencoder (VAE)?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
