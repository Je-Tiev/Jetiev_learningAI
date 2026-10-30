# Day 08: GAN — Generative Adversarial Nets

> **Bài báo gốc:** *Generative Adversarial Nets*  
> **Tác giả:** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
> **Xuất bản:** Hội nghị *Advances in Neural Information Processing Systems (NeurIPS / NIPS)*, Montreal, Canada, 2014, Trang 2672–2680.  
> **Tài liệu PDF gốc:** [`resources/Day 8 GAN.pdf`](./resources/Day%208%20GAN.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Điểm Nghẽn Của Các Mô Hình Sinh Cũ](#2-bối-cảnh-lịch-sử--điểm-nghẽn-của-các-mô-hình-sinh-cũ)
3. [Ý Tưởng Cốt Lõi: Trò Chơi Đối Nghịch Hai Người (Minimax Game)](#3-ý-tưởng-cốt-lõi-trò-chơi-đối-nghịch-hai-người-minimax-game)
4. [Nền Tảng Toán Học & Sự Hội Tụ Về Jensen-Shannon Divergence](#4-nền-tảng-toán-học--sự-hội-tụ-về-jensen-shannon-divergence)
5. [Đột Phá Kỹ Thuật: Hàm Mất Mát Không Bão Hòa (Non-Saturating Heuristic)](#5-đột-phá-kỹ-thuật-hàm-mất-mát-không-bão-hòa-non-saturating-heuristic)
6. [Quy Trình Huấn Luyện & Giải Thuật Luân Phiên (Alternating SGD)](#6-quy-trình-huấn-luyện--giải-thuật-luân-phiên-alternating-sgd)
7. [Kết Quả Thực Nghiệm Trong Bài Báo](#7-kết-quả-thực-nghiệm-trong-bài-báo)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Vào năm 2014, **Ian Goodfellow** (khi đó là nghiên cứu sinh tiến sĩ dưới sự hướng dẫn của Giáo sư Yoshua Bengio tại Đại học Montreal) đã phát minh ra một ý tưởng làm chấn động toàn bộ giới trí tuệ nhân tạo: **Generative Adversarial Networks (GAN)**.

Trước năm 2014, các mô hình học máy chủ yếu làm rất tốt tác vụ **phân biệt (Discriminative Tasks)** — phân loại xem bức ảnh là con chó hay con mèo. Nhưng khi yêu cầu mô hình **sáng tạo (Generative Tasks)** — tự vẽ ra một bức ảnh con mèo chưa từng tồn tại — kết quả thường là những đốm ảnh nhòe mờ, vô hồn.

Yann LeCun (Giải thưởng Turing 2018) từng nhận định:  
> *"GAN và các biến thể của nó là ý tưởng thú vị và đột phá nhất trong lĩnh vực Học Máy suốt 10 năm qua."*

Khác với tất cả các mô hình sinh truyền thống vốn dựa trên việc ước lượng hàm mật độ xác suất rõ ràng (Explicit Density Estimation), GAN từ bỏ hoàn toàn việc tính toán hàm mật độ. Thay vào đó, GAN thiết lập một **cuộc đối đầu mang tính trò chơi (Game Theory)** giữa hai mạng nơ-ron:
- **Kẻ làm tiền giả (Generator - Mạng Sinh $G$):** Cố gắng tạo ra tiền giả giống thật nhất có thể.
- **Cảnh sát điều tra (Discriminator - Mạng Phân Biệt $D$):** Cố gắng phát hiện ra đâu là tiền thật, đâu là tiền giả.

Qua hàng nghìn vòng đấu trí không ngừng nghỉ, cả hai cùng tiến bộ vượt bậc cho đến khi tiền giả hoàn hảo tới mức cảnh sát chỉ có thể đoán mò với xác suất $50\%$.

```text
       TRUYỀN THỐNG: MÔ HÌNH HỢP LÝ CỰC ĐẠI (MLE / VAE)              ĐỘT PHÁ CỦA GAN (Ian Goodfellow 2014)
┌────────────────────────────────────────────────────────┐     ┌──────────────────────────────────────────────────┐
│ - Ước lượng mật độ tường minh: p(x)                    │     │ - Lấy mẫu ngầm định (Implicit Sampling)          │
│ - Tối đa hóa log-likelihood hoặc xấp xỉ ELBO           │ ==> │ - Trò chơi đối nghịch Minimax giữa G và D        │
│ - Phạt nặng vùng rỗng ==> Ép trung bình cộng các mode  │     │ - Đạt cân bằng Nash khi p_g = p_data             │
│ - HẬU QUẢ: ẢNH SINH RA BỊ MỜ (BLURRY IMAGES)          │     │ - KẾT QUẢ: ẢNH SINH RA CỰC KỲ SẮC NÉT (SHARP)!   │
└────────────────────────────────────────────────────────┘     └──────────────────────────────────────────────────┘
```

---

## 2. Bối Cảnh Lịch Sử & Điểm Nghẽn Của Các Mô Hình Sinh Cũ

### 2.1. Điểm nghẽn của Ước lượng Mật độ Tường minh (Explicit Density Estimation)
Trước GAN, các mô hình sinh thuộc hai nhánh chính:
1. **Mô hình có mật độ giải được (Tractable Density):** PixelRNN, PixelCNN.
   - Sinh ảnh tuần tự từng điểm ảnh (Pixel-by-pixel).
   - Tốc độ sinh ảnh cực kỳ chậm, không thể ứng dụng trong thời gian thực.
2. **Mô hình xấp xỉ mật độ (Approximate Density):**
   - **Variational Autoencoder (VAE):** Dùng suy diễn biến phân tối ưu cận dưới ELBO. Do hàm mất mát điểm ảnh (MSE/BCE) lấy trung bình khoảng cách, VAE thường tạo ra ảnh nhòe mờ ở các đường viền.
   - **Boltzmann Machines (RBM / DBM):** Cần chuỗi Markov Monte Carlo (MCMC) để lấy mẫu ngẫu nhiên. Trong không gian ảnh nhiều chiều, MCMC hội tụ rất chậm và bế tắc tính toán.

### 2.2. Vấn đề "Bình quân hóa các Mode" (Mode Averaging of Maximum Likelihood)
Khi tối đa hóa hàm hợp lý $\mathbb{E}_{x \sim p_{\text{data}}} [\log p_\theta(x)]$, mô hình bị phạt cực nặng nếu gán xác suất thấp cho một mẫu dữ liệu thực tế ($p_\theta(x) \to 0 \implies \log p_\theta(x) \to -\infty$).  
Để an toàn, mô hình chọn giải pháp "dàn mỏng" phân phối trải rộng khắp không gian, bao trùm lên cả các vùng rác giữa hai mode dữ liệu thực. Kết quả là ảnh sinh ra bị nhòe và mất nét.

---

## 3. Ý Tưởng Cốt Lõi: Trò Chơi Đối Nghịch Hai Người (Minimax Game)

GAN giải quyết bài toán sinh dữ liệu thông qua lý thuyết trò chơi tổng bằng không (Zero-Sum Game):

```text
                  KIẾN TRÚC TỔNG THỂ CỦA GENERATIVE ADVERSARIAL NETWORKS (GAN)

  Nhiễu ngẫu nhiên                      Dữ liệu thật x
    z ~ p_z(z)                         (Real Images)
        │                                    │
        ▼                                    │
 ┌──────────────┐                            │
 │  Mạng Sinh   │ ──► Ảnh giả G(z) ──┐       │
 │ Generator G  │    (Fake Image)    │       │
 └──────────────┘                    ▼       ▼
                              ┌──────────────────────┐
                              │   Mạng Phân Biệt     │
                              │   Discriminator D    │
                              └──────────────────────┘
                                         │
                                         ▼
                            D(x) in [0, 1] (Thật hay Giả?)
                            - D(Real) -> 1  (Đúng là thật!)
                            - D(Fake) -> 0  (Bắt quả tang giả!)
```

### 3.1. Mạng Sinh (Generator $G$)
- Đầu vào: Vector nhiễu ngẫu nhiên $z \sim p_z(z)$ (thường là phân phối đều $\mathcal{U}[-1, 1]$ hoặc Gauss $\mathcal{N}(0, I)$).
- Đầu ra: Mẫu dữ liệu giả lập $G(z; \theta_g)$ có cùng kích thước và số kênh như dữ liệu thật.
- Mục tiêu: Đánh lừa mạng $D$, khiến $D(G(z)) \to 1$.

### 3.2. Mạng Phân Biệt (Discriminator $D$)
- Đầu vào: Một mẫu dữ liệu $x$ (có thể là ảnh thật từ tập dữ liệu hoặc ảnh giả từ $G$).
- Đầu ra: Giá trị vô hướng $D(x; \theta_d) \in [0, 1]$ biểu thị xác suất mẫu $x$ là ảnh thật chứ không phải ảnh sinh ra từ $G$.
- Mục tiêu: Nhận diện chính xác ảnh thật ($D(x) \to 1$) và vạch trần ảnh giả ($D(G(z)) \to 0$).

---

## 4. Nền Tảng Toán Học & Sự Hội Tụ Về Jensen-Shannon Divergence

### 4.1. Hàm Mục Tiêu Minimax
Hàm giá trị $V(D, G)$ của trò chơi hai người được định nghĩa:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log (1 - D(G(z))) \right]$$

- Khi tối đa hóa theo $D$: $D$ học cách gán giá trị cao cho ảnh thật và giá trị thấp cho ảnh giả.
- Khi tối thiểu hóa theo $G$: $G$ học cách tạo ra ảnh sao cho $D(G(z))$ cao nhất có thể (để $\log(1 - D(G(z)))$ nhỏ nhất).

### 4.2. Mạng Phân Biệt Tối Ưu $D^*_G(x)$
Với bất kỳ mạng sinh $G$ cố định nào, ta tìm $D$ cực đại hóa $V(D, G)$:

$$V(D, G) = \int_x \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$

Lấy đạo hàm biểu thức bên trong tích phân theo $D(x)$ và cho bằng $0$:

$$\frac{p_{\text{data}}(x)}{D(x)} - \frac{p_g(x)}{1 - D(x)} = 0 \implies D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

> 💡 **Ý nghĩa:**  
> - Nếu $p_g(x) = 0$ (mô hình chưa học được vùng dữ liệu này): $D^*(x) = 1$.  
> - Nếu $p_g(x) = p_{\text{data}}(x)$ (mô hình sinh hoàn hảo): $D^*(x) = \frac{p_{\text{data}}}{2 p_{\text{data}}} = \frac{1}{2}$.  
> Khi đó $D$ hoàn toàn bị bối rối và chỉ có thể đoán mò với xác suất $50\%$!

### 4.3. Liên Hệ Chặt Chẽ Với Độ Phân Kỳ Jensen-Shannon (JSD)
Thay nghiệm tối ưu $D^*_G$ ngược trở lại hàm mục tiêu $V(D^*_G, G)$:

$$V(D^*_G, G) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g} \left[ \log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)} \right]$$

Biến đổi đại số bằng cách cộng trừ $\log 2$:

$$V(D^*_G, G) = -\log 4 + D_{\text{KL}}\left( p_{\text{data}} \parallel \frac{p_{\text{data}} + p_g}{2} \right) + D_{\text{KL}}\left( p_g \parallel \frac{p_{\text{data}} + p_g}{2} \right)$$
$$V(D^*_G, G) = -\log 4 + 2 \cdot D_{\text{JS}}(p_{\text{data}} \parallel p_g)$$

Trong đó $D_{\text{JS}}$ là **Độ phân kỳ Jensen-Shannon (Jensen-Shannon Divergence)**:
- $D_{\text{JS}} \ge 0$ và đối xứng ($D_{\text{JS}}(p \parallel q) = D_{\text{JS}}(q \parallel p)$).
- $D_{\text{JS}}(p_{\text{data}} \parallel p_g) = 0 \iff p_g = p_{\text{data}}$.
- Khi đạt cân bằng Nash toàn cục: $V(D^*, G^*) = -\log 4 \approx -1.3863$.

---

## 5. Đột Phá Kỹ Thuật: Hàm Mất Mát Không Bão Hòa (Non-Saturating Heuristic)

Trong lý thuyết, $G$ tối thiểu hóa $\log(1 - D(G(z)))$.  
Tuy nhiên, trong thực tế giai đoạn đầu huấn luyện:
- $G$ sinh ra ảnh rất xấu, $D$ dễ dàng phân biệt và gán $D(G(z)) \approx 0$.
- Hàm số $\log(1 - D(G(z)))$ tại lân cận $D(G(z)) = 0$ có **độ dốc cực kỳ phẳng (gradient tiệm cận về 0)** $\implies$ **Hiện tượng bão hòa gradient (Vanishing Gradient)** khiến $G$ không thể học được gì!

```text
       HIỆN TƯỢNG TRIỆT TIÊU GRADIENT VÀ GIẢI PHÁP NON-SATURATING LOSS
  Gradient của G
        ▲
        │   /╰─────── Cực đại hóa log D(G(z)) (Non-Saturating Loss)
        │  /           ==> Gradient cực mạnh khi mới bắt đầu huấn luyện!
        │ /
        │/
        ├──────────── Tối thiểu hóa log(1 - D(G(z))) (Minimax Gốc)
        │             ==> Gradient = 0 khi G còn yếu, không học được!
        └────────────────────────────────────────► D(G(z))
       0.0                                      1.0
```

Goodfellow đề xuất một mẹo kỹ thuật vô cùng quan trọng:  
Thay vì tối thiểu hóa $\log(1 - D(G(z)))$, **huấn luyện $G$ để tối đa hóa $\log D(G(z))$** (hoặc tối thiểu hóa $-\log D(G(z))$):

$$\mathcal{L}_G^{\text{Non-Saturating}} = -\mathbb{E}_{z \sim p_z(z)} \left[ \log D(G(z)) \right]$$

Hàm mục tiêu mới này có cùng điểm dừng tối ưu nhưng cung cấp gradient cực mạnh ngay từ những bước đầu tiên, giải cứu quá trình huấn luyện GAN khỏi ngõ cụt.

---

## 6. Quy Trình Huấn Luyện & Giải Thuật Luân Phiên (Alternating SGD)

Huấn luyện GAN là một bài toán tối ưu động luân phiên (Alternating Optimization):

```text
VÒNG LẶP HUẤN LUYỆN GAN (Mỗi Epoch):
┌────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 1: CẬP NHẬT DISCRIMINATOR D (k bước, thường k = 1)                │
│   1. Lấy mẻ m ảnh thật: {x^(1), ..., x^(m)} ~ p_data                   │
│   2. Lấy mẻ m vector nhiễu: {z^(1), ..., z^(m)} ~ p_z                  │
│   3. Tính gradient theo theta_d:                                       │
│      grad_D = nabla_{theta_d} (1/m) sum [ log D(x) + log(1 - D(G(z))) ]│
│   4. Cập nhật theta_d bằng Gradient Ascent                             │
├────────────────────────────────────────────────────────────────────────┤
│ BƯỚC 2: CẬP NHẬT GENERATOR G (1 bước)                                  │
│   1. Lấy mẻ m vector nhiễu mới: {z^(1), ..., z^(m)} ~ p_z              │
│   2. Tính gradient theo theta_g:                                       │
│      grad_G = nabla_{theta_g} (1/m) sum [ -log D(G(z)) ]               │
│   3. Cập nhật theta_g bằng Gradient Descent                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Kết Quả Thực Nghiệm Trong Bài Báo

Trong bài báo gốc năm 2014, nhóm tác giả đã thử nghiệm trên các tập dữ liệu:
1. **MNIST chữ số viết tay:** Sinh ra các chữ số hoàn chỉnh, sắc nét mà không cần bất kỳ giả định thống kê nào.
2. **Toronto Face Database (TFD):** Sinh các khuôn mặt biểu cảm với độ tự nhiên vượt trội so với Deep Boltzmann Machine.
3. **CIFAR-10:** Mặc dù lúc bấy giờ chỉ dùng mạng MLP đa tầng (Fully Connected), GAN đã chứng minh khả năng học các cụm đối tượng phức tạp.
4. **Đánh giá qua Parzen Window Log-Likelihood:** GAN đạt điểm số hợp lý cực đại cạnh tranh sòng phẳng với các mô hình sinh đương thời nhưng với độ sắc nét hình ảnh vượt trội.

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Các Thách Thức Lớn Của GAN Cổ Điển
1. **Sụp đổ Mode (Mode Collapse):** $G$ phát hiện một vài mẫu ảnh dễ lừa $D$ nhất và chỉ sinh đi sinh lại duy nhất các mẫu đó (ví dụ: chỉ vẽ toàn số 1 và số 8, hoàn toàn bỏ quên các chữ số khác).
2. **Bất ổn định huấn luyện (Training Instability):** Nếu $D$ học quá nhanh và áp đảo hoàn toàn, gradient truyền về $G$ bị triệt tiêu. Nếu $D$ quá yếu, $G$ học theo hướng vô nghĩa.
3. **Không có độ đo hội tụ rõ ràng:** Giá trị hàm mất mát của GAN dao động liên tục và không thể dùng để biết mô hình đã hội tụ tốt hay chưa.

### 8.2. Cây Phả Hệ Tiến Hóa Từ GAN (2014 - Nay)
```text
GAN (Goodfellow 2014) - Minimax Game qua MLP thuần túy
     │
     ├──► DCGAN (Radford et al. 2015) - Kiến trúc Convolutional chuẩn hóa (BatchNorm, LeakyReLU)
     │
     ├──► Conditional GAN / cGAN (Mirza 2014) - Sinh ảnh có kiểm soát theo nhãn điều kiện y
     │
     ├──► WGAN / WGAN-GP (Arjovsky 2017) - Khoảng cách Wasserstein chống Mode Collapse
     │
     ├──► Pix2Pix & CycleGAN (Isola / Zhu 2017) - Chuyển đổi phong cách hình ảnh (Image-to-Image)
     │
     └──► StyleGAN 1/2/3 (Karras et al. 2019-2021) - Đỉnh cao sinh chân dung người sắc nét 1024x1024!
```

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục cung cấp hệ thống 4 tệp cài đặt mã nguồn chuẩn mực (From Scratch & PyTorch) cùng báo cáo lý thuyết và PDF gốc:

```text
Day 08 - GAN/
├── README.md                      # Báo cáo lý thuyết chuyên sâu 10 mục chuẩn này
├── gan_scratch.py                 # Cài đặt thuần Python & NumPy (Minimax Game, Non-Saturating Loss & Academic Benchmark)
├── gan_scratch.ipynb              # Notebook thực hành tương tác trực quan (From Scratch)
├── gan_pytorch.py                 # Triển khai GAN PyTorch, sinh phân phối đa mode & đối chuẩn 3 mô hình học thuật
├── gan_pytorch.ipynb              # Notebook thực hành tương tác & trực quan hóa 4 bảng đối chiếu
├── outputs/                       # Thư mục lưu ảnh sinh & biểu đồ đối chiếu học thuật
│   ├── gan_generated_samples.png  # Các mẫu dữ liệu sinh ra từ Generator
│   └── gan_academic_comparison.png# Biểu đồ 4 bảng phân tích học thuật chuyên sâu
└── resources/
    └── Day 8 GAN.pdf              # Toàn văn bài báo gốc NeurIPS 2014
```

### 9.1. Tệp `gan_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python chuẩn + NumPy (không dùng framework học sâu).
- **Tính năng nổi bật**:
  - `LinearLayer`, `LeakyReLULayer`, `SigmoidLayer`, `TanhLayer` với lan truyền ngược giải tích.
  - `GeneratorScratch` & `DiscriminatorScratch`: Hai mạng nơ-ron đối đầu độc lập.
  - Cài đặt **Non-Saturating Loss**: $G$ tối đa hóa $\log D(G(z))$, khắc phục triệt để hiện tượng triệt tiêu gradient.
  - `run_academic_value_comparison()`: **Module đối chiếu giá trị học thuật** so sánh trực diện Mô hình Hợp Lý Cực Đại Cũ (Mode-Averaging MLE / Gaussian Mixture) với GAN.

### 9.2. Tệp `gan_pytorch.py` (Triển khai bằng PyTorch)
- **Công nghệ**: PyTorch, matplotlib, numpy.
- **Tính năng nổi bật**:
  - Module `Generator` và `Discriminator` chuẩn hóa với `nn.LeakyReLU(0.2)`.
  - Cơ chế cập nhật luân phiên hai bộ tối ưu hóa `optim.Adam` độc lập.
  - `run_academic_value_benchmark()`: Khối đối chuẩn 3 trường phái mô hình:
    1. *Mô Hình 1 (Tiền-GAN: Ước lượng mật độ MLE / Gaussian Mixture):* Bị bẫy mode-averaging, làm mờ dữ liệu.
    2. *Mô Hình 2 (GAN Minimax Gốc / Saturating Loss):* Bị triệt tiêu gradient nghiêm trọng trong giai đoạn đầu.
    3. *Mô Hình 3 (GAN Non-Saturating Đột Phá - Goodfellow 2014):* Gradient dồi dào, hội tụ cân bằng Nash tại $D(x) \approx 0.5$.
  - `plot_comprehensive_academic_comparison()`: Tự động lưu biểu đồ 4 bảng phân tích vào `outputs/gan_academic_comparison.png`.

### 9.3. Bảng Tổng Kết Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark)

| Tiêu Chí So Sánh | Mô Hình 1: MLE / Gaussian Mixture | Mô Hình 2: GAN Minimax Gốc (Saturating) | Mô Hình 3: GAN Non-Saturating (Đột Phá) |
|---|:---:|:---:|:---:|
| **Phương pháp tiếp cận** | Ước lượng mật độ tường minh $p(x)$ | Trò chơi Minimax $\log(1 - D)$ | **Mẹo Heuristic $\max \log D(G(z))$** |
| **Đặc tính gradient của G** | Phụ thuộc tích phân | Bị triệt tiêu khi $D$ mạnh | **Mạnh mẽ ngay từ đầu ($\sim 0.3-0.5$)** |
| **Độ sắc nét của mẫu sinh** | Mờ, nhòe viền (Mode-averaging) | Không hội tụ hoặc rất chậm | **Cực kỳ sắc nét, bám sát phân phối thật** |
| **Tốc độ hội tụ** | Chậm khi số chiều lớn | Thường bị treo tại epoch đầu | **Hội tụ ổn định về cân bằng Nash** |
| **Giá trị tiệm cận của D** | Không áp dụng | Dao động mất kiểm soát | **Tiệm cận lý thuyết $D(x) \approx 0.5$** |
| **Bảo toàn đa mode** | Gom trung bình các mode | Dễ sụp đổ mode | **Học bao phủ các mode dữ liệu** |

### 9.4. Hướng Dẫn Chạy & Thực Hành

```bash
# 1. Chạy mô phỏng From Scratch (Thuần Python + NumPy)
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 08 - GAN/gan_scratch.py"

# 2. Chạy mô hình PyTorch & Xuất biểu đồ đối chiếu học thuật
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 08 - GAN/gan_pytorch.py"

# 3. Khởi chạy Jupyter Notebook tương tác trực quan
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 08 - GAN/gan_scratch.ipynb"
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 08 - GAN/gan_pytorch.ipynb"
```

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Phân tích sự khác biệt căn bản giữa mô hình sinh ước lượng mật độ tường minh (Explicit Density) và mô hình sinh ngầm định (Implicit Generative Model) như GAN?**
2. **Chứng minh rằng với một Generator $G$ cố định, mạng phân biệt tối ưu $D^*_G(x)$ có dạng $\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$. Giá trị này mang ý nghĩa gì khi $p_g = p_{\text{data}}$?**
3. **Tại sao việc thay thế hàm mục tiêu ban đầu $\min_G \log(1 - D(G(z)))$ bằng $\max_G \log D(G(z))$ lại giải quyết được hiện tượng triệt tiêu gradient (Vanishing Gradient) ở những epoch đầu tiên?**
4. **Hiện tượng "Sụp đổ Mode" (Mode Collapse) trong GAN là gì? Nguyên nhân lý thuyết trò chơi nào dẫn tới hiện tượng này?**
5. **So sánh ưu điểm và nhược điểm giữa GAN (Goodfellow 2014) và VAE (Kingma & Welling 2013). Vì sao GAN tạo ra ảnh sắc nét hơn VAE nhưng lại khó huấn luyện hơn rất nhiều?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
