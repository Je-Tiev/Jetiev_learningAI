# Day 07: VAE — Auto-Encoding Variational Bayes

> **Bài báo gốc:** *Auto-Encoding Variational Bayes*  
> **Tác giả:** Diederik P. Kingma, Max Welling  
> **Xuất bản:** Hội nghị *International Conference on Learning Representations (ICLR)*, Banff, Canada, 2014 (Công bố tháng 12/2013).  
> **Tài liệu PDF gốc:** [`resources/Day 7 VAE - Auto-Encoding Variational Bayes.pdf`](./resources/Day%207%20VAE%20-%20Auto-Encoding%20Variational%20Bayes.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Điểm Nghẽn Của Mô Hình Tự Mã Hóa Cũ](#2-bối-cảnh-lịch-sử--điểm-nghẽn-của-mô-hình-tự-mã-hóa-cũ)
3. [Nền Tảng Toán Học & Hàm Mục Tiêu ELBO](#3-nền-tảng-toán-học--hàm-mục-tiêu-elbo)
4. [Đột Phá Kỹ Thuật: Reparameterization Trick](#4-đột-phá-kỹ-thuật-reparameterization-trick)
5. [Kiến Trúc Mô Hình Chi Tiết: Encoder - Latent - Decoder](#5-kiến-trúc-mô-hình-chi-tiết-encoder---latent---decoder)
6. [Quy Trình Huấn Luyện & Tối Ưu Hóa](#6-quy-trình-huấn-luyện--tối-ưu-hóa)
7. [Kết Quả Thực Nghiệm Trong Bài Báo](#7-kết-quả-thực-nghiệm-trong-bài-báo)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Vào cuối năm 2013, hai nhà khoa học **Diederik P. Kingma** và **Max Welling** tại Đại học Amsterdam đã công bố bài báo mang tính bước ngoặt: **Auto-Encoding Variational Bayes** (AEVB), giới thiệu kiến trúc **Variational Autoencoder (VAE)**.

Trước khi VAE ra đời:
- **Autoencoder truyền thống (Deterministic Autoencoder):** Chỉ là một công cụ nén dữ liệu. Không gian ẩn (latent space) bị phân mảnh, rời rạc và đầy "lỗ hổng" (holes). Nếu lấy mẫu một điểm ngẫu nhiên trong không gian ẩn, bộ giải mã (decoder) chỉ tái tạo ra ảnh nhiễu vô nghĩa $\rightarrow$ **Không thể dùng để sinh dữ liệu mới**.
- **Mô hình suy diễn biến phân cổ điển (Variational Inference / MCMC):** Không thể mở rộng trên tập dữ liệu lớn vì phép tính tích phân biên (marginal likelihood) hoàn toàn bất khả thi (intractable) trên mạng nơ-ron phi tuyến sâu.

VAE đã tạo ra một cuộc cách mạng khi kết hợp hoàn hảo giữa **Học Sâu (Deep Learning)** và **Thống Kê Bayes Biến Phân (Variational Bayes)**:
1. Đưa ra **Thủ thuật Tái Tham Số Hóa (Reparameterization Trick)** giúp gradient có thể lan truyền ngược qua các phép lấy mẫu ngẫu nhiên.
2. Ép không gian ẩn phải tuân theo một phân phối Gauss chuẩn tắc $\mathcal{N}(0, I)$ liên tục, mượt mà và không có lỗ hổng.
3. Biến Autoencoder từ một bộ nén dữ liệu thuần túy thành một **Mô Hình Sinh (Generative Model)** thực thụ, có khả năng sinh ra các mẫu dữ liệu mới chưa từng tồn tại và nội suy chuyển đổi mượt mà giữa các vật thể.

```text
AUTOENCODER TRUYỀN THỐNG (Hinton 2006)           VARIATIONAL AUTOENCODER (Kingma 2013)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
│ x ──► [Encoder] ──► z (Điểm rời rạc) │     │ x ──► [Encoder] ──► mu(x), logvar(x)         │
│                     │                │     │                         │                    │
│                     ▼                │ ==> │                         ▼ [Reparameterization]│
│         [Decoder] ──► x_hat          │     │                     z ~ N(mu, sigma^2)       │
│                                      │     │                         │                    │
│ ==> Không gian ẩn nứt nẻ, có lỗ hổng │     │                         ▼                    │
│ ==> KHÔNG THỂ LẤY MẪU SINH ẢNH MỚI!  │     │         [Decoder] ──► x_hat                  │
│                                      │     │ ==> Không gian liên tục, mượt mà, sinh ảnh!  │
└──────────────────────────────────────┘     └──────────────────────────────────────────────┘
```

Công trình này cùng với GAN (2014) và Diffusion Models (2020) tạo nên ba trụ cột vĩ đại nhất của kỷ nguyên **AI Tạo Sinh (Generative AI)**.

---

## 2. Bối Cảnh Lịch Sử & Điểm Nghẽn Của Mô Hình Tự Mã Hóa Cũ

### 2.1. Giới hạn của Autoencoder Tất Định (Deterministic Autoencoder)
Trong một Autoencoder chuẩn:
- Mạng mã hóa ánh xạ mỗi ảnh đầu vào $x$ thành một vector điểm duy nhất $z = f_\phi(x)$.
- Bộ giải mã học ánh xạ ngược $x' = g_\theta(z)$ sao cho $\|x - x'\|^2$ nhỏ nhất.

```text
KHÔNG GIAN ẨN CỦA AUTOENCODER CŨ:
      z2 ▲
         │     ● (Cụm chữ số 0)
         │       ●●
         │           [Vùng rỗng / Khoảng trống]
         │           ▲
         │           │ (Lấy mẫu ở đây -> Decoder sinh ảnh rác!)
         │
         │                        ■■
         │                      ■■■ (Cụm chữ số 1)
         └──────────────────────────────────► z1
```

- **Hậu quả:** Không có bất kỳ cơ chế nào ép các cụm biểu diễn phải nằm sát nhau hay lấp đầy không gian. Giữa cụm chữ số $0$ và cụm chữ số $1$ là một khoảng chân không rộng lớn. Nếu lấy mẫu $z \sim \mathcal{N}(0, I)$ rơi vào khoảng chân không này, Decoder chưa từng được huấn luyện tại tọa độ đó và sẽ trả về ảnh nhòe méo mó.

### 2.2. Bài toán Tích Phân Bất Khả Thi (Intractable Marginal Likelihood)
Giả sử dữ liệu $x$ được sinh ra từ biến ẩn $z$ theo phân phối tiền nghiệm $p(z)$ (thường là $\mathcal{N}(0, I)$):
$$p(x) = \int p(x \mid z) p(z) dz$$

Để học tham số mô hình bằng ước lượng hợp lý cực đại (Maximum Likelihood), ta cần tính xác suất hậu nghiệm thực sự:
$$p(z \mid x) = \frac{p(x \mid z) p(z)}{p(x)} = \frac{p(x \mid z) p(z)}{\int p(x \mid z) p(z) dz}$$

- Khi $z$ có nhiều chiều ($D \ge 20$) và hàm $p(x \mid z)$ là một mạng nơ-ron phi tuyến sâu, tích phân trên toàn bộ không gian $\int dz$ là **hoàn toàn bất khả thi** để tính toán giải tích hoặc số học!

---

## 3. Nền Tảng Toán Học & Hàm Mục Tiêu ELBO

Để giải quyết tích phân bất khả thi, Kingma & Welling sử dụng **Suy Diễn Biến Phân (Variational Inference)**:
Dùng một mạng nơ-ron nhận thức $q_\phi(z \mid x)$ (Encoder) để xấp xỉ phân phối hậu nghiệm thực sự $p(z \mid x)$.

### 3.1. Thiết lập Cận Dưới Biến Phân (Evidence Lower Bound - ELBO)
Khoảng cách giữa phân phối xấp xỉ $q_\phi(z \mid x)$ và phân phối thực $p(z \mid x)$ được đo bằng độ phân kỳ Kullback-Leibler (KL Divergence):

$$D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z \mid x)) = \int q_\phi(z \mid x) \log \frac{q_\phi(z \mid x)}{p(z \mid x)} dz$$

Biến đổi đại số dựa trên quy tắc Bayes $p(z \mid x) = \frac{p(x, z)}{p(x)}$:

$$\log p(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x, z)}{q_\phi(z \mid x)} \right] + D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z \mid x))$$

Vì độ phân kỳ KL luôn không âm ($D_{\text{KL}} \ge 0$), ta có:

$$\log p(x) \ge \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(x, z)}{q_\phi(z \mid x)} \right] = \mathcal{L}_{\text{ELBO}}(\theta, \phi; x)$$

Tách $\log p(x, z) = \log p_\theta(x \mid z) + \log p(z)$, ta thu được **Hàm Mục Tiêu Cốt Lõi Của VAE**:

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]}_{\text{Thành phần Tái Tạo (Reconstruction)}} - \underbrace{D_{\text{KL}}\left(q_\phi(z \mid x) \parallel p(z)\right)}_{\text{Thành phần Điều Hòa (KL Regularizer)}}$$

```text
                  CƠ CHẾ GIẰNG CO CỦA HÀM MẤT MÁT ELBO
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Loss = L_Reconstruction            +     D_KL( q(z|x) || N(0, I) )         │
│                                                                              │
│   [Mục tiêu tái tạo sắc nét]                [Mục tiêu điều hòa không gian]   │
│   - Kéo mu(x) tách xa nhau                  - Kéo mu(x) về 0                 │
│   - Bóp sigma(x) về 0                       - Ép sigma(x) về 1               │
│   ==> Tránh nhòe ảnh                        ==> Tránh lỗ hổng không gian     │
│                                                                              │
│   ===> CÂN BẰNG HOÀN HẢO: Không gian liên tục, trơn tru, giải mã sắc nét!    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Công thức Giải Tích của $D_{\text{KL}}$ với Phân Phối Gauss
Giả sử $q_\phi(z \mid x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ và tiền nghiệm $p(z) = \mathcal{N}(0, I)$, độ phân kỳ KL có nghiệm giải tích khép kín cực đẹp:

$$D_{\text{KL}}\left(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I)\right) = -\frac{1}{2} \sum_{j=1}^J \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

---

## 4. Đột Phá Kỹ Thuật: Reparameterization Trick

### 4.1. Bế tắc của phép lấy mẫu ngẫu nhiên
Để tính toán thành phần tái tạo $\mathbb{E}_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)]$, mô hình cần rút mẫu $z \sim q_\phi(z \mid x)$.  
Tuy nhiên, **phép lấy mẫu ngẫu nhiên là một nút không khả vi (non-differentiable)**. Gradient lan truyền ngược từ Decoder bị chặn đứng hoàn toàn tại nút ngẫu nhiên, không thể truyền về cập nhật trọng số cho Encoder!

```text
BẾ TẮC CŨ (Không truyền được gradient):
x ──► [Encoder] ──► (mu, sigma) ──► [ RÚT MẪU NGẪU NHIÊN z ] ──► [Decoder] ──► Loss
                                                  ▲
                                                  │ (BỊ CHẶN! Không có đạo hàm!)
```

### 4.2. Giải pháp: Tách biệt tính ngẫu nhiên ra khỏi đồ thị tính toán
Kingma & Welling nhận ra rằng một biến ngẫu nhiên Gauss $z \sim \mathcal{N}(\mu, \sigma^2)$ có thể được viết lại dưới dạng một **hàm tất định (deterministic function)** của một biến nhiễu ngoại sinh độc lập $\epsilon$:

$$\epsilon \sim \mathcal{N}(0, I)$$
$$z = g_\phi(x, \epsilon) = \mu(x) + \sigma(x) \odot \epsilon$$

Để đảm bảo tính ổn định số học và tránh phương sai âm, mạng Encoder dự đoán $\log(\sigma^2)$ thay vì $\sigma$:
$$z = \mu + \exp\left(\frac{1}{2} \log \sigma^2\right) \odot \epsilon$$

```text
THỦ THUẬT TÁI THAM SỐ HÓA (Reparameterization Trick):
                      epsilon ~ N(0, I) (Nhiễu chuẩn ngoại sinh)
                             │
                             ▼
x ──► [Encoder] ──► mu  ──► (+) ──► z ──► [Decoder] ──► Loss
                 └──► sigma ─► (x) 
                       ▲
                       │ (Gradient lan truyền ngược trơn tru qua mu và sigma!)
```

- **Ý nghĩa:** Biến ngẫu nhiên $\epsilon$ không chứa tham số $\phi$, nên đồ thị tính toán từ $z$ ngược về $\mu$ và $\sigma$ là hoàn toàn liên tục và khả vi! Gradient truyền tự do qua mạng để tối ưu bằng SGD tiêu chuẩn.

---

## 5. Kiến Trúc Mô Hình Chi Tiết: Encoder - Latent - Decoder

```text
           MẠNG NHẬN THỨC (ENCODER)                   MẠNG SINH (DECODER)
                q_phi(z | x)                              p_theta(x | z)

       Ảnh đầu vào x: (Batch, 784)                  Ảnh tái tạo x_hat: (Batch, 784)
                   │                                             ▲
                   ▼                                             │
             [Dense: 400]                                  [Dense: 400]
                   │                                             │
                   ▼                                             │
             ┌─────┴─────┐                                       │
             ▼           ▼                                       │
          mu (20)    logvar (20)                                 │
             │           │                                       │
             └─────┬─────┘                                       │
                   ▼ [z = mu + exp(0.5*logvar) * eps]            │
             Vector ẩn z: (Batch, 20) ───────────────────────────┘
```

1. **Encoder Network ($q_\phi(z \mid x)$):**
   - Tiếp nhận ảnh $x$ (ví dụ: MNIST $28 \times 28 = 784$).
   - Đi qua các tầng ẩn (Dense / Conv + ReLU).
   - Tầng đầu ra phân nhánh thành hai vector kích thước $D$ (ví dụ $D=20$ hoặc $D=2$ để trực quan hóa):
     - Vector kỳ vọng $\mu(x) \in \mathbb{R}^D$.
     - Vector log-phương sai $\log \sigma^2(x) \in \mathbb{R}^D$.
2. **Latent Space (Không gian ẩn):**
   - Rút mẫu $\epsilon \sim \mathcal{N}(0, I)$.
   - Tính $z = \mu + \exp(0.5 \cdot \log \sigma^2) \odot \epsilon$.
3. **Decoder Network ($p_\theta(x \mid z)$):**
   - Tiếp nhận vector $z$.
   - Chiếu qua các tầng ẩn đối xứng.
   - Đầu ra qua hàm Sigmoid trả về xác suất điểm ảnh tái tạo $\hat{x} \in [0, 1]^{784}$.

---

## 6. Quy Trình Huấn Luyện & Tối Ưu Hóa

- **Hàm mất mát tổng thể (Tổng quát hóa):**
  $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{Recon}}(x, \hat{x}) + \beta \cdot D_{\text{KL}}$$
  - Đối với ảnh nhị phân hoặc chuẩn hóa $[0, 1]$ (như MNIST): Dùng **Binary Cross-Entropy (BCE)**:
    $$\mathcal{L}_{\text{Recon}} = -\sum_{i} \left[ x_i \log \hat{x}_i + (1 - x_i) \log (1 - \hat{x}_i) \right]$$
  - Đối với ảnh thực tế liên tục: Dùng **Mean Squared Error (MSE)**.
- **Thuật toán tối ưu:** Thường sử dụng **Adam** hoặc **Adagrad** với tốc độ học $\alpha = 0.001$.
- **Kích thước mẻ (Batch size):** $100$ hoặc $128$ mẫu.

---

## 7. Kết Quả Thực Nghiệm Trong Bài Báo

### 7.1. Cận Dưới Log-Likelihood Trên Tập MNIST & Frey Face

| Mô Hình | Cận Dưới Marginal Likelihood ($\le \log p(x)$) | Ghi Chú |
|---|:---:|---|
| Wake-Sleep Algorithm (Hinton 1995) | $\approx -90.5$ | Không tối ưu trực tiếp cận dưới biến phân |
| Deep Boltzmann Machine (Salakhutdinov) | $\approx -86.3$ | Cần chuỗi Markov MCMC rất chậm |
| **VAE (Kingma & Welling 2014)** | **$\approx -85.7$** | **Vượt trội, huấn luyện song song cực nhanh** |

### 7.2. Khám Phá Cấu Trúc Đa Tạp Ẩn 2D (2D Latent Manifold Visualization)
Khi thiết lập số chiều ẩn $D = 2$ trên tập ảnh chữ số viết tay MNIST:
- Các chữ số tự động sắp xếp theo thứ tự hình học mượt mà: góc trên cùng bên trái là số $1$, dịch dần sang số $7$, chuyển tiếp mượt sang $9$, rồi đến $4$, $0$, $6$, $8$.
- **Khả năng chuyển đổi mượt mà (Smooth Interpolation):** Di chuyển một đường thẳng trong không gian $z$ từ tọa độ của số $3$ sang số $8$ làm ảnh chữ số biến hình liên tục từ số $3$ dần dần mọc thêm nét nối để thành số $8$ hoàn hảo mà không bị đứt gãy.

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Hạn chế của VAE cổ điển
1. **Hiện tượng ảnh sinh bị mờ (Blurry Images):** Do sử dụng phân phối hậu nghiệm Gauss và hàm mất mát pixel-wise (BCE/MSE), mô hình có xu hướng lấy trung bình cộng các khả năng xảy ra, khiến các chi tiết cạnh sắc nét bị nhòe hơn so với GAN.
2. **Hiện tượng triệt tiêu biến ẩn (Posterior Collapse):** Khi kết hợp với Decoder quá mạnh (như PixelCNN hoặc autoregressive models), thành phần $D_{\text{KL}}$ có thể bị kéo hoàn toàn về 0, khiến Decoder bỏ qua hoàn toàn biến ẩn $z$.

### 8.2. Cây phả hệ tiến hóa từ VAE
```text
VAE (Kingma & Welling 2013) - Khởi nguyên suy diễn biến phân qua Reparameterization Trick
     │
     ├──► Conditional VAE / CVAE (Sohn et al. 2015) - Sinh dữ liệu có điều kiện nhãn lớp y
     │
     ├──► beta-VAE (Higgins et al. 2017) - Tách bạch các đặc trưng ngữ nghĩa độc lập (Disentanglement)
     │
     ├──► VQ-VAE / VQ-VAE-2 (van den Oord 2017) - Lượng tử hóa vector rời rạc, triệt tiêu ảnh mờ
     │
     └──► Latent Diffusion Models / Stable Diffusion (Rombach et al. 2022) -
          Sử dụng VAE Encoder để nén ảnh không gian điểm ảnh thành Latent Space, 
          sau đó áp dụng Diffusion Process để sinh ảnh chân thực đỉnh cao!
```

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục cung cấp hệ thống 4 tệp cài đặt mã nguồn chuẩn mực (From Scratch & PyTorch) cùng báo cáo lý thuyết và PDF gốc:

```text
Day 07 - VAE - Auto-Encoding Variational Bayes/
├── README.md                      # Báo cáo lý thuyết chuyên sâu 10 mục chuẩn này
├── vae_scratch.py                 # Cài đặt thuần Python & NumPy (Reparameterization, ELBO, Đạo hàm giải tích & Benchmark)
├── vae_scratch.ipynb              # Notebook thực hành tương tác trực quan (From Scratch)
├── vae_pytorch.py                 # Triển khai VAE PyTorch, sinh lưới đa tạp 2D, đối chuẩn 3 mô hình học thuật
├── vae_pytorch.ipynb              # Notebook thực hành tương tác & trực quan hóa 4 bảng đối chiếu
├── outputs/                       # Thư mục lưu ảnh sinh & biểu đồ đối chiếu học thuật
│   ├── vae_manifold_grid.png      # Lưới 2D nội suy các chữ số sinh ra từ không gian ẩn
│   └── vae_academic_comparison.png# Biểu đồ 4 bảng phân tích học thuật chuyên sâu
└── resources/
    └── Day 7 VAE - Auto-Encoding Variational Bayes.pdf # Toàn văn bài báo gốc ICLR 2014
```

### 9.1. Tệp `vae_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python chuẩn + NumPy (không dùng bất kỳ framework ngoài nào).
- **Tính năng nổi bật**:
  - Tự cài đặt các tầng `Dense` với lan truyền ngược giải tích.
  - Cài đặt **Reparameterization Trick**: $z = \mu + \exp(0.5 \cdot \log \sigma^2) \odot \epsilon$ kèm đạo hàm theo $\mu$ và $\log \sigma^2$.
  - Cài đặt hàm mất mát **ELBO = Reconstruction (BCE) + KL Divergence**.
  - Kiểm chứng khả năng sinh mẫu mới từ phân phối chuẩn tắc $z \sim \mathcal{N}(0, I)$.
  - `run_academic_value_comparison()`: **Module đối chiếu giá trị học thuật** so sánh trực diện Autoencoder cổ điển (Deterministic) với VAE.

### 9.2. Tệp `vae_pytorch.py` (Triển khai bằng PyTorch)
- **Công nghệ**: PyTorch, torchvision, matplotlib.
- **Tính năng nổi bật**:
  - Module `VariationalAutoencoder`: Mạng Encoder dự đoán đồng thời $\mu$ và $\log \sigma^2$, tầng Decoder tái tạo điểm ảnh qua Sigmoid.
  - `generate_manifold_grid()`: Quét lưới tọa độ 2D biến ẩn $[-3, 3] \times [-3, 3]$ và giải mã để sinh bản đồ chữ số biến đổi mượt mà.
  - `run_academic_value_benchmark()`: Khối đối chuẩn 3 trường phái mô hình:
    1. *Mô Hình 1 (Deterministic Autoencoder - Hinton 2006):* Không gian ẩn rỗng, sinh mẫu ngẫu nhiên thất bại hoàn toàn.
    2. *Mô Hình 2 (VAE Không Có KL Penalty / beta=0):* Tái tạo tốt nhưng không gian ẩn phân mảnh.
    3. *Mô Hình 3 (Full Variational Autoencoder - Kingma 2013):* Cân bằng hoàn hảo giữa tái tạo sắc nét và chuẩn hóa không gian ẩn $\mathcal{N}(0, I)$.
  - `plot_comprehensive_academic_comparison()`: Tự động lưu biểu đồ 4 bảng phân tích vào `outputs/vae_academic_comparison.png`.

### 9.3. Bảng Tổng Kết Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark)

| Tiêu Chí So Sánh | Mô Hình 1: Deterministic AE (2006) | Mô Hình 2: VAE Không KL ($\beta=0$) | Mô Hình 3: Full VAE (Kingma 2013) |
|---|:---:|:---:|:---:|
| **Bản chất không gian ẩn** | Các điểm rời rạc cô lập | Cụm phân tán tự do | **Phân phối chuẩn liên tục $\mathcal{N}(0, I)$** |
| **Khả năng sinh mẫu ngẫu nhiên** | **Thất bại hoàn toàn (Ảnh rác)** | Kém (Nhiều vùng mờ) | **Xuất sắc (Mẫu sắc nét, hợp lý)** |
| **Nội suy không gian ẩn (Morphing)**| Đứt gãy đột ngột | Chuyển tiếp giật cục | **Biến đổi mượt mà không góc chết** |
| **Độ phân kỳ KL Divergence** | Không áp dụng | Vô hạn / Không kiểm soát | **Được điều hòa chặt chẽ ($\sim 10-15$)** |
| **Tính khả vi khi rút mẫu** | Không có rút mẫu | Reparameterization Trick | **Reparameterization Trick ($z = \mu + \sigma \epsilon$)** |
| **Ứng dụng hiện đại** | Nén dữ liệu truyền thống | Ít sử dụng | **Nền tảng của CVAE, VQ-VAE & Stable Diffusion** |

### 9.4. Hướng Dẫn Chạy & Thực Hành

```bash
# 1. Chạy mô phỏng From Scratch (Thuần Python + NumPy)
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 07 - VAE - Auto-Encoding Variational Bayes/vae_scratch.py"

# 2. Chạy mô hình PyTorch & Xuất lưới đa tạp 2D + Đối chiếu học thuật
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 07 - VAE - Auto-Encoding Variational Bayes/vae_pytorch.py"

# 3. Khởi chạy Jupyter Notebook tương tác trực quan
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 07 - VAE - Auto-Encoding Variational Bayes/vae_scratch.ipynb"
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 07 - VAE - Auto-Encoding Variational Bayes/vae_pytorch.ipynb"
```

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Tại sao Autoencoder truyền thống (Deterministic Autoencoder) lại thất bại trong vai trò là một mô hình sinh (Generative Model)? Vấn đề của không gian ẩn (Latent Space) cũ là gì?**
2. **Phân tích cơ chế giằng co giữa hai thành phần trong hàm mất mát ELBO: Reconstruction Loss và KL Divergence Loss. Điều gì sẽ xảy ra nếu ta bỏ qua một trong hai thành phần này?**
3. **Giải thích chi tiết về Thủ thuật Tái Tham Số Hóa (Reparameterization Trick). Vì sao phép biến đổi $z = \mu + \sigma \odot \epsilon$ lại cho phép gradient lan truyền ngược qua một quá trình ngẫu nhiên?**
4. **Chứng minh công thức giải tích tính độ phân kỳ KL giữa một phân phối chuẩn đa biến $q(z \mid x) = \mathcal{N}(\mu, \sigma^2)$ và phân phối chuẩn tắc $p(z) = \mathcal{N}(0, I)$.**
5. **Hiện tượng "Posterior Collapse" (Triệt tiêu biến ẩn) trong VAE là gì và các kiến trúc hiện đại (như $\beta$-VAE, VQ-VAE) đã khắc phục nó như thế nào?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
