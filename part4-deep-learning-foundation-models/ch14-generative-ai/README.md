# Chương 14: Trí tuệ nhân tạo tạo sinh (Generative AI)

## Mục tiêu chương học
1. **Nắm vững bản chất toán học của mô hình sinh**: Phân biệt rạch ròi giữa mô hình phân biệt (Discriminative) và mô hình sinh (Generative), hiểu bài toán xấp xỉ phân phối xác suất thực $p_{\text{data}}(\mathbf{x})$ thông qua nguyên lý hợp lý cực đại (Maximum Likelihood Estimation).
2. **Khảo sát các mô hình sinh truyền thống**: Phân tích mô hình hỗn hợp Gauss (Gaussian Mixture Models - GMM) qua thuật toán Kỳ vọng - Cực đại hóa (EM), và các mô hình tự hồi quy trực quan (PixelCNN, PixelRNN).
3. **Làm chủ Variational Autoencoders (VAE)**: Dẫn xuất toán học chặt chẽ của Cận dưới bằng chứng (Evidence Lower Bound - ELBO), phân rã sai số tái tạo và độ phân kỳ Kullback-Leibler (KL Divergence), cùng thủ thuật tái tham số hóa (Reparameterization Trick).
4. **Nắm vững mạng đối kháng tạo sinh (GAN)**: Dẫn xuất trò chơi Min-Max hai người chơi, chứng minh nghiệm tối ưu toàn cục dẫn tới độ phân kỳ Jensen-Shannon (JSD), và xử lý các thách thức thực nghiệm (Mode Collapse, Vanishing Gradient).
5. **Làm chủ mô hình khuếch tán xác suất (Diffusion Models / DDPM)**: Dẫn xuất quá trình khuếch tán thuận (Forward Process) với công thức bước nhảy thời gian dạng đóng, quá trình đảo ngược khử nhiễu (Reverse Process), và hàm mục tiêu dự đoán nhiễu $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$.

---

## 1. Giới thiệu về Trí tuệ nhân tạo tạo sinh

### 1.1. Ứng dụng thực tiễn
AI tạo sinh (Generative AI) đại diện cho bước chuyển mình vĩ đại của trí tuệ nhân tạo từ khả năng *nhận biết/phân loại* (Perception) sang *sáng tạo nội dung* (Creation):
- **Thị giác máy tính**: Sinh ảnh chân thực từ văn bản (Midjourney, Stable Diffusion, DALL-E), phục chế ảnh cũ, inpainting, tạo video nhất quán theo thời gian (Sora, Runway Gen-2).
- **Ngôn ngữ và mã nguồn**: Trợ lý đối thoại thông minh (ChatGPT, Claude), sinh mã nguồn tự động, tóm tắt và dịch thuật chuyên sâu.
- **Khoa học sự sống và vật liệu**: Dự đoán cấu trúc và thiết kế chuỗi protein mới (AlphaFold), khám phá phân tử thuốc kháng sinh mới.

### 1.2. Bản chất toán học của bài toán sinh dữ liệu
Cho tập dữ liệu quan sát $\mathcal{D} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(N)}\}$ độc lập cùng phân phối (i.i.d.) được rút ra từ một phân phối thực tế chưa biết $p_{\text{data}}(\mathbf{x})$ trong không gian đa chiều $\mathbb{R}^D$.
- **Mô hình phân biệt (Discriminative Models)**: Ước lượng phân phối xác suất có điều kiện $p(y \mid \mathbf{x})$ để dự đoán nhãn $y$ khi biết đặc trưng $\mathbf{x}$.
- **Mô hình sinh (Generative Models)**: Ước lượng phân phối đồng thời $p(\mathbf{x}, y)$ hoặc phân phối biên $p(\mathbf{x})$, cho phép lấy mẫu các điểm dữ liệu mới $\mathbf{x}_{\text{new}} \sim p_\theta(\mathbf{x})$ trông tự nhiên như thể được rút ra từ $p_{\text{data}}(\mathbf{x})$.

Theo nguyên lý Ước lượng hợp lý cực đại (MLE):
$$\theta^* = \arg\max_\theta \sum_{i=1}^N \log p_\theta(\mathbf{x}^{(i)})$$
Thách thức: Trong không gian dữ liệu nhiều chiều (ví dụ ảnh $256 \times 256 \times 3 \approx 196,608$ chiều), tích phân biên của các biến ẩn (latent variables) $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z}$ thường là một bài toán bất khả tính (intractable).

---

## 2. Mô hình sinh truyền thống

### 2.1. Mô hình hỗn hợp Gauss (Gaussian Mixture Models - GMM)
GMM giả định mỗi điểm dữ liệu $\mathbf{x}$ được sinh ra từ một trong $K$ thành phần phân phối chuẩn đa biến:
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
với trọng số hỗn hợp $\pi_k \ge 0, \sum_{k=1}^K \pi_k = 1$.

#### Thuật toán Kỳ vọng - Cực đại hóa (Expectation-Maximization - EM):
Do không quan sát được biến chỉ số thành phần $z \in \{1, \dots, K\}$, ta sử dụng vòng lặp EM:
1. **Bước E (Expectation)**: Tính xác suất hậu nghiệm (trách nhiệm - responsibility) của cụm $k$ đối với điểm dữ liệu $\mathbf{x}^{(i)}$:
   $$\gamma_{ik} = P(z_i = k \mid \mathbf{x}^{(i)}) = \frac{\pi_k \mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$
2. **Bước M (Maximization)**: Cập nhật các tham số phân phối dựa trên trọng số mềm $\gamma_{ik}$:
   $$\boldsymbol{\mu}_k^{\text{new}} = \frac{\sum_{i=1}^N \gamma_{ik} \mathbf{x}^{(i)}}{N_k}, \quad \boldsymbol{\Sigma}_k^{\text{new}} = \frac{\sum_{i=1}^N \gamma_{ik} (\mathbf{x}^{(i)} - \boldsymbol{\mu}_k)(\mathbf{x}^{(i)} - \boldsymbol{\mu}_k)^T}{N_k}, \quad \pi_k^{\text{new}} = \frac{N_k}{N}$$
   với $N_k = \sum_{i=1}^N \gamma_{ik}$.

### 2.2. Mô hình tự hồi quy (Autoregressive Generative Models)
Khai triển phân phối xác suất đồng thời thông qua quy tắc nhân chuỗi xích xác suất:
$$p(\mathbf{x}) = \prod_{i=1}^D p(x_i \mid x_1, x_2, \dots, x_{i-1})$$
- **PixelRNN / PixelCNN (van den Oord et al., 2016)**: Mô hình hóa từng pixel ảnh dựa trên các pixel đã sinh ở phía trên và bên trái thông qua các lớp tích chập có mặt nạ (Masked Convolutions).
- Ưu điểm: Khả năng tính toán chính xác hàm hợp lý $p(\mathbf{x})$ (Tractability).
- Nhược điểm: Tốc độ sinh dữ liệu cực kỳ chậm do phải lấy mẫu tuần tự từng pixel một theo thời gian $O(D)$.

---

## 3. Variational Autoencoders (VAE)

```
        +--------------+         z ~ N(mu, sigma^2)
x ----> |  Encoder     | ----> [mu, log_var] ----> Reparameterization: z = mu + eps * sigma
        |  q_phi(z|x)  |                                 |
        +--------------+                                 v
                                                   +--------------+
                                                   |  Decoder     | ----> x_hat (Reconstruction)
                                                   |  p_theta(x|z)|
                                                   +--------------+
```

### 3.1. Từ Autoencoder truyền thống (AE) đến VAE
- **Autoencoder cổ điển**: Mã hóa $\mathbf{x} \rightarrow \mathbf{z} = f(\mathbf{x})$ rồi giải mã $\hat{\mathbf{x}} = g(\mathbf{z})$ nhằm cực tiểu hóa lỗi tái tạo $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$. Tuy nhiên, không gian ẩn $\mathbf{z}$ của AE bị phân mảnh và chứa nhiều "vùng chết" (dead zones) không có dữ liệu; nếu ta lấy mẫu một điểm ngẫu nhiên $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, Decoder sẽ sinh ra những hình ảnh vô nghĩa, hỗn loạn.
- **VAE (Kingma & Welling, 2013)**: Ép không gian ẩn $\mathbf{z}$ phải tuân theo một phân phối xác suất liên tục định trước (thường là chuẩn tắc đa biến $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$).

### 3.2. Dẫn xuất toán học của Cận dưới bằng chứng (ELBO)
Ta muốn tối đa hóa $\log p_\theta(\mathbf{x})$. Với phân phối xấp xỉ biến phân $q_\phi(\mathbf{z} \mid \mathbf{x})$ do mạng Encoder dự đoán:
$$\begin{aligned}
\log p_\theta(\mathbf{x}) &= \int q_\phi(\mathbf{z} \mid \mathbf{x}) \log p_\theta(\mathbf{x}) d\mathbf{z} = \int q_\phi(\mathbf{z} \mid \mathbf{x}) \log \left( \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \cdot \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})} \right) d\mathbf{z} \\
&= \int q_\phi(\mathbf{z} \mid \mathbf{x}) \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} d\mathbf{z} + \int q_\phi(\mathbf{z} \mid \mathbf{x}) \log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})} d\mathbf{z} \\
&= \mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x}) + D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p_\theta(\mathbf{z} \mid \mathbf{x}))
\end{aligned}$$

Vì độ phân kỳ KL không bao giờ âm ($D_{\text{KL}} \ge 0$), suy ra:
$$\log p_\theta(\mathbf{x}) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x})$$

Khai triển tiếp $\mathcal{L}_{\text{ELBO}}$:
$$\mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{Tái tạo dữ liệu (Reconstruction)}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}))}_{\text{Chuẩn hóa không gian ẩn (Prior Regularizer)}}$$

Khi $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ và $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, số hạng phân kỳ KL có công thức giải tích đóng tường minh:
$$D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

### 3.3. Thủ thuật tái tham số hóa (Reparameterization Trick)
Phép lấy mẫu ngẫu nhiên $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ là một phép toán ngẫu nhiên bất khả vi, triệt tiêu gradient truyền từ Decoder ngược về Encoder.
**Giải pháp**: Tách nguồn ngẫu nhiên ra khỏi đồ thị tính toán bằng cách rút vector nhiễu chuẩn độc lập $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, sau đó xác định $\mathbf{z}$ thông qua phép biến đổi tất định:
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$
Nhờ đó, gradient có thể lan truyền ngược trực tiếp qua $\boldsymbol{\mu}$ và $\boldsymbol{\sigma}$:
$$\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} = 1, \quad \frac{\partial \mathbf{z}}{\partial \boldsymbol{\sigma}} = \boldsymbol{\epsilon}$$

---

## 4. Mạng đối kháng tạo sinh (Generative Adversarial Networks - GAN)

### 4.1. Cấu trúc trò chơi Min-Max (Goodfellow et al., 2014)
GAN tiếp cận bài toán sinh bằng lý thuyết trò chơi hai người chơi tổng bằng 0 giữa:
- **Bộ sinh (Generator $G_\theta$)**: Nhận vector nhiễu $\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})$, ánh xạ thành mẫu dữ liệu $\mathbf{x}_{\text{fake}} = G(\mathbf{z})$.
- **Bộ phân biệt (Discriminator $D_\phi$)**: Nhận đầu vào $\mathbf{x}$, xuất ra xác suất vô hướng $D(\mathbf{x}) \in [0, 1]$ dự đoán xem $\mathbf{x}$ là ảnh thật từ tập huấn luyện hay ảnh giả từ $G$.

Hàm mục tiêu đối kháng (Value Function):
$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

### 4.2. Nghiệm tối ưu toàn cục và mối liên hệ với Jensen-Shannon Divergence
Với một Generator $G$ cố định sinh ra phân phối $p_g(\mathbf{x})$, đạo hàm $V(D, G)$ theo $D(\mathbf{x})$ và cho bằng 0 cho ta nghiệm tối ưu của bộ phân biệt:
$$D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

Thay $D^*$ ngược trở lại vào hàm giá trị $V(D^*, G)$:
$$\begin{aligned}
V(D^*, G) &= \int p_{\text{data}}(\mathbf{x}) \log \frac{p_{\text{data}}(\mathbf{x})}{\frac{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}{2}} d\mathbf{x} + \int p_g(\mathbf{x}) \log \frac{p_g(\mathbf{x})}{\frac{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}{2}} d\mathbf{x} - 2\log 2 \\
&= D_{\text{KL}}\left( p_{\text{data}} \,\|\, \frac{p_{\text{data}} + p_g}{2} \right) + D_{\text{KL}}\left( p_g \,\|\, \frac{p_{\text{data}} + p_g}{2} \right) - \log 4 \\
&= 2 \cdot D_{\text{JS}}(p_{\text{data}} \,\|\, p_g) - \log 4
\end{aligned}$$

Vì khoảng cách Jensen-Shannon $D_{\text{JS}} \ge 0$ và đạt cực tiểu bằng $0$ khi và chỉ khi hai phân phối trùng khít hoàn toàn:
$$p_g(\mathbf{x}) \equiv p_{\text{data}}(\mathbf{x}) \iff V(D^*, G^*) = -\log 4 \approx -1.386$$
Khi đó, $D^*(\mathbf{x}) = \frac{1}{2}$, nghĩa là bộ phân biệt hoàn toàn bối rối và chỉ đoán mò ngẫu nhiên.

### 4.3. Các thách thức kỹ thuật khi huấn luyện GAN
1. **Biến mất Gradient ở Generator (Vanishing Gradients)**: Khi $D$ quá mạnh so với $G$, $D(G(\mathbf{z})) \to 0$, dẫn đến $\log(1 - D(G(\mathbf{z}))) \to 0$ và đạo hàm bị bão hòa. Khắc phục bằng hàm mục tiêu không bão hòa (Non-saturating heuristic): $G$ tối đa hóa $\log D(G(\mathbf{z}))$.
2. **Sụp đổ chế độ (Mode Collapse)**: Generator chỉ học cách sinh ra một vài mẫu dữ liệu cực kỳ thuyết phục mà Discriminator luôn chấp nhận, hoàn toàn bỏ qua tính đa dạng của phân phối dữ liệu thật.
3. **Mất ổn định khi tối ưu**: Vòng lặp cập nhật gradient descent đồng thời cho hai người chơi dễ rơi vào dao động vô tận (limit cycles) thay vì hội tụ về điểm cân bằng Nash (Nash Equilibrium).

---

## 5. Mô hình khuếch tán xác suất (Diffusion Models / DDPM)

Mô hình khuếch tán (Sohl-Dickstein et al., 2015; Ho et al., 2020) giải quyết triệt để vấn đề sụp đổ mốt của GAN và tạo ra chất lượng hình ảnh vượt trội bằng cách phân rã bài toán sinh dữ liệu thành một chuỗi $T$ bước khử nhiễu vi mô.

```
x_0 --------> x_1 --------> ... --------> x_{T-1} --------> x_T ~ N(0, I)
     q(x_1|x_0)    q(x_2|x_1)               q(x_T|x_{T-1})
                    QUÁ TRÌNH KHUẾCH TÁN THUẬN (FORWARD NOISING)
 <-------------------------------------------------------------------------
                     QUÁ TRÌNH KHỬ NHIỄU NGƯỢC (REVERSE DENOISING)
     p_theta(x_0|x_1)                      p_theta(x_{T-1}|x_T)
```

### 5.1. Quá trình khuếch tán thuận (Forward Noising Process)
Là một xích Markov tăng dần nhiễu Gauss vào dữ liệu gốc $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ qua các bước thời gian $t = 1, \dots, T$:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$
với lịch trình phương sai phương sai cố định $\beta_1 < \beta_2 < \dots < \beta_T \in (0, 1)$.

**Công thức bước nhảy thời gian dạng đóng (Closed-form sampling)**:
Đặt $\alpha_t = 1 - \beta_t$ và $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Ta có thể trực tiếp lấy mẫu $\mathbf{x}_t$ tại thời điểm bất kỳ mà không cần lặp lại $t$ bước:
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
Khi $T \to \infty$ (thường chọn $T = 1000$), $\bar{\alpha}_T \approx 0$, dẫn đến $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (nhiễu trắng đẳng hướng).

### 5.2. Quá trình khử nhiễu ngược (Reverse Denoising Process)
Bắt đầu từ nhiễu thuần túy $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, mô hình học cách đảo ngược bước thời gian để phục hồi dữ liệu ban đầu:
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$

### 5.3. Hàm mất mát tối ưu hóa xấp xỉ nhiễu (DDPM Loss)
Bằng cách tối ưu hóa cận dưới biến phân ELBO, Ho et al. (2020) chứng minh rằng việc dự đoán kỳ vọng $\boldsymbol{\mu}_\theta$ có thể quy đổi tương đương về bài toán huấn luyện một mạng nơ-ron $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ (thường là U-Net với time embeddings) để dự đoán chính xác vector nhiễu $\boldsymbol{\epsilon}$ đã được thêm vào:
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t \right) \right\|^2 \right]$$
Trong đó $t$ được chọn ngẫu nhiên đều từ $\{1, \dots, T\}$, và $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

---

## 6. Tài liệu tham khảo chính thống
1. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. *Deep Learning*, MIT Press. (Chương 20: Deep Generative Models).
2. **Bishop, C. M. (2006)**. *Pattern Recognition and Machine Learning*, Springer. (Chương 9: Mixture Models and EM).
3. **Kingma, D. P., & Welling, M. (2013)**. *Auto-Encoding Variational Bayes* (VAE). arXiv:1312.6114.
4. **Goodfellow, I., et al. (2014)**. *Generative Adversarial Nets* (GAN). NeurIPS 2014.
5. **Ho, J., Jain, A., & Abbeel, P. (2020)**. *Denoising Diffusion Probabilistic Models* (DDPM). NeurIPS 2020.
6. **Sohl-Dickstein, J., et al. (2015)**. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015.

---

## 7. Định hướng thực hành (`demo.ipynb`)
Trong tệp `demo.ipynb`, chúng ta sẽ trực tiếp hiện thực hóa 3 mô hình sinh kinh điển từ các phép tính tensor toán học cơ bản:
1. **Variational Autoencoder (VAE) từ đầu**: Xây dựng kiến trúc Encoder, Decoder, triển khai cơ chế Tái tham số hóa (Reparameterization Trick) và hàm mất mát kết hợp giữa Reconstruction Loss và phân kỳ KL Divergence; sinh các điểm dữ liệu mới từ không gian ẩn.
2. **Mạng đối kháng tạo sinh GAN đồ chơi (Toy 1D/2D GAN)**: Cài đặt Generator và Discriminator học cách mô phỏng phân phối Gaussian 1D/2D từ nhiễu đồng nhất, quan sát quá trình tiến hóa phân phối của Generator tiến về dữ liệu thật.
3. **Mô phỏng quá trình khuếch tán thuận (Forward Diffusion Schedule)**: Lập trình công thức đóng $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$ để mô phỏng sự suy biến thông tin qua các mốc thời gian $t = 0, 50, 200, 500, 999$.
