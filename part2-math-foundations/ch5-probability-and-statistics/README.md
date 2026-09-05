# Chương 5: Xác suất và Thống kê cho Học máy (Probability and Statistics for Machine Learning)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Nắm vững bản chất Biến ngẫu nhiên:** Phân biệt biến ngẫu nhiên rời rạc và liên tục; làm chủ các hàm phân phối xác suất PMF, PDF và CDF.
2. **Thấu hiểu nền tảng lý thuyết xác suất:** Phân biệt hai góc nhìn Tần suất (Frequentist) vs. Bayes (Bayesian); vận dụng thành thạo các quy tắc xác suất đồng thời, biên, có điều kiện và Định lý Bayes.
3. **Phân tích đặc trưng thống kê của dữ liệu:** Tính toán kỳ vọng, phương sai, hiệp phương sai và hiểu bản chất ma trận hiệp phương sai $\boldsymbol{\Sigma}$ trong không gian đa chiều.
4. **Làm chủ các phân phối xác suất kinh điển:** Phân tích phân phối Bernoulli, Categorical, Gaussian 1 chiều và Gaussian đa biến (Multivariate Gaussian); giải thích nguồn gốc xác suất của các hàm mất mát trong học máy.
5. **Ứng dụng Định lý Bayes và thuật toán Naive Bayes:** Giải quyết bài toán chẩn đoán y khoa (nghịch lý tỷ lệ nền) và xây dựng bộ phân loại văn bản Naive Bayes Classifier.
6. **Thành thạo hai phương pháp ước lượng tham số MLE và MAP:** Chứng minh bằng toán học mối liên hệ mật thiết giữa Ước lượng cực đại hợp lý (MLE) với hàm mất mát, và giữa Ước lượng cực đại hậu nghiệm (MAP) với các kỹ thuật Chính quy hóa ($L_1, L_2$ Regularization).
7. **Kết nối Lý thuyết thông tin và Hàm mất mát:** Nắm vững Entropy của Shannon, Cross-Entropy và Phân kỳ Kullback-Leibler (KL Divergence) – chìa khóa tối ưu hóa của mạng nơ-ron hiện đại.

---

## 5.1 Biến ngẫu nhiên (Random Variables)

### 5.1.1 Khái niệm Biến ngẫu nhiên
Một **biến ngẫu nhiên (Random Variable - RV)** $X$ là một hàm số đo lường ánh xạ từ không gian mẫu các biến cố $\Omega$ sang tập số thực $\mathbb{R}$:
$$X: \Omega \to \mathbb{R}$$
Biến ngẫu nhiên không phải là một biến số thông thường, mà là một quy tắc gán giá trị số cho kết quả của một hiện tượng ngẫu nhiên.

### 5.1.2 Biến ngẫu nhiên rời rạc và Hàm khối xác suất (PMF)
- $X$ nhận các giá trị trong tập đếm được $\{x_1, x_2, \dots\}$.
- **Hàm khối xác suất (Probability Mass Function - PMF):** $p(x) = P(X = x)$ thỏa mãn:
  $$p(x) \ge 0, \quad \forall x \in \text{Val}(X) \qquad \text{và} \qquad \sum_{x \in \text{Val}(X)} p(x) = 1$$

### 5.1.3 Biến ngẫu nhiên liên tục và Hàm mật độ xác suất (PDF)
- $X$ nhận giá trị trên một khoảng liên tục trong $\mathbb{R}$.
- **Hàm mật độ xác suất (Probability Density Function - PDF):** $p(x)$ thỏa mãn $p(x) \ge 0$ và $\int_{-\infty}^{\infty} p(x) \, dx = 1$.
- *Lưu ý quan trọng:* Với biến liên tục, xác suất tại một điểm chính xác luôn bằng $0$: $P(X = x) = 0$. Xác suất chỉ có ý nghĩa trên một khoảng $[a, b]$:
  $$P(a \le X \le b) = \int_a^b p(x) \, dx$$

### 5.1.4 Hàm phân phối tích lũy (Cumulative Distribution Function - CDF)
Xác định xác suất biến ngẫu nhiên $X$ nhận giá trị nhỏ hơn hoặc bằng $x$:
$$F_X(x) = P(X \le x) = \begin{cases} \sum_{x_i \le x} p(x_i) & \text{nếu } X \text{ rời rạc} \\ \int_{-\infty}^x p(t) \, dt & \text{nếu } X \text{ liên tục} \end{cases}$$
Tính chất: $F_X(x)$ là hàm không giảm, $\lim_{x \to -\infty} F_X(x) = 0$ và $\lim_{x \to \infty} F_X(x) = 1$.

---

## 5.2 Nền tảng Lý thuyết Xác suất

### 5.2.1 Hai góc nhìn: Tần suất (Frequentist) vs. Bayes (Bayesian)
- **Trường phái Tần suất:** Xác suất là giới hạn của tần suất xuất hiện một biến cố khi số lần thử nghiệm tiến tới vô hạn: $P(A) = \lim_{N \to \infty} \frac{n_A}{N}$. Các tham số mô hình $\theta$ là các hằng số cố định nhưng chưa biết.
- **Trường phái Bayes:** Xác suất biểu diễn mức độ tin tưởng (Degree of belief) của con người về một biến cố dựa trên thông tin hiện có. Tham số $\theta$ bản thân nó cũng là một biến ngẫu nhiên tuân theo một phân phối xác suất.

### 5.2.2 Tiên đề xác suất Kolmogorov (1933)
1. **Tính không âm:** $P(E) \ge 0$ với mọi biến cố $E \subseteq \Omega$.
2. **Tính chuẩn hóa:** $P(\Omega) = 1$.
3. **Tính cộng tính đếm được:** Nếu $E_1, E_2, \dots$ là các biến cố đôi một xung khắc ($E_i \cap E_j = \emptyset$):
   $$P\left(\bigcup_{i=1}^\infty E_i\right) = \sum_{i=1}^\infty P(E_i)$$

### 5.2.3 Xác suất đồng thời (Joint) và Xác suất biên (Marginal)
- **Xác suất đồng thời $P(X = x, Y = y)$:** Xác suất để $X=x$ VÀ $Y=y$ cùng xảy ra.
- **Quy tắc tính Xác suất biên (Sum Rule / Marginalization):**
  $$P(X = x) = \sum_{y} P(X = x, Y = y) \quad (\text{rời rạc}) \qquad \text{hoặc} \qquad p(x) = \int p(x, y) \, dy \quad (\text{liên tục})$$

### 5.2.4 Xác suất có điều kiện (Conditional Probability)
Xác suất của biến cố $Y$ khi biết biến cố $X$ đã xảy ra:
$$P(Y = y \mid X = x) = \frac{P(X = x, Y = y)}{P(X = x)}, \quad \text{với } P(X = x) > 0$$

### 5.2.5 Quy tắc nhân và Quy tắc Xác suất toàn phần
- **Quy tắc nhân (Product Rule):**
  $$P(X, Y) = P(Y \mid X) P(X) = P(X \mid Y) P(Y)$$
  Mở rộng chuỗi cho $n$ biến: $P(X_1, \dots, X_n) = P(X_1) P(X_2 \mid X_1) \dots P(X_n \mid X_1, \dots, X_{n-1})$.
- **Quy tắc xác suất toàn phần (Law of Total Probability):** Nếu các tập hợp $B_1, \dots, B_k$ tạo thành một phân hoạch của không gian mẫu $\Omega$:
  $$P(A) = \sum_{i=1}^k P(A \mid B_i) P(B_i)$$

### 5.2.6 Định lý Bayes (Bayes' Theorem)
Được công bố bởi Thomas Bayes (1763), định lý thiết lập công thức đảo ngược xác suất có điều kiện:
$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})} = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{\sum_{\theta'} P(\mathcal{D} \mid \theta') P(\theta')}$$

```
                Likelihood (Độ hợp lý)   Prior (Tiên nghiệm)
                        │                     │
                        ▼                     ▼
               P(θ | D) =  P(D | θ)  x  P(θ)
                           ─────────────────
                                 P(D)
                                  ▲
                                  │
                           Evidence (Chứng cứ)
```

- **Tiên nghiệm $P(\theta)$ (Prior):** Mức độ tin tưởng ban đầu về tham số $\theta$ trước khi quan sát dữ liệu.
- **Độ hợp lý $P(\mathcal{D} \mid \theta)$ (Likelihood):** Khả năng quan sát thấy dữ liệu $\mathcal{D}$ nếu tham số nhận giá trị $\theta$.
- **Chứng cứ $P(\mathcal{D})$ (Evidence / Marginal Likelihood):** Hằng số chuẩn hóa đảm bảo tổng xác suất bằng 1.
- **Hậu nghiệm $P(\theta \mid \mathcal{D})$ (Posterior):** Niềm tin được cập nhật về $\theta$ sau khi đã tiếp nhận dữ liệu thực nghiệm $\mathcal{D}$.

---

## 5.3 Đặc trưng dữ liệu (Moments and Independence)

### 5.3.1 Kỳ vọng toán học (Expected Value)
Kỳ vọng là giá trị trung bình theo xác suất của một biến ngẫu nhiên hoặc hàm số $g(X)$:
$$\mathbb{E}[X] = \begin{cases} \sum_x x \, p(x) & \text{(rời rạc)} \\ \int_{-\infty}^\infty x \, p(x) \, dx & \text{(liên tục)} \end{cases}, \qquad \mathbb{E}[g(X)] = \int_{-\infty}^\infty g(x) \, p(x) \, dx$$
- Tính chất tuyến tính: $\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$.

### 5.3.2 Phương sai và Độ lệch chuẩn (Variance & Standard Deviation)
Đo lường mức độ phân tán của các giá trị xung quanh giá trị kỳ vọng trung bình $\mu = \mathbb{E}[X]$:
$$\text{Var}(X) = \sigma^2 = \mathbb{E}\left[(X - \mu)^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
Độ lệch chuẩn: $\sigma = \sqrt{\text{Var}(X)}$.  
Tính chất: $\text{Var}(aX + b) = a^2 \text{Var}(X)$.

### 5.3.3 Tính độc lập thống kê (Statistical Independence)
Hai biến ngẫu nhiên $X$ và $Y$ là độc lập thống kê (ký hiệu: $X \perp Y$) khi và chỉ khi phân phối đồng thời bằng tích các phân phối biên:
$$P(X = x, Y = y) = P(X = x) P(Y = y), \quad \forall x, y \iff P(Y \mid X) = P(Y)$$
Nếu $X \perp Y$, thì $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ và $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

### 5.3.4 Hiệp phương sai và Ma trận hiệp phương sai (Covariance Matrix)
- **Hiệp phương sai giữa hai biến $X$ và $Y$:**
  $$\text{Cov}(X, Y) = \sigma_{XY} = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mu_X \mu_Y$$
  - $\text{Cov}(X, Y) > 0$: Hai biến có xu hướng cùng tăng hoặc cùng giảm (đồng biến).
  - $\text{Cov}(X, Y) < 0$: Một biến tăng thì biến kia có xu hướng giảm (nghịch biến).
  - *Lưu ý:* $X \perp Y \implies \text{Cov}(X, Y) = 0$. Tuy nhiên, chiều ngược lại KHÔNG đúng (Hiệp phương sai bằng 0 chỉ thể hiện không có tương quan tuyến tính, nhưng vẫn có thể có quan hệ phi tuyến mạnh mẽ).
- **Hệ số tương quan Pearson:**
  $$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$
- **Ma trận hiệp phương sai (Covariance Matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$):** Cho vector ngẫu nhiên $\mathbf{x} = [X_1, \dots, X_D]^T$:
  $$\boldsymbol{\Sigma} = \mathbb{E}\left[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T\right] = \begin{bmatrix} \text{Var}(X_1) & \text{Cov}(X_1, X_2) & \dots & \text{Cov}(X_1, X_D) \\ \text{Cov}(X_2, X_1) & \text{Var}(X_2) & \dots & \text{Cov}(X_2, X_D) \\ \vdots & \vdots & \ddots & \vdots \\ \text{Cov}(X_D, X_1) & \text{Cov}(X_D, X_2) & \dots & \text{Var}(X_D) \end{bmatrix}$$
  $\boldsymbol{\Sigma}$ luôn là một ma trận đối xứng và bán xác định dương ($\boldsymbol{\Sigma} \succeq 0$).

---

## 5.4 Các phân phối xác suất kinh điển trong Học máy

### 5.4.1 Phân phối Bernoulli (Bernoulli Distribution)
- Biểu diễn một phép thử nhị phân thành công ($x=1$) với xác suất $p \in [0, 1]$ hoặc thất bại ($x=0$) với xác suất $1-p$:
  $$P(X = x; p) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$
- Kỳ vọng: $\mathbb{E}[X] = p$, Phương sai: $\text{Var}(X) = p(1-p)$.
- **Vai trò trong học máy:** Mô hình hóa đầu ra của bài toán phân loại nhị phân (Binary Classification) và hàm kích hoạt Sigmoid.

### 5.4.2 Phân phối Phân loại (Categorical / Multinoulli Distribution)
- Mở rộng Bernoulli cho $K$ kết quả rời rạc có thể có: $\mathbf{x} \in \{0, 1\}^K$ với $\sum_{k=1}^K x_k = 1$ (One-hot encoding):
  $$P(\mathbf{x}; \mathbf{p}) = \prod_{k=1}^K p_k^{x_k}, \quad \text{với } \sum_{k=1}^K p_k = 1$$
- **Vai trò trong học máy:** Mô hình hóa phân loại đa lớp (Multi-class Classification) với hàm kích hoạt Softmax $\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$.

### 5.4.3 Phân phối Chuẩn 1 chiều (Univariate Gaussian / Normal Distribution)
- Phân phối quan trọng nhất trong tự nhiên theo Định lý giới hạn trung tâm (Central Limit Theorem):
  $$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$
- Đường cong hình chuông đối xứng qua $\mu$. Quy tắc thực nghiệm 68–95–99.7:
  - $68.27\%$ dữ liệu nằm trong $[\mu - \sigma, \mu + \sigma]$.
  - $95.45\%$ dữ liệu nằm trong $[\mu - 2\sigma, \mu + 2\sigma]$.
  - $99.73\%$ dữ liệu nằm trong $[\mu - 3\sigma, \mu + 3\sigma]$.

### 5.4.4 Phân phối Chuẩn nhiều chiều (Multivariate Normal Distribution)
- Cho vector ngẫu nhiên $\mathbf{x} \in \mathbb{R}^D$ với vector kỳ vọng $\boldsymbol{\mu} \in \mathbb{R}^D$ và ma trận hiệp phương sai $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$:
  $$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$
- **Khoảng cách Mahalanobis:** Đại lượng $D_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$ đo khoảng cách chuẩn hóa theo hình học của elip hiệp phương sai.
- **Hình dạng của đường mức xác suất (Contours):**
  - $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ (Đẳng hướng - Isotropic): Các đường mức là hình cầu đối xứng hoàn hảo.
  - $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \dots, \sigma_D^2)$ (Đường chéo): Các đường mức là hình elip có trục song song với các trục tọa độ.
  - $\boldsymbol{\Sigma}$ đầy đủ: Các đường mức là hình elip bị xoay nghiêng trong không gian, các trục chính của elip chính là các vector riêng của $\boldsymbol{\Sigma}$.

---

## 5.5 Ứng dụng Định lý Bayes trong Học máy

### 5.5.1 Chẩn đoán Y khoa và Nghịch lý Tỷ lệ nền (Base Rate Fallacy)
Giả sử một căn bệnh hiếm gặp có tỷ lệ mắc trong cộng đồng là $0.1\%$ ($P(B) = 0.001$). Một xét nghiệm có độ nhạy (True Positive Rate) $99\%$ ($P(+ \mid B) = 0.99$) và tỷ lệ dương tính giả (False Positive Rate) $5\%$ ($P(+ \mid \neg B) = 0.05$).  
Hỏi một người có kết quả dương tính thì xác suất thực sự bị bệnh là bao nhiêu?
$$P(B \mid +) = \frac{P(+ \mid B) P(B)}{P(+ \mid B) P(B) + P(+ \mid \neg B) P(\neg B)} = \frac{0.99 \times 0.001}{(0.99 \times 0.001) + (0.05 \times 0.999)} \approx \frac{0.00099}{0.00099 + 0.04995} \approx 1.94\%$$
> **Bài học sâu sắc:** Dù xét nghiệm có vẻ rất chính xác ($99\%$), xác suất một người dương tính thực sự mắc bệnh chỉ chưa đầy $2\%$! Lý do là tỷ lệ nền của bệnh quá hiếm so với tỷ lệ dương tính giả. Đây là nguyên lý sống còn khi ứng dụng AI chẩn đoán bệnh hiếm hoặc phát hiện gian lận tài chính (Fraud Detection).

### 5.5.2 Bộ phân loại Naive Bayes (Naive Bayes Classifier)
Cho vector đặc trưng $\mathbf{x} = [x_1, \dots, x_D]^T$ và nhãn lớp $y \in \{1, \dots, C\}$. Theo định lý Bayes:
$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) P(y)}{P(\mathbf{x})} \propto P(y) P(\mathbf{x} \mid y)$$
- **Giả định "Ngây thơ" (Naive Assumption):** Giả định rằng tất cả các đặc trưng $x_j$ độc lập có điều kiện với nhau khi biết nhãn $y$:
  $$P(\mathbf{x} \mid y) = \prod_{j=1}^D P(x_j \mid y)$$
- **Quy tắc quyết định phân loại (Maximum A Posteriori Rule):**
  $$\hat{y} = \arg\max_{y \in \{1, \dots, C\}} \left[ \log P(y) + \sum_{j=1}^D \log P(x_j \mid y) \right]$$
- **Ưu điểm:** Tốc độ huấn luyện và suy luận siêu nhanh $\mathcal{O}(ND)$, hoạt động tốt bất ngờ trên bài toán lọc email rác (Spam Filtering) và phân tích cảm xúc văn bản ngay cả khi dữ liệu ít.

---

## 5.6 Ước lượng tham số: MLE và MAP

### 5.6.1 Hàm Hợp lý (Likelihood Function)
Cho tập dữ liệu $\mathcal{D} = \{x_1, \dots, x_N\}$ gồm các mẫu độc lập cùng phân phối ($i.i.d.$) từ phân phối $P(x \mid \theta)$. Hàm hợp lý là hàm số của tham số $\theta$:
$$L(\theta) = P(\mathcal{D} \mid \theta) = \prod_{i=1}^N P(x_i \mid \theta)$$
Để thuận tiện tính toán đạo hàm và tránh hiện tượng tràn số dưới (arithmetic underflow) do nhân nhiều số nhỏ hơn 1, ta làm việc với **Hàm log-likelihood**:
$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^N \log P(x_i \mid \theta)$$

### 5.6.2 Ước lượng Cực đại Hợp lý (Maximum Likelihood Estimation - MLE)
MLE tìm bộ tham số $\theta$ sao cho khả năng sinh ra tập dữ liệu quan sát được là cao nhất:
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\min_\theta \left[ -\sum_{i=1}^N \log P(x_i \mid \theta) \right]$$
- **Ví dụ MLE cho Phân phối Chuẩn 1 chiều:** Cho $\mathcal{D} = \{x_1, \dots, x_N\} \sim \mathcal{N}(\mu, \sigma^2)$:
  $$\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi) - \frac{N}{2}\log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (x_i - \mu)^2$$
  Giải hệ phương trình đạo hàm bằng 0:
  $$\frac{\partial \ell}{\partial \mu} = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N x_i \quad (\text{Trung bình mẫu})$$
  $$\frac{\partial \ell}{\partial \sigma^2} = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{\mu})^2 \quad (\text{Phương sai mẫu có chệch})$$

### 5.6.3 Ước lượng Cực đại Hậu nghiệm (Maximum A Posteriori - MAP)
Khi có kiến thức tiên nghiệm $P(\theta)$, MAP tìm tham số làm cực đại hóa xác suất hậu nghiệm $P(\theta \mid \mathcal{D})$:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \mathcal{D}) = \arg\max_\theta [P(\mathcal{D} \mid \theta) P(\theta)] = \arg\max_\theta \left[ \sum_{i=1}^N \log P(x_i \mid \theta) + \log P(\theta) \right]$$

### 5.6.4 Kết nối vĩ đại: MAP chính là MLE kết hợp Chính quy hóa (Regularization)
Giả sử ta huấn luyện mô hình hồi quy tuyến tính $y = \mathbf{w}^T \mathbf{x} + \epsilon$ với nhiễu Gaussian $\epsilon \sim \mathcal{N}(0, \sigma^2)$:
1. **Nếu đặt phân phối tiên nghiệm Gaussian cho trọng số:** $P(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$:
   $$\log P(\mathbf{w}) = -\frac{1}{2\tau^2} \|\mathbf{w}\|_2^2 + \text{const}$$
   Bài toán MAP trở thành:
   $$\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_{\mathbf{w}} \left[ \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \frac{1}{2\tau^2} \|\mathbf{w}\|_2^2 \right] \iff \min_{\mathbf{w}} \left[ \text{MSE} + \lambda \|\mathbf{w}\|_2^2 \right]$$
   > **Kết luận 1:** Chính quy hóa $L_2$ (Ridge Regression / Weight Decay) chính là phép ước lượng MAP với **tiên nghiệm Gaussian**!
2. **Nếu đặt phân phối tiên nghiệm Laplace cho trọng số:** $P(\mathbf{w}) = \prod_j \frac{1}{2b} \exp\left(-\frac{|w_j|}{b}\right)$:
   $$\log P(\mathbf{w}) = -\frac{1}{b} \|\mathbf{w}\|_1 + \text{const}$$
   Bài toán MAP trở thành:
   $$\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_{\mathbf{w}} \left[ \text{MSE} + \lambda \|\mathbf{w}\|_1 \right]$$
   > **Kết luận 2:** Chính quy hóa $L_1$ (Lasso Regression) chính là phép ước lượng MAP với **tiên nghiệm Laplace**!

---

## 5.7 Lý thuyết Thông tin và Hàm mất mát (Information Theory)

### 5.7.1 Lượng thông tin tự thân (Self-Information)
Claude Shannon (1948) định nghĩa lượng thông tin của một biến cố $x$ có xác suất xảy ra $P(x)$ là:
$$I(x) = -\log_2 P(x) = \log_2 \frac{1}{P(x)} \quad (\text{đơn vị: bits})$$
- Một biến cố chắc chắn xảy ra ($P(x) = 1$) mang $0$ bit thông tin.
- Một biến cố càng bất ngờ, hiếm gặp ($P(x) \to 0$) thì khi nó xảy ra, nó mang lại lượng thông tin càng khổng lồ.

### 5.7.2 Entropy Shannon (Độ hỗn loạn thông tin)
Entropy là kỳ vọng lượng thông tin của một biến ngẫu nhiên, đo lường độ bất định trung bình của một phân phối:
$$H(P) = \mathbb{E}_{X \sim P}[I(X)] = -\sum_{x} P(x) \log P(x)$$
- Với biến nhị phân $P(X=1) = p$, Entropy đạt cực đại bằng $1$ bit khi $p = 0.5$ (hoàn toàn khó đoán), và bằng $0$ khi $p=0$ hoặc $p=1$ (hoàn toàn chắc chắn).

### 5.7.3 Cross-Entropy và Phân kỳ Kullback-Leibler (KL Divergence)
- **Cross-Entropy (Entropy chéo):** Đo lượng bit trung bình cần thiết để mã hóa các biến cố từ phân phối thực $P$ bằng cách sử dụng phân phối mô hình $Q$:
  $$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$
- **Phân kỳ KL (Relative Entropy):** Đo khoảng cách (sự khác biệt phi đối xứng) giữa phân phối thực $P$ và phân phối xấp xỉ $Q$:
  $$D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$
- **Bất đẳng thức Gibbs:** $D_{\text{KL}}(P \parallel Q) \ge 0$, và dấu bằng xảy ra khi và chỉ khi $P = Q$.

```
               ┌────────────────────────────────────────┐
               │           H(P, Q) (Cross-Entropy)      │
               ├───────────────────┬────────────────────┤
               │   H(P) (Entropy)  │ D_KL(P || Q) (KL)  │
               │   (Không đổi theo │ (Phần chênh lệch   │
               │    tham số mô hình)│  cần tối thiểu hóa)│
               └───────────────────┴────────────────────┘
```

> **Hệ quả nền tảng cho Học máy:** Vì nhãn dữ liệu thực tế $P$ là cố định nên $H(P)$ là hằng số. Do đó:
> $$\arg\min_\theta H(P, Q_\theta) \equiv \arg\min_\theta D_{\text{KL}}(P \parallel Q_\theta) \equiv \arg\min_\theta \text{NLL}(\theta)$$
> Việc huấn luyện mạng nơ-ron bằng hàm mất mát **Cross-Entropy Loss** chính là việc điều chỉnh mô hình $Q_\theta$ để kéo phân kỳ KL giữa mô hình và phân phối dữ liệu thực về 0!

---

## 5.8 Tài liệu tham khảo (References)

1. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Cambridge University Press. [Tham khảo: `resources/Mathematics for Machine learning.pdf`] (Chương 6: Probability and Distributions).
2. **Murphy, K. P. (2012).** *Machine Learning: A Probabilistic Perspective*. MIT Press. [Tham khảo: `resources/Machine Learning A Probabilistic Perspective.pdf`] (Chương 2: Probability; Chương 3: Generative Models; Chương 5: Bayesian Statistics).
3. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. [Tham khảo: `resources/Pattern Recognition and Machine Learning.pdf`] (Chương 1: Introduction - Probability Theory; Chương 2: Probability Distributions).
4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo: `resources/Deep Learning.pdf`] (Chương 3: Probability and Information Theory).
5. **Vũ Hữu Tiệp (2018).** *Machine Learning Cơ Bản*. NXB Khoa học và Kỹ thuật. [Tham khảo: `resources/Machine Learning Co Ban.pdf`] (Chương 8: Maximum Likelihood and Maximum A Posteriori Estimation).
6. **Shannon, C. E. (1948).** A mathematical theory of communication. *The Bell System Technical Journal*, 27(3), 379-423.

---

## 5.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta cài đặt và trực quan hóa các nguyên lý xác suất thống kê cốt lõi trong Học máy:

1. **Thành phần 1 - Mô phỏng Phân phối và Định lý Giới hạn Trung tâm (CLT):**
   - Lấy mẫu từ phân phối đều (Uniform) và minh họa sự xuất hiện kỳ diệu của đường cong hình chuông Gaussian khi lấy trung bình mẫu với $N \in \{1, 5, 30\}$.
2. **Thành phần 2 - Trực quan hóa Phân phối Chuẩn 2 chiều (2D Gaussian & Covariance Shapes):**
   - Vẽ các đường mức elip (Contours) tương ứng với ma trận hiệp phương sai đẳng hướng, ma trận đường chéo và ma trận hiệp phương sai đầy đủ.
3. **Thành phần 3 - Cài đặt Bộ phân loại Naive Bayes từ đầu (Gaussian Naive Bayes from scratch):**
   - Tự cài đặt thuật toán ước lượng kỳ vọng $\mu_c$, phương sai $\sigma_c^2$ cho từng lớp và suy diễn xác suất hậu nghiệm $P(y \mid \mathbf{x})$.
   - Đánh giá trên bài toán phân loại nhị phân và vẽ ranh giới quyết định xác suất.
4. **Thành phần 4 - Đối chiếu thực nghiệm MLE vs. MAP:**
   - Cài đặt bài toán ước lượng xác suất tung đồng xu (Bernoulli Parameter Estimation).
   - Thử nghiệm khi kích thước mẫu nhỏ ($N=5$) và lớn ($N=100$) để chứng minh: Khi $N$ nhỏ, MAP tránh được overfitting nhờ có tiên nghiệm; khi $N \to \infty$, MAP hội tụ về MLE.
5. **Thành phần 5 - Minh họa trực quan Entropy và Phân kỳ KL:**
   - Vẽ đồ thị Entropy nhị phân $H(p)$ đạt đỉnh tại $p=0.5$.
   - Tính toán và vẽ phân kỳ $D_{\text{KL}}(P \parallel Q)$ giữa phân phối thực và các phân phối mô hình xấp xỉ khác nhau.
