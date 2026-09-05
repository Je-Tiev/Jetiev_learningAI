# Chương 6: Tối ưu hóa và Giải tích cho Học máy (Optimization and Calculus for Machine Learning)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Thấu hiểu vai trò của tối ưu hóa trong học máy:** Phân biệt tối ưu hóa toán học thuần túy và tối ưu hóa trong ML (mục tiêu tổng quát hóa trên dữ liệu chưa thấy); nắm vững bài toán bình phương tối thiểu và nghiệm dạng đóng (Normal Equation).
2. **Khảo sát đặc tính cực trị trong không gian nhiều chiều:** Phân biệt cực tiểu địa phương, cực tiểu toàn cục; hiểu vì sao **điểm yên ngựa (Saddle Points)** là rào cản chủ yếu trong tối ưu hóa mạng nơ-ron sâu.
3. **Làm chủ giải tích đa biến:** Tính toán đạo hàm riêng, vector Gradient, ma trận Hessian (đạo hàm bậc hai), ma trận Jacobian và vận dụng quy tắc chuỗi (Chain Rule) làm nền tảng cho thuật toán Lan truyền ngược (Backpropagation).
4. **Phân tích hàm lồi và không lồi:** Hiểu định nghĩa hình thức tập lồi, hàm lồi, điều kiện cực trị bậc một và bậc hai ($\nabla^2 f \succeq 0$); hiểu tính chất "mọi cực tiểu cục bộ đều là cực tiểu toàn cục".
5. **Cài đặt và phân tích giải thuật Giảm dần đạo hàm (Gradient Descent):** Nắm vững vai trò của tốc độ học (Learning rate $\eta$), điều kiện hội tụ và hiện tượng dao động phân kỳ.
6. **So sánh toàn diện các biến thể Gradient Descent:** Đối chiếu Batch GD, Stochastic GD (SGD) và Mini-batch GD về chi phí tính toán, mức độ nhiễu và khả năng tận dụng phần cứng GPU.
7. **Làm chủ các thuật toán tối ưu hóa thích ứng nâng cao:** Hiểu cơ chế vật lý của Động lượng (Momentum), RMSProp và giải thuật chuẩn công nghiệp **Adam (Adaptive Moment Estimation)**.

---

## 6.1 Tối ưu hóa trong Học máy (Optimization in ML)

### 6.1.1 Tối ưu hóa truyền thống vs. Tối ưu hóa trong Học máy
- **Tối ưu hóa toán học truyền thống:** Cực tiểu hóa một hàm mục tiêu đã biết hoàn toàn $J(\theta)$ trên miền xác định.
- **Tối ưu hóa trong Học máy:** Ta chỉ có quyền truy cập vào tập dữ liệu hữu hạn $\mathcal{D}_{\text{train}}$. Mục tiêu thực sự không phải là làm cho hàm mất mát trên tập huấn luyện bằng 0, mà là cực tiểu hóa **rủi ro kỳ vọng (Expected Risk)** trên phân phối dữ liệu tổng thể $\mathcal{P}_{\text{data}}$:
  $$R(\theta) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{P}_{\text{data}}} [\mathcal{L}(f_\theta(\mathbf{x}), y)]$$
  Vì $\mathcal{P}_{\text{data}}$ không biết trước, ta tối ưu hóa xấp xỉ thông qua **Rủi ro kinh nghiệm (Empirical Risk)**:
  $$R_{\text{emp}}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

### 6.1.2 Ví dụ kinh điển: Bài toán Hồi quy Tuyến tính (Dự đoán giá nhà)
Cho tập dữ liệu gồm diện tích và số phòng $\mathbf{X} \in \mathbb{R}^{N \times D}$ và giá nhà $\mathbf{y} \in \mathbb{R}^N$. Mô hình tuyến tính: $\hat{y} = \mathbf{X}\mathbf{w}$.
- **Hàm mất mát Mean Squared Error (MSE):**
  $$J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 = \frac{1}{2N} (\mathbf{X}\mathbf{w} - \mathbf{y})^T (\mathbf{X}\mathbf{w} - \mathbf{y})$$
- **Nghiệm dạng đóng (Normal Equation):** Tính đạo hàm và cho bằng 0:
  $$\nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) = \mathbf{0} \implies \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
- **Giới hạn của nghiệm giải tích:** Phép nghịch đảo ma trận $(\mathbf{X}^T \mathbf{X})^{-1}$ có độ phức tạp $\mathcal{O}(D^3)$. Khi số đặc trưng $D$ lên tới hàng triệu hoặc ma trận không khả nghịch, phương pháp giải tích đóng thất bại hoàn toàn. Ta bắt buộc phải sử dụng các **phương pháp tối ưu hóa lặp bằng đạo hàm (Iterative Gradient-Based Optimization)**.

---

## 6.2 Đặc tính Cực trị trong Không gian Nhiều chiều

### 6.2.1 Cực tiểu cục bộ (Local Minima) và Cực tiểu toàn cục (Global Minima)
- Điểm $\mathbf{x}^*$ là **cực tiểu địa phương** nếu tồn tại một lân cận $\epsilon > 0$ sao cho $f(\mathbf{x}^*) \le f(\mathbf{x})$ với mọi $\|\mathbf{x} - \mathbf{x}^*\| < \epsilon$.
- Điểm $\mathbf{x}^*$ là **cực tiểu toàn cục** nếu $f(\mathbf{x}^*) \le f(\mathbf{x})$ với mọi $\mathbf{x} \in \text{dom}(f)$.

### 6.2.2 Khó khăn trong Tối ưu hóa Mạng nơ-ron sâu
1. **Hẻm núi hẹp (Ravines / Ill-conditioned Curvature):** Địa hình dốc đứng theo một chiều nhưng thoai thoải theo chiều khác, khiến thuật toán Gradient Descent dao động ziczac chậm chạp.
2. **Cao nguyên và Vùng phẳng (Plateaux):** Đạo hàm gần bằng 0 trên diện rộng, làm tốc độ học bị tê liệt.
3. **Vách đá dựng đứng (Cliffs):** Gradient tăng vọt đột ngột, có thể đẩy tham số văng xa khỏi vùng nghiệm tốt (gây bùng nổ gradient).

### 6.2.3 Điểm dừng và Điểm yên ngựa (Saddle Points)
- **Điểm dừng (Stationary Point):** Điểm mà tại đó vector gradient triệt tiêu: $\nabla f(\mathbf{x}) = \mathbf{0}$.
- **Điểm yên ngựa (Saddle Point):** Điểm dừng mà tại đó hàm số đạt cực tiểu theo một số hướng nhưng lại đạt cực đại theo các hướng khác.
> **Khám phá bước ngoặt của Deep Learning (Dauphin et al., 2014):** Trong không gian tham số hàng tỷ chiều của mạng nơ-ron sâu, xác suất để tất cả các chiều cùng đạt cực tiểu là $2^{-D} \approx 0$. Do đó, chướng ngại vật nguy hiểm nhất kìm hãm mạng nơ-ron **không phải là cực tiểu cục bộ xấu**, mà là vô số các **điểm yên ngựa** và các cao nguyên bao quanh chúng.

---

## 6.3 Giải tích Nền tảng cho Học máy

### 6.3.1 Đạo hàm một chiều và Xấp xỉ Taylor
- Đạo hàm đo tốc độ biến thiên tức thời: $f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$.
- **Khai triển Taylor bậc 2 quanh điểm $x$:**
  $$f(x + \Delta x) \approx f(x) + f'(x) \Delta x + \frac{1}{2} f''(x) (\Delta x)^2$$

### 6.3.2 Đạo hàm đa biến và Vector Gradient
Cho hàm vô hướng đa biến $f: \mathbb{R}^D \to \mathbb{R}$. Vector Gradient $\nabla f(\mathbf{x}) \in \mathbb{R}^D$ gom tất cả các đạo hàm riêng:
$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_D} \end{bmatrix}$$
- **Ý nghĩa hình học cốt lõi:**
  1. Vector Gradient $\nabla f(\mathbf{x})$ luôn chỉ theo **hướng hàm số tăng nhanh nhất (Direction of Steepest Ascent)**.
  2. Độ dài $\|\nabla f(\mathbf{x})\|_2$ thể hiện tốc độ dốc lớn nhất đó.
  3. Do đó, hướng ngược lại $-\nabla f(\mathbf{x})$ là **hướng hàm số giảm nhanh nhất (Direction of Steepest Descent)**.
  4. Vector Gradient luôn trực giao (vuông góc) với đường mức đồng mức (Level curve / Contour) tại điểm đó.

### 6.3.3 Ma trận đạo hàm bậc hai Hessian
Ma trận Hessian $\mathbf{H} \in \mathbb{R}^{D \times D}$ chứa tất cả các đạo hàm riêng bậc hai:
$$\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$
Theo Định lý Schwarz, nếu các đạo hàm riêng bậc hai liên tục thì $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i} \implies \mathbf{H}$ luôn là một **ma trận đối xứng**.

### 6.3.4 Tính xác định của Hessian và Phân loại Điểm dừng
Tại điểm dừng $\mathbf{x}^*$ thỏa mãn $\nabla f(\mathbf{x}^*) = \mathbf{0}$:
- Nếu $\mathbf{H}(\mathbf{x}^*) \succ 0$ (Xác định dương, mọi trị riêng $\lambda_i > 0$): $\mathbf{x}^*$ là **Cực tiểu địa phương**.
- Nếu $\mathbf{H}(\mathbf{x}^*) \prec 0$ (Âm xác định, mọi trị riêng $\lambda_i < 0$): $\mathbf{x}^*$ là **Cực đại địa phương**.
- Nếu $\mathbf{H}(\mathbf{x}^*)$ có cả trị riêng dương và âm: $\mathbf{x}^*$ là **Điểm yên ngựa**.
- Nếu $\mathbf{H}(\mathbf{x}^*)$ có trị riêng bằng 0 ($\det(\mathbf{H}) = 0$): Chưa đủ kết luận (kiểm tra bậc cao hơn).

### 6.3.5 Ma trận Jacobian cho Hàm Vector
Cho hàm vector $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, ma trận Jacobian $\mathbf{J} \in \mathbb{R}^{m \times n}$ chứa các đạo hàm riêng của từng hàm thành phần:
$$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}, \quad \mathbf{J} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

### 6.3.6 Quy tắc Chuỗi (Chain Rule) và Thuật toán Lan truyền ngược
Cho biến số $z = f(y_1, \dots, y_m)$ và mỗi $y_k = g_k(x_1, \dots, x_n)$:
$$\frac{\partial z}{\partial x_j} = \sum_{k=1}^m \frac{\partial z}{\partial y_k} \frac{\partial y_k}{\partial x_j}$$
Ở dạng ma trận cho chuỗi hàm vector $\mathbf{z} = \mathbf{f}(\mathbf{y}), \mathbf{y} = \mathbf{g}(\mathbf{x})$:
$$\mathbf{J}_{\mathbf{z}}(\mathbf{x}) = \mathbf{J}_{\mathbf{z}}(\mathbf{y}) \mathbf{J}_{\mathbf{y}}(\mathbf{x})$$
> **Bản chất của Lan truyền ngược (Backpropagation):** Chính là việc áp dụng quy tắc chuỗi đệ quy từ đầu ra ngược về đầu vào để tính đạo hàm riêng của hàm mất mát đối với toàn bộ trọng số $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_l}$, tái sử dụng các kết quả trung gian để đạt độ phức tạp tuyến tính $\mathcal{O}(\text{số kết nối})$.

---

## 6.4 Hàm lồi và Hàm không lồi (Convexity)

### 6.4.1 Tập lồi (Convex Set)
Tập hợp $\mathcal{C} \subseteq \mathbb{R}^D$ là tập lồi nếu với mọi cặp điểm $\mathbf{x}, \mathbf{y} \in \mathcal{C}$, đoạn thẳng nối hai điểm đó nằm hoàn toàn bên trong $\mathcal{C}$:
$$\alpha \mathbf{x} + (1 - \alpha)\mathbf{y} \in \mathcal{C}, \quad \forall \alpha \in [0, 1]$$

### 6.4.2 Hàm lồi (Convex Function)
Hàm số $f: \mathcal{C} \to \mathbb{R}$ là lồi nếu với mọi $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ và $\alpha \in [0, 1]$:
$$f(\alpha \mathbf{x} + (1 - \alpha)\mathbf{y}) \le \alpha f(\mathbf{x}) + (1 - \alpha) f(\mathbf{y})$$
*Ý nghĩa hình học:* Đoạn dây cung nối hai điểm bất kỳ trên đồ thị luôn nằm phía trên hoặc trùng với đồ thị của hàm số.

#### Các điều kiện tương đương của hàm lồi khả vi:
1. **Điều kiện bậc một:** Đồ thị hàm số luôn nằm trên mặt phẳng tiếp diện tại mọi điểm:
   $$f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}), \quad \forall \mathbf{x}, \mathbf{y}$$
2. **Điều kiện bậc hai:** Ma trận Hessian luôn là ma trận bán xác định dương tại mọi điểm:
   $$\nabla^2 f(\mathbf{x}) \succeq 0, \quad \forall \mathbf{x} \in \text{dom}(f)$$
3. **Tính chất vàng của tối ưu hóa lồi:** Mọi điểm cực tiểu địa phương đều là **cực tiểu toàn cục duy nhất** (nếu hàm lồi nghiêm ngặt). Không có điểm yên ngựa, không có bẫy cực trị.
- **Ví dụ hàm lồi trong ML:** Hồi quy tuyến tính (MSE), Hồi quy Logistic (Binary Cross-Entropy), Support Vector Machines (Hinge Loss).

### 6.4.3 Hàm không lồi (Non-convex Function)
Hàm mất mát của Mạng nơ-ron nhiều tầng (MLP, CNN, Transformer) là hàm **phi lồi cao độ (Highly Non-convex)** do có sự lồng ghép của các hàm kích hoạt phi tuyến tính và sự hoán vị tương đương của các nơ-ron (Permutation symmetry). Tối ưu hóa phi lồi là bài toán NP-khó trong trường hợp tổng quát.

---

## 6.5 Thuật toán Giảm dần đạo hàm (Gradient Descent - GD)

### 6.5.1 Mã giả và Cơ chế cập nhật
Để tìm $\mathbf{x}^* = \arg\min f(\mathbf{x})$, ta khởi tạo điểm xuất phát $\mathbf{x}_0$ và cập nhật ngược hướng gradient:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$
Trong đó $\eta > 0$ là **tốc độ học (Learning Rate / Step Size)**.

### 6.5.2 Phân tích tác động của Tốc độ học $\eta$
- **$\eta$ quá nhỏ:** Thuật toán bước đi cực kỳ chậm chạp, tốn kém tài nguyên tính toán và dễ bị kẹt vô tận trong các vùng yên ngựa hoặc cao nguyên.
- **$\eta$ vừa phải:** Thuật toán hội tụ êm ái về điểm tối ưu.
- **$\eta$ quá lớn:** Thuật toán nhảy vượt qua đáy thung lũng (Overshooting), dao động dữ dội và cuối cùng phân kỳ ra vô cực ($f(\mathbf{x}_t) \to \infty$).
- **Phân tích hội tụ lý thuyết:** Nếu hàm mục tiêu là $L$-smooth ($\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L \|\mathbf{x} - \mathbf{y}\|$), thuật toán đảm bảo hội tụ đơn điệu nếu chọn:
  $$0 < \eta < \frac{2}{L}$$

---

## 6.6 Ba biến thể kinh điển của Gradient Descent

| Đặc tính | Batch Gradient Descent (BGD) | Stochastic Gradient Descent (SGD) | Mini-batch Gradient Descent |
| :--- | :--- | :--- | :--- |
| **Kích thước dữ liệu mỗi bước** | Toàn bộ tập dữ liệu: $B = N$ | Một mẫu duy nhất: $B = 1$ | Nhóm mẫu nhỏ: $B \in [32, 512]$ |
| **Công thức cập nhật** | $\theta \leftarrow \theta - \frac{\eta}{N} \sum_{i=1}^N \nabla \mathcal{L}_i$ | $\theta \leftarrow \theta - \eta \nabla \mathcal{L}_i$ | $\theta \leftarrow \theta - \frac{\eta}{B} \sum_{i \in \mathcal{B}} \nabla \mathcal{L}_i$ |
| **Độ mượt quỹ đạo** | Rất mượt, đi thẳng về đáy | Dao động ziczac rất mạnh | Dao động vừa phải, có định hướng |
| **Tận dụng phần cứng GPU** | Kém (nguy cơ tràn VRAM) | Rất kém (không song song hóa được) | **Tối ưu tuyệt hảo** (phù hợp Tensor Core) |
| **Khả năng thoát điểm yên ngựa** | Rất kém (dễ tắc tại vùng phẳng) | Rất tốt (nhiễu ngẫu nhiên giúp thoát) | Rất tốt (cân bằng giữa ổn định và nhiễu) |

---

## 6.7 Tối ưu hóa nâng cao: Momentum và Adam

### 6.7.1 Thuật toán Động lượng (Momentum)
Lấy cảm hứng từ vật lý: Một quả bóng lăn xuống dốc sẽ tích lũy động lượng để duy trì vận tốc theo hướng dốc chính và triệt tiêu các dao động ziczac hai bên sườn núi:
$$\mathbf{v}_{t} = \gamma \mathbf{v}_{t-1} + \eta \nabla_\theta J(\theta_t)$$
$$\theta_{t+1} = \theta_t - \mathbf{v}_t$$
- $\gamma \in [0, 1)$ là hệ số ma sát/quán tính (thường chọn $\gamma = 0.9$).
- Giúp tăng tốc gấp nhiều lần ở những vùng dốc phẳng thoai thoải và vượt qua các hẻm núi hẹp.

### 6.7.2 Thuật toán Adam (Adaptive Moment Estimation - Kingma & Ba, 2014)
Adam là thuật toán tối ưu hóa tiêu chuẩn vàng trong huấn luyện mô hình sâu hiện đại. Thuật toán kết hợp ưu điểm của **Momentum** (ước lượng moment bậc 1) và **RMSProp** (chia tỷ lệ theo moment bậc 2 để thích ứng tốc độ học cho từng tham số):

1. **Tính gradient tại bước $t$:** $\mathbf{g}_t = \nabla_\theta J(\theta_t)$
2. **Cập nhật ước lượng trung bình động có trọng số (Moment bậc 1):**
   $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \quad (\text{thường chọn } \beta_1 = 0.9)$$
3. **Cập nhật ước lượng phương sai trung bình động (Moment bậc 2 không định tâm):**
   $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \quad (\text{thường chọn } \beta_2 = 0.999)$$
4. **Hiệu chỉnh chệch (Bias Correction) để triệt tiêu độ lệch về 0 ở các bước đầu:**
   $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
5. **Cập nhật tham số:**
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t \quad (\epsilon \approx 10^{-8})$$
- *Cơ chế tự thích ứng:* Những tham số có gradient lớn thường xuyên sẽ bị ghìm nhỏ bước nhảy; những tham số ít khi xuất hiện (gradient thưa) sẽ được cấp bước nhảy lớn hơn.

---

## 6.8 Tài liệu tham khảo (References)

1. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Cambridge University Press. [Tham khảo: `resources/Mathematics for Machine learning.pdf`] (Chương 5: Vector Calculus; Chương 7: Continuous Optimization).
2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo: `resources/Deep Learning.pdf`] (Chương 4: Numerical Computation; Chương 8: Optimization for Training Deep Models).
3. **Vũ Hữu Tiệp (2018).** *Machine Learning Cơ Bản*. NXB Khoa học và Kỹ thuật. [Tham khảo: `resources/Machine Learning Co Ban.pdf`] (Chương 6: Gradient Descent; Chương 7: Đạo hàm cho ma trận).
4. **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization*. Cambridge University Press. (Giáo trình toàn diện nhất thế giới về tối ưu hóa lồi).
5. **Kingma, D. P., & Ba, J. (2014).** Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*. (Bài báo gốc khai sinh thuật toán Adam).
6. **Dauphin, Y. N., et al. (2014).** Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *NeurIPS*, 27.

---

## 6.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng hệ thống thực nghiệm khảo sát và so sánh trực quan các thuật toán tối ưu hóa:

1. **Thành phần 1 - Thử nghiệm Tốc độ học $\eta$ trên hàm 1 chiều:**
   - Khảo sát 3 trường hợp: $\eta$ quá nhỏ (chậm chạp), $\eta$ tối ưu (hội tụ nhanh), $\eta$ quá lớn (phân kỳ).
2. **Thành phần 2 - Trực quan hóa Địa hình 2D và Đường đi tối ưu hóa:**
   - Sử dụng hàm phi lồi nổi tiếng **Rosenbrock Function** ("Banana valley") hoặc hàm Beale để thử nghiệm.
3. **Thành phần 3 - Cài đặt từ đầu các thuật toán tối ưu hóa:**
   - Tự cài đặt Vanilla Gradient Descent, Momentum, RMSProp, và Adam chỉ bằng `NumPy`.
4. **Thành phần 4 - Đối chiếu quỹ đạo hội tụ trên mặt phẳng đường mức:**
   - Vẽ quỹ đạo di chuyển của Vanilla GD vs. Momentum vs. Adam trên cùng một đồ thị đường mức 2D để thấy rõ khả năng triệt tiêu dao động ngang của Momentum và Adam.
5. **Thành phần 5 - Huấn luyện Hồi quy Tuyến tính so sánh BGD vs. SGD vs. Mini-batch GD:**
   - So sánh tốc độ hội tụ (Loss vs. Wall-clock time / Epochs) và mức độ ổn định của hàm mất mát.
