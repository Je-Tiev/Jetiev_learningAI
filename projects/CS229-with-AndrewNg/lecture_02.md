# Lecture 02: Hồi quy tuyến tính và Gradient Descent

**Khóa học:** Stanford CS229 - Machine Learning (Autumn 2018)  
**Giảng viên:** Andrew Ng  

---

## 1. Quy trình của Supervised Learning

Một bài toán Học có giám sát điển hình diễn ra theo luồng sau:
$$\text{Training Set} \longrightarrow \text{Learning Algorithm} \longrightarrow \text{Hypothesis } h$$
Sau đó, khi có dữ liệu đầu vào mới $x$, ta đưa qua hàm giả thuyết $h$ để nhận được giá trị dự đoán $\hat{y} = h(x)$.

---

## 2. Mô hình Hồi quy tuyến tính (Linear Regression)

Giả sử ta muốn dự đoán giá nhà ($y$) dựa trên diện tích ($x_1$) và số phòng ngủ ($x_2$). Ta xấp xỉ $y$ dưới dạng một hàm tuyến tính của $x$:

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2$$

Trong đó $\theta_i$ là các **tham số (parameters / weights)** của mô hình. Để viết gọn lại dưới dạng tích vô hướng của hai vector, ta quy ước thêm đặc trưng giả $x_0 = 1$:

$$h_\theta(x) = \sum_{j=0}^{n} \theta_j x_j = \theta^T x$$

Với $\theta, x \in \mathbb{R}^{n+1}$.

### Hàm mất mát (Cost Function)
Để tìm ra bộ tham số $\theta$ tốt nhất sao cho $h_\theta(x^{(i)})$ gần với $y^{(i)}$ nhất trên toàn bộ tập huấn luyện, ta định nghĩa hàm mục tiêu bình phương tối thiểu (Least-Squares Cost Function):

$$J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

*(Hệ số $\frac{1}{2}$ được thêm vào để triệt tiêu số $2$ khi lấy đạo hàm).*

Mục tiêu của chúng ta là giải bài toán tối ưu:
$$\min_{\theta} J(\theta)$$

---

## 3. Thuật toán Gradient Descent

Gradient Descent là một thuật toán lặp bắt đầu từ một giá trị $\theta$ khởi tạo (ví dụ $\theta = \vec{0}$), sau đó liên tục cập nhật $\theta$ theo hướng ngược với vector gradient của $J(\theta)$ để giảm dần giá trị hàm mất mát:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

Trong đó $\alpha > 0$ là **tốc độ học (learning rate)**.

### Tính đạo hàm riêng cho một mẫu huấn luyện duy nhất ($m=1$):
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_\theta(x) - y)^2 = (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} \left( \sum_{k=0}^{n} \theta_k x_k - y \right) = (h_\theta(x) - y) x_j$$

Từ đó ta thu được **Quy tắc cập nhật LMS (Least Mean Squares)** hay **Quy tắc Widrow-Hoff**:
$$\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

### Hai biến thể của Gradient Descent:

1. **Batch Gradient Descent:**
   Mỗi bước cập nhật tham số phải duyệt qua **toàn bộ** $m$ mẫu huấn luyện:
   $$\text{Lặp cho đến khi hội tụ: } \quad \theta_j := \theta_j + \alpha \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)} \quad (\text{với mọi } j)$$
   * **Ưu điểm:** Hàm $J(\theta)$ trong Linear Regression là hàm lồi toàn cục (convex quadratic function), do đó Batch Gradient Descent luôn đảm bảo hội tụ về cực tiểu toàn cục (global minimum).
   * **Nhược điểm:** Rất chậm nếu tập dữ liệu $m$ cực lớn (ví dụ hàng triệu bản ghi) vì mỗi bước đi phải tính tổng trên toàn bộ dữ liệu.

2. **Stochastic Gradient Descent (SGD):**
   Cập nhật tham số ngay lập tức sau khi xét **từng** mẫu huấn luyện một:
   $$\text{Lặp: } \quad \text{Với } i = 1 \text{ đến } m: \quad \theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$
   * **Ưu điểm:** Tốc độ hội tụ tới vùng lân cận cực tiểu cực kỳ nhanh trên dữ liệu lớn.
   * **Nhược điểm:** Không hội tụ tĩnh tại điểm cực tiểu mà dao động liên tục xung quanh vùng cực tiểu (thường giải quyết bằng cách giảm dần $\alpha$ theo thời gian).

---

## 4. Phương trình chuẩn (The Normal Equations)

Thay vì dùng thuật toán lặp Gradient Descent, ta có thể giải tìm $\theta$ tối ưu một cách giải tích (closed-form solution) bằng đạo hàm ma trận.

### 4.1. Định nghĩa đạo hàm ma trận
Cho hàm $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$ ánh xạ từ một ma trận $A$ sang một số thực. Đạo hàm của $f$ theo $A$ là ma trận cùng kích thước:
$$\nabla_A f(A) = \begin{bmatrix} \frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial A_{m1}} & \cdots & \frac{\partial f}{\partial A_{mn}} \end{bmatrix}$$

**Tính chất vết của ma trận (Trace operator):** Với ma trận vuông $A \in \mathbb{R}^{n \times n}$, $\text{tr}(A) = \sum_{i=1}^n A_{ii}$.
* $\text{tr}(AB) = \text{tr}(BA)$
* $\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$
* $\nabla_A \text{tr}(AB) = B^T$
* $\nabla_{A^T} f(A) = (\nabla_A f(A))^T$
* $\nabla_A \text{tr}(ABA^T C) = CAB + C^T A B^T$

### 4.2. Viết lại $J(\theta)$ dưới dạng ma trận
Định nghĩa **Ma trận thiết kế (Design Matrix)** $X \in \mathbb{R}^{m \times (n+1)}$ chứa các mẫu huấn luyện theo từng hàng, và vector mục tiêu $\vec{y} \in \mathbb{R}^m$:

$$X = \begin{bmatrix} — (x^{(1)})^T — \\ — (x^{(2)})^T — \\ \vdots \\ — (x^{(m)})^T — \end{bmatrix}, \qquad \vec{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

Khi đó, vector sai số là $X\theta - \vec{y}$, và hàm mất mát được viết gọn thành:
$$J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y})$$

### 4.3. Tìm cực tiểu của $J(\theta)$
Lấy gradient của $J(\theta)$ theo $\theta$:
$$\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) \\
&= \frac{1}{2} \nabla_\theta \left( \theta^T X^T X \theta - \theta^T X^T \vec{y} - \vec{y}^T X \theta + \vec{y}^T \vec{y} \right) \\
&= \frac{1}{2} \left( 2 X^T X \theta - 2 X^T \vec{y} \right) \\
&= X^T X \theta - X^T \vec{y}
\end{aligned}$$

Cho $\nabla_\theta J(\theta) = 0$, ta thu được **Phương trình chuẩn (Normal Equations)**:
$$X^T X \theta = X^T \vec{y}$$

Nếu $X^T X$ khả nghịch, nghiệm tối ưu $\theta^*$ là:
$$\theta^* = (X^T X)^{-1} X^T \vec{y}$$

*(Lưu ý: Nếu $X^T X$ suy biến/không khả nghịch do dư thừa đặc trưng tuyến tính hoặc $m \le n$, ta có thể dùng giả nghịch đảo Moore-Penrose hoặc thêm điều chuẩn Regularization $\lambda I$).*

---

## 5. So sánh Gradient Descent và Normal Equations

| Tiêu chí | Gradient Descent | Normal Equations |
| :--- | :--- | :--- |
| **Tốc độ học $\alpha$** | Cần lựa chọn và tinh chỉnh $\alpha$. | Không cần chọn $\alpha$. |
| **Số vòng lặp** | Cần chạy nhiều vòng lặp. | Không cần lặp (tính trực tiếp). |
| **Số lượng đặc trưng ($n$)** | Hoạt động tốt ngay cả khi $n$ rất lớn ($n > 10^4$). | Rất chậm khi $n$ lớn vì tính nghịch đảo ma trận mất độ phức tạp $\mathcal{O}(n^3)$. |
| **Áp dụng cho mô hình khác** | Áp dụng được cho hầu hết mọi mô hình (Logistic Regression, Neural Networks...). | Chỉ áp dụng được cho Linear Regression (và một số ít biến thể tuyến tính). |