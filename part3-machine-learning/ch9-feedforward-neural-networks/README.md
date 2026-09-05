# Chương 9: Mạng nơ-ron đa tầng truyền thẳng (Feedforward Neural Networks - MLP)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Hiểu rõ tổng quan mạng MLP:** Nắm vững cấu trúc đồ thị có hướng không chu trình (DAG) của mạng nơ-ron truyền thẳng (Feedforward Neural Network) và vai trò lịch sử từ Perceptron đến Deep Learning.
2. **Giải thích triệt để giới hạn của các mô hình tuyến tính:** Chứng minh bằng hình học và đại số tại sao mô hình tuyến tính thất bại trước bài toán **XOR**; chứng minh việc xếp chồng nhiều tầng tuyến tính không tạo ra thêm năng lực biểu diễn mới nếu thiếu hàm kích hoạt phi tuyến.
3. **Làm chủ cấu trúc tầng của mạng MLP:** Phân biệt tầng đầu vào, các tầng ẩn (Hidden layers) và tầng đầu ra; hiểu sâu sắc **Định lý Xấp xỉ Vạn năng (Universal Approximation Theorem)**.
4. **Phân tích toàn diện các Hàm kích hoạt (Activation Functions):** Nắm vững ưu nhược điểm của Sigmoid, Tanh, ReLU, Leaky ReLU và GELU; giải thích bản chất hiện tượng triệt tiêu gradient (Vanishing Gradient) và nơ-ron chết (Dying ReLU).
5. **Khai triển toán học trọn vẹn Cơ chế Lan truyền (Propagation Mechanism):** Tự chứng minh và dẫn xuất các công thức của **Lan truyền tiến (Forward Pass)** và **Lan truyền ngược (Backpropagation)** dựa trên quy tắc chuỗi ma trận.

---

## 9.1 Tổng quan Mạng nơ-ron đa tầng truyền thẳng (MLP Overview)

**Mạng nơ-ron đa tầng truyền thẳng (Multi-Layer Perceptron - MLP)**, hay còn gọi là **Mạng nơ-ron truyền thẳng sâu (Deep Feedforward Networks)**, là kiến trúc nền tảng tinh túy nhất của Học sâu (Deep Learning).
- **Đặc trưng cốt lõi:** Thông tin truyền thẳng một chiều từ đầu vào $\mathbf{x}$, qua các tầng tính toán trung gian (tầng ẩn), đến đầu ra $\mathbf{y}$. Không có các vòng lặp phản hồi ngược (Feedback connections) như trong mạng nơ-ron hồi quy (RNN).
- **Mục tiêu:** Xấp xỉ một hàm số phức tạp $f^*(\mathbf{x})$. Một mạng MLP định nghĩa một ánh xạ $\mathbf{y} = f(\mathbf{x}; \boldsymbol{\theta})$ và học bộ tham số $\boldsymbol{\theta} = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}$ sao cho mô hình xấp xỉ tốt nhất hàm mục tiêu thực tế.

```
       Tầng Đầu vào              Tầng Ẩn                Tầng Đầu ra
      (Input Layer)           (Hidden Layer)          (Output Layer)

          x1 ● ───────────────► ● h1
                ╲           ╱        ╲
                  ╲       ╱            ╲
                    ╲   ╱                ▼
          x2 ● ───────╳───────► ● h2 ────► ● ŷ (Dự đoán)
                    ╱   ╲                ▲
                  ╱       ╲            ╱
                ╱           ╲        ╱
          x3 ● ───────────────► ● h3
```

---

## 9.2 Hạn chế của các Mô hình Tuyến tính và Vấn đề hàm XOR

### 9.2.1 Bài toán hàm XOR (The XOR Problem)
Năm 1969, Marvin Minsky và Seymour Papert xuất bản cuốn sách *Perceptrons*, chứng minh rằng mô hình Perceptron đơn tầng tuyến tính không thể phân loại được dữ liệu không phân tách tuyến tính (Non-linearly separable).

| $x_1$ | $x_2$ | $x_1 \oplus x_2$ (XOR) |
| :---: | :---: | :---: |
| 0 | 0 | **0** |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

```
              x2 ▲
                 │
               1 ■ (0, 1)      ● (1, 1)      ■ : Nhãn 1
                 │                           ● : Nhãn 0
                 │   Không thể phân tách
                 │   bằng 1 đường thẳng duy nhất!
               0 ● (0, 0)      ■ (1, 0)
                 └─────────────┴────────► x1
                 0             1
```

Không tồn tại bất kỳ bộ trọng số tuyến tính $(w_1, w_2, b)$ nào sao cho đường thẳng $w_1 x_1 + w_2 x_2 + b = 0$ có thể phân tách thành công hai nhóm điểm $(0, 1), (1, 0)$ khỏi $(0, 0), (1, 1)$.

### 9.2.2 Vì sao bắt buộc phải có Hàm kích hoạt phi tuyến?
Xét một mạng nơ-ron sâu $L$ tầng nhưng **chỉ sử dụng các phép biến đổi tuyến tính** (không có hàm kích hoạt phi tuyến):
$$\mathbf{h}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{h}^{(2)} = \mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)} = \mathbf{W}^{(2)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} = (\mathbf{W}^{(2)}\mathbf{W}^{(1)})\mathbf{x} + (\mathbf{W}^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)})$$
$$\mathbf{h}^{(L)} = \mathbf{W}' \mathbf{x} + \mathbf{b}'$$
> **Chứng minh toán học:** Tích của các ma trận tuyến tính vẫn là một ma trận tuyến tính duy nhất. Do đó, một mạng nơ-ron dù sâu 1000 tầng nhưng không có hàm kích hoạt phi tuyến tính sẽ **chỉ tương đương với một mô hình hồi quy tuyến tính một tầng duy nhất**! Hàm kích hoạt phi tuyến chính là "linh hồn" cho phép mạng nơ-ron bẻ cong không gian để giải quyết bài toán XOR.

---

## 9.3 Cấu trúc mạng MLP (Architecture of MLP)

Một mạng MLP hoàn chỉnh gồm 3 thành phần chính:
1. **Tầng đầu vào (Input Layer):** Nhận vector đặc trưng $\mathbf{x} = [x_1, \dots, x_D]^T \in \mathbb{R}^D$.
2. **Các tầng ẩn (Hidden Layers):** Gồm $L-1$ tầng tính toán trung gian. Tại mỗi tầng $l \in \{1, \dots, L-1\}$:
   - **Tổ hợp tuyến tính (Affine Transformation):**
     $$\mathbf{a}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$$
   - **Kích hoạt phi tuyến (Non-linear Activation):**
     $$\mathbf{h}^{(l)} = g^{(l)}(\mathbf{a}^{(l)})$$
     (Quy ước: $\mathbf{h}^{(0)} = \mathbf{x}$).
3. **Tầng đầu ra (Output Layer):** Tầng thứ $L$, đưa ra dự đoán cuối cùng:
   $$\hat{\mathbf{y}} = g^{(L)}(\mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)})$$

### 9.3.1 Định lý Xấp xỉ Vạn năng (Universal Approximation Theorem)
Được chứng minh bởi George Cybenko (1989) cho hàm kích hoạt Sigmoidal và Kurt Hornik (1991) cho các hàm kích hoạt phi tuyến tổng quát:
> *"Một mạng nơ-ron truyền thẳng chỉ cần **duy nhất một tầng ẩn** với số lượng nơ-ron hữu hạn đủ lớn và hàm kích hoạt phi tuyến liên tục có thể xấp xỉ bất kỳ hàm số liên tục nào trên một tập compact trong $\mathbb{R}^D$ với độ chính xác tùy ý $\epsilon > 0$."*

- **Tại sao ta vẫn cần mạng nơ-ron SÂU (Deep) thay vì chỉ một tầng RỘNG (Wide)?**  
  Mặc dù một tầng ẩn về mặt lý thuyết có thể xấp xỉ mọi hàm, nhưng số lượng nơ-ron cần thiết có thể bùng nổ theo hàm mũ $\mathcal{O}(2^D)$. Kiến trúc sâu (nhiều tầng) cho phép mô hình học các biểu diễn phân cấp (Hierarchical representations: từ cạnh viền $\to$ bộ phận $\to$ vật thể hoàn chỉnh), tiết kiệm số lượng tham số theo hàm mũ và tổng quát hóa tốt hơn nhiều.

---

## 9.4 Các Hàm Kích hoạt Phổ biến (Activation Functions)

```
        Sigmoid                  Tanh                    ReLU
     1 ┌──────╭─────       1 ┌──────╭─────        ▲        ╱
       │     ╱               │     ╱              │       ╱
   0.5 ┼────●──────        0 ┼────●──────         │      ╱
       │   ╱                -1 ─────╯──────        └─────●────────►
     0 └───╯────────                              0
```

### 9.4.1 Hàm Sigmoid (Logistic Activation)
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
- **Ưu điểm:** Đầu ra giới hạn trong $(0, 1)$, biểu diễn tự nhiên xác suất nhị phân.
- **Tử huyệt:**
  1. **Triệt tiêu Gradient (Vanishing Gradient):** Đạo hàm $\sigma'(z) \le 0.25$. Khi $|z| > 4$, $\sigma'(z) \approx 0$. Khi truyền ngược qua nhiều tầng, gradient bị nhân liên tục với các số $< 0.25$ và nhanh chóng triệt tiêu về 0, khiến các tầng đầu tiên ngừng học hoàn toàn.
  2. **Không định tâm quanh 0 (Not zero-centered):** Đầu ra luôn dương $\implies$ Gradient của các trọng số tầng tiếp theo luôn cùng dấu (cùng dương hoặc cùng âm), gây ra hiện tượng dao động ziczac khi cập nhật tham số.

### 9.4.2 Hàm Tanh (Hyperbolic Tangent)
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z)$$
- **Ưu điểm:** Đầu ra thuộc khoảng $(-1, 1)$, hoàn toàn định tâm quanh 0 (zero-centered), giúp tối ưu hóa nhanh hơn Sigmoid.
- **Nhược điểm:** Vẫn bị hiện tượng bão hòa gradient (Vanishing Gradient) khi $|z|$ lớn.

### 9.4.3 Hàm ReLU (Rectified Linear Unit - Nair & Hinton, 2010)
$$f(z) = \max(0, z), \qquad f'(z) = \begin{cases} 1 & \text{nếu } z > 0 \\ 0 & \text{nếu } z < 0 \end{cases}$$
- **Đột phá lịch sử đưa Deep Learning bùng nổ:**
  1. **Không bị bão hòa ở miền dương:** Đạo hàm luôn bằng $1$ khi $z > 0$, triệt tiêu hoàn toàn hiện tượng triệt tiêu gradient ở miền dương!
  2. **Tốc độ tính toán siêu nhanh:** Chỉ là phép so sánh ngưỡng `max(0, z)`, nhanh hơn hàm $\exp$ hàng chục lần.
  3. **Tạo tính thưa tự nhiên (Sparsity):** Khoảng $50\%$ nơ-ron có đầu ra bằng 0, mô hình trở nên thưa và tiết kiệm tài nguyên.
- **Nhược điểm - Hiện tượng Nơ-ron Chết (Dying ReLU):** Nếu một nơ-ron bị cập nhật bởi bước nhảy gradient quá lớn khiến $z < 0$ với mọi mẫu dữ liệu, gradient qua nó sẽ luôn bằng 0 mãi mãi $\implies$ Nơ-ron bị "chết vĩnh viễn".

### 9.4.4 Các biến thể hiện đại của ReLU
- **Leaky ReLU:** Cho phép một độ dốc nhỏ khi $z < 0$: $f(z) = \max(\alpha z, z)$ (với $\alpha \approx 0.01$), loại bỏ hoàn toàn vấn đề Dying ReLU.
- **GELU (Gaussian Error Linear Unit - Hendrycks & Gimpel, 2016):**
  $$f(z) = z \cdot \Phi(z) = z \cdot P(X \le z), \quad X \sim \mathcal{N}(0, 1)$$
  GELU là hàm kích hoạt tiêu chuẩn vàng trong các mô hình Transformer hiện đại (BERT, GPT-2/3/4, ViT).

---

## 9.5 Cơ chế Lan truyền (Forward and Backward Propagation)

### 9.5.1 Lan truyền tiến (Forward Propagation)
Cho một batch dữ liệu gồm $N$ mẫu $\mathbf{X} \in \mathbb{R}^{N \times D}$. Tại mỗi tầng $l = 1, \dots, L$:
$$\mathbf{Z}^{(l)} = \mathbf{H}^{(l-1)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)}, \qquad \mathbf{H}^{(l)} = g^{(l)}(\mathbf{Z}^{(l)})$$
(Quy ước: $\mathbf{H}^{(0)} = \mathbf{X}$).  
Tại tầng đầu ra $L$, tính giá trị mất mát $\mathcal{L}(\mathbf{H}^{(L)}, \mathbf{Y})$.

### 9.5.2 Lan truyền ngược (Backpropagation)
Thuật toán Lan truyền ngược áp dụng quy tắc chuỗi (Chain Rule) để tính toán gradient của hàm mất mát đối với toàn bộ các trọng số và độ lệch trong mạng.

```
       Z(l)  ───────────►  H(l) = g(Z(l))  ───────────►  Z(l+1)
        ▲                        ▲                         ▲
        │                        │                         │
    δ(l) =                   ∂L/∂H(l) =                  δ(l+1) =
  ∂L/∂Z(l)                δ(l+1) (W(l+1))^T             ∂L/∂Z(l+1)
        │                        │                         │
        ◄────────────────────────┴─────────────────────────┘
                    Lan truyền ngược tín hiệu lỗi
```

#### Bốn phương trình cốt lõi của Backpropagation:
1. **Lỗi tại tầng đầu ra ($l = L$):**
   $$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{Z}^{(L)}} \mathcal{L} = \nabla_{\mathbf{H}^{(L)}} \mathcal{L} \odot {g^{(L)}}'(\mathbf{Z}^{(L)})$$
   *(Đặc biệt: Với Softmax kết hợp Cross-Entropy hoặc Sigmoid kết hợp Binary Cross-Entropy, công thức rút gọn tuyệt đẹp: $\boldsymbol{\delta}^{(L)} = \hat{\mathbf{Y}} - \mathbf{Y}$).*

2. **Lan truyền lỗi ngược về các tầng ẩn ($l = L-1, \dots, 1$):**
   $$\boldsymbol{\delta}^{(l)} = \left( \boldsymbol{\delta}^{(l+1)} (\mathbf{W}^{(l+1)})^T \right) \odot {g^{(l)}}'(\mathbf{Z}^{(l)})$$

3. **Đạo hàm theo ma trận trọng số $\mathbf{W}^{(l)}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{1}{N} (\mathbf{H}^{(l-1)})^T \boldsymbol{\delta}^{(l)}$$

4. **Đạo hàm theo vector độ lệch $\mathbf{b}^{(l)}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \frac{1}{N} \sum_{i=1}^N \boldsymbol{\delta}_{i,:}^{(l)}$$

5. **Cập nhật trọng số bằng Gradient Descent:**
   $$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \qquad \mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}$$

---

## 9.6 Tài liệu tham khảo (References)

1. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo: `resources/Deep Learning.pdf`] (Chương 6: Deep Feedforward Networks).
2. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. [Tham khảo: `resources/Pattern Recognition and Machine Learning.pdf`] (Chương 5: Neural Networks - Feed-forward Network Functions, Error Backpropagation).
3. **Vũ Hữu Tiệp (2018).** *Machine Learning Cơ Bản*. NXB Khoa học và Kỹ thuật. [Tham khảo: `resources/Machine Learning Co Ban.pdf`] (Chương 15: Multi-layer Perceptron and Backpropagation).
4. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536. (Bài báo lịch sử phổ biến thuật toán Lan truyền ngược).
5. **Cybenko, G. (1989).** Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314. (Định lý Xấp xỉ Vạn năng).
6. **Nair, V., & Hinton, G. E. (2010).** Rectified linear units improve restricted boltzmann machines. *ICML 2010*.

---

## 9.7 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng trọn vẹn một mạng nơ-ron MLP từ đầu bằng `NumPy`:

1. **Thành phần 1 - Giải quyết bài toán XOR kinh điển:**
   - Xây dựng một mạng MLP nhỏ (2 nơ-ron đầu vào, 1 tầng ẩn gồm 4 nơ-ron ReLU, 1 nơ-ron đầu ra Sigmoid).
   - Chứng minh mạng giải quyết thành công hàm XOR với độ chính xác $100\%$, trong khi mô hình tuyến tính đơn tầng thất bại.
2. **Thành phần 2 - Tự cài đặt Lớp Mạng MLP tổng quát (`NeuralNetwork` from scratch):**
   - Hỗ trợ số lượng tầng ẩn tùy ý (`hidden_dims`).
   - Cài đặt đầy đủ các hàm kích hoạt (ReLU, Sigmoid, Tanh) kèm đạo hàm giải tích tương ứng.
   - Cài đặt trọn vẹn thuật toán Lan truyền tiến và Lan truyền ngược.
3. **Thành phần 3 - Thử thách trên Dữ liệu Xoắn ốc Phi tuyến (Spiral Dataset):**
   - Sinh bộ dữ liệu 2 vòng xoắn ốc lồng nhau (một trong những bài toán phi tuyến tính khó nhất của học máy).
   - Huấn luyện mạng MLP 2 tầng ẩn để phân loại thành công 2 vòng xoắn.
4. **Thành phần 4 - Trực quan hóa Ranh giới Quyết định Phi tuyến tính:**
   - Sử dụng `matplotlib contourf` vẽ ranh giới quyết định uốn lượn mềm mại của mạng nơ-ron trên mặt phẳng 2D.
5. **Thành phần 5 - Minh họa hiện tượng Triệt tiêu Gradient (Vanishing Gradient Simulation):**
   - So sánh độ lớn của gradient tại tầng đầu tiên giữa mạng dùng hàm Sigmoid sâu 5 tầng vs. mạng dùng hàm ReLU sâu 5 tầng.
