# Chương 4: Đại số tuyến tính cho học máy (Linear Algebra for Machine Learning)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Hình học hóa dữ liệu trong không gian vector:** Hiểu bản chất dữ liệu như các điểm/vector trong không gian $\mathbb{R}^n$, nắm vững các khái niệm tổ hợp tuyến tính, bao tuyến tính (span), và độc lập tuyến tính.
2. **Làm chủ ma trận và phép biến đổi tuyến tính:** Thấu hiểu ma trận vừa là bảng dữ liệu (Design Matrix) vừa là hàm ánh xạ tuyến tính $f(\mathbf{x}) = \mathbf{A}\mathbf{x}$ làm biến dạng không gian.
3. **Hiểu bản chất hình học của phép nhân ma trận:** Nắm vững tích vô hướng (Dot product), tích ngoài (Outer product), ý nghĩa hình học của phép chiếu và các tính chất đại số cốt lõi.
4. **Phân loại và khai thác các ma trận đặc biệt:** Nắm vững tính chất của ma trận đối xứng, đường chéo, trực giao, định thức, vết (trace), hạng (rank) và mở rộng sang Tensor nhiều chiều.
5. **Định lượng tương đồng và khoảng cách:** Phân biệt chuẩn $L_1, L_2, L_p, L_\infty$ và độ tương đồng Cosine trong không gian vector nhúng (Embeddings).
6. **Làm chủ Trị riêng, Vector riêng và SVD:** Hiểu sâu sắc phân rã phổ (Eigendecomposition), ma trận nửa xác định dương (Positive Semi-Definite), và Phân tích giá trị suy biến (SVD) – nền tảng của PCA, nén dữ liệu và giảm chiều biểu diễn.
7. **Đánh giá tính ổn định số học trong AI:** Hiểu chuẩn ma trận Frobenius, số điều kiện (Condition number) và hiện tượng bùng nổ/triệt tiêu gradient trong mạng nơ-ron sâu.

---

## 4.1 Toán học không gian vector (Vector Space Mathematics)

### 4.1.1 Hình học hóa dữ liệu (Geometric View of Data)
Trong học máy hiện đại, mọi dạng dữ liệu từ thế giới thực (văn bản, âm thanh, hình ảnh, hồ sơ người dùng) đều được chuyển đổi thành các vector số thực trong không gian nhiều chiều:
- Một bức ảnh đen trắng kích thước $28 \times 28$ pixel là một vector $\mathbf{x} \in \mathbb{R}^{784}$.
- Một vector nhúng văn bản (Text Embedding) trong các mô hình LLM là một điểm trong không gian $\mathbb{R}^{1536}$ hoặc $\mathbb{R}^{4096}$.
- Một khách hàng vay vốn là một vector đặc trưng $\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$.

```
              x2 ▲
                 │           ● Vector x = [x1, x2]^T
                 │          /│
                 │         / │
                 │        /  │
                 │       /   │ x2
                 │      /    │
                 │     /     │
                 │    /θ     │
                 └────┴──────┴────────► x1
                      O      x1
```

### 4.1.2 Không gian $\mathbb{R}^n$ và Tiên đề không gian vector
Một không gian vector $\mathcal{V}$ trên trường số thực $\mathbb{R}$ là một tập hợp các phần tử (vector) được trang bị hai phép toán:
1. **Phép cộng vector:** $+: \mathcal{V} \times \mathcal{V} \to \mathcal{V}$ thỏa mãn tính kết hợp, giao hoán, tồn tại phần tử không $\mathbf{0}$ và phần tử đối.
2. **Phép nhân với vô hướng:** $\cdot: \mathbb{R} \times \mathcal{V} \to \mathcal{V}$ thỏa mãn tính phân phối đối với phép cộng vector và phép cộng vô hướng.

### 4.1.3 Các phép toán cơ bản trên vector
Cho hai vector $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ và số vô hướng $c \in \mathbb{R}$:
- **Phép cộng:** $\mathbf{u} + \mathbf{v} = [u_1 + v_1, u_2 + v_2, \dots, u_n + v_n]^T$
- **Phép nhân vô hướng:** $c\mathbf{u} = [cu_1, cu_2, \dots, cu_n]^T$
- **Phép nhân từng phần tử (Hadamard product):** $\mathbf{u} \odot \mathbf{v} = [u_1 v_1, u_2 v_2, \dots, u_n v_n]^T$

### 4.1.4 Tổ hợp tuyến tính và Bao tuyến tính (Linear Combination & Span)
- Cho tập hợp $k$ vector $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\} \subset \mathbb{R}^n$. Một **tổ hợp tuyến tính** là một biểu thức có dạng:
  $$\mathbf{x} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \sum_{i=1}^k c_i \mathbf{v}_i, \quad c_i \in \mathbb{R}$$
- **Bao tuyến tính (Span):** Tập hợp tất cả các tổ hợp tuyến tính có thể có từ tập vector:
  $$\text{Span}(\{\mathbf{v}_1, \dots, \mathbf{v}_k\}) = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\}$$

### 4.1.5 Độc lập tuyến tính và Cơ sở (Linear Independence & Basis)
- Tập vector $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ được gọi là **độc lập tuyến tính (Linearly Independent)** nếu phương trình:
  $$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0}$$
  chỉ có nghiệm duy nhất là nghiệm tầm thường: $c_1 = c_2 = \dots = c_k = 0$.
- Nếu tồn tại ít nhất một $c_i \neq 0$, tập vector được gọi là **phụ thuộc tuyến tính (Linearly Dependent)**. Khi đó, ít nhất một vector có thể biểu diễn qua các vector còn lại (thông tin bị dư thừa).
- **Cơ sở (Basis):** Một tập vector $\mathcal{B} = \{\mathbf{b}_1, \dots, \mathbf{b}_n\}$ là một cơ sở của không gian $\mathbb{R}^n$ nếu $\mathcal{B}$ độc lập tuyến tính và $\text{Span}(\mathcal{B}) = \mathbb{R}^n$. Mọi vector $\mathbf{x} \in \mathbb{R}^n$ đều được biểu diễn duy nhất dưới dạng tọa độ theo cơ sở $\mathcal{B}$.

---

## 4.2 Ma trận (Matrices)

### 4.2.1 Ma trận dữ liệu (Design Matrix)
Trong học máy có giám sát, tập dữ liệu gồm $N$ mẫu quan sát, mỗi mẫu có $D$ đặc trưng được biểu diễn bằng một ma trận thiết kế $\mathbf{X} \in \mathbb{R}^{N \times D}$:
$$\mathbf{X} = \begin{bmatrix} x_{11} & x_{12} & \dots & x_{1D} \\ x_{21} & x_{22} & \dots & x_{2D} \\ \vdots & \vdots & \ddots & \vdots \\ x_{N1} & x_{N2} & \dots & x_{ND} \end{bmatrix} = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_N^T \end{bmatrix}$$
- Mỗi hàng $\mathbf{x}_i^T \in \mathbb{R}^{1 \times D}$ đại diện cho một mẫu dữ liệu (Data instance).
- Mỗi cột $\mathbf{f}_j \in \mathbb{R}^{N \times 1}$ đại diện cho một đặc trưng (Feature).

### 4.2.2 Ma trận như một Phép biến đổi tuyến tính (Linear Transformation)
Một ánh xạ $T: \mathbb{R}^n \to \mathbb{R}^m$ là tuyến tính nếu thỏa mãn:
$$T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v}), \quad \forall \mathbf{u}, \mathbf{v} \in \mathbb{R}^n, \; c, d \in \mathbb{R}$$
Mọi phép biến đổi tuyến tính $T$ đều tương ứng với một ma trận duy nhất $\mathbf{A} \in \mathbb{R}^{m \times n}$ sao cho:
$$T(\mathbf{x}) = \mathbf{A}\mathbf{x}$$
- **Ý nghĩa hình học:** Ma trận $\mathbf{A}$ biến đổi không gian vector ban đầu bằng cách quay (rotation), phản xạ (reflection), giãn nén (scaling), hoặc trượt biến dạng (shearing). Cột thứ $j$ của $\mathbf{A}$ chính là tọa độ ảnh của vector cơ sở chuẩn $\mathbf{e}_j$ sau phép biến đổi.

---

## 4.3 Phép nhân ma trận (Matrix Multiplication)

### 4.3.1 Tích vô hướng (Dot Product / Inner Product)
Cho $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, tích vô hướng là một số thực:
$$\mathbf{u}^T \mathbf{v} = \langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$
- Đo lường mức độ cùng hướng giữa hai vector.
- Nếu $\mathbf{u}^T \mathbf{v} = 0$, hai vector **trực giao (orthogonal)** với nhau ($\theta = 90^\circ$).
- Độ dài hình chiếu vuông góc của $\mathbf{u}$ lên phương của $\mathbf{v}$ là: $\text{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{v}\|_2}$.

### 4.3.2 Nhân ma trận với ma trận
Cho $\mathbf{A} \in \mathbb{R}^{m \times k}$ và $\mathbf{B} \in \mathbb{R}^{k \times n}$, ma trận tích $\mathbf{C} = \mathbf{AB} \in \mathbb{R}^{m \times n}$ có phần tử tại vị trí $(i, j)$ là:
$$C_{ij} = \sum_{r=1}^k A_{ir} B_{rj} = \mathbf{a}_{i,:}^T \mathbf{b}_{:,j}$$

#### Ba góc nhìn cốt lõi về phép nhân ma trận:
1. **Góc nhìn Tích vô hướng (Hàng $\times$ Cột):** $C_{ij}$ là tích vô hướng của hàng $i$ của $\mathbf{A}$ với cột $j$ của $\mathbf{B}$.
2. **Góc nhìn Tổ hợp cột (Column Combination):** Mỗi cột của $\mathbf{C}$ là một tổ hợp tuyến tính của các cột của $\mathbf{A}$, với trọng số là các phần tử trong cột tương ứng của $\mathbf{B}$:
   $$\mathbf{c}_{:,j} = \mathbf{A} \mathbf{b}_{:,j} = \sum_{r=1}^k B_{rj} \mathbf{a}_{:,r}$$
3. **Góc nhìn Tổng tích ngoài (Outer Product Expansion):** Ma trận $\mathbf{C}$ là tổng của $k$ ma trận hạng 1:
   $$\mathbf{AB} = \sum_{r=1}^k \mathbf{a}_{:,r} \mathbf{b}_{r,:}^T$$

### 4.3.3 Tính chất đại số cốt lõi
- **Tính kết hợp:** $\mathbf{A}(\mathbf{BC}) = (\mathbf{AB})\mathbf{C}$
- **Tính phân phối:** $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}$
- **KHÔNG giao hoán:** Nhìn chung, $\mathbf{AB} \neq \mathbf{BA}$ (thứ tự biến đổi không gian là quan trọng).
- **Chuyển vị của tích:** $(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T$

---

## 4.4 Các loại ma trận đặc biệt và Thuộc tính cấu trúc

### 4.4.1 Ma trận chuyển vị, Đối xứng và Vết
- **Chuyển vị (Transpose):** $\mathbf{A}^T$ đổi hàng thành cột: $(\mathbf{A}^T)_{ij} = A_{ji}$.
- **Ma trận đối xứng (Symmetric Matrix):** $\mathbf{A} \in \mathbb{R}^{n \times n}$ đối xứng nếu $\mathbf{A} = \mathbf{A}^T$. Ma trận hiệp phương sai $\boldsymbol{\Sigma} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ và ma trận Hessian $\mathbf{H}$ luôn là ma trận đối xứng.
- **Vết (Trace):** Tổng các phần tử trên đường chéo chính của ma trận vuông:
  $$\text{Tr}(\mathbf{A}) = \sum_{i=1}^n A_{ii}$$
  Tính chất hoán vị vòng quanh (Cyclic Property): $\text{Tr}(\mathbf{ABC}) = \text{Tr}(\mathbf{BCA}) = \text{Tr}(\mathbf{CAB})$.

### 4.4.2 Ma trận đơn vị và Ma trận đường chéo
- **Ma trận đơn vị ($\mathbf{I}_n$):** $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$.
- **Ma trận đường chéo ($\mathbf{D} = \text{diag}(d_1, \dots, d_n)$):** Các phần tử ngoài đường chéo bằng 0. Phép nhân với ma trận đường chéo tương đương với việc co giãn độc lập từng trục tọa độ.

### 4.4.3 Định thức và Nghịch đảo ma trận
- **Định thức ($\det(\mathbf{A})$ hoặc $|\mathbf{A}|$):** Đo lường hệ số thay đổi thể tích (Volume scaling factor) của không gian dưới phép biến đổi $\mathbf{A}$. Nếu $\det(\mathbf{A}) = 0$, phép biến đổi làm sụp đổ không gian thành một chiều thấp hơn (mất thông tin).
- **Nghịch đảo ($\mathbf{A}^{-1}$):** Ma trận vuông $\mathbf{A}$ khả nghịch khi và chỉ khi $\det(\mathbf{A}) \neq 0$:
  $$\mathbf{A} \mathbf{A}^{-1} = \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}$$
  Tính chất: $(\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$.

### 4.4.4 Hạng ma trận (Matrix Rank)
- **Hạng ($\text{rank}(\mathbf{A})$):** Số lượng tối đa các cột (hoặc hàng) độc lập tuyến tính của $\mathbf{A}$.
- Ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ có $\text{rank}(\mathbf{A}) \le \min(m, n)$. Nếu $\text{rank}(\mathbf{A}) = \min(m, n)$, ma trận có **đủ hạng (Full rank)**.
- **Hạng thấp (Low-rank):** Cơ sở của các thuật toán nén mô hình (LoRA - Low-Rank Adaptation trong tinh chỉnh LLM) và hệ thống gợi ý ma trận (Matrix Factorization).

### 4.4.5 Ma trận trực giao (Orthogonal Matrix)
Ma trận vuông $\mathbf{Q} \in \mathbb{R}^{n \times n}$ là trực giao nếu các cột của nó đôi một trực chuẩn:
$$\mathbf{Q}^T \mathbf{Q} = \mathbf{Q} \mathbf{Q}^T = \mathbf{I} \implies \mathbf{Q}^{-1} = \mathbf{Q}^T$$
- Phép biến đổi bởi ma trận trực giao bảo toàn độ dài vector và góc giữa các vector: $\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$. Đây là phép quay hoặc phản xạ thuần túy, không làm méo mó không gian.

### 4.4.6 Tensor trong Học sâu
- **Tensor** là sự tổng quát hóa của vô hướng (bậc 0), vector (bậc 1), ma trận (bậc 2) lên không gian $N$ chiều ($N$-dimensional array).
  - Tensor bậc 3: Ảnh màu $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$.
  - Tensor bậc 4: Batch ảnh trong CNN $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$.
  - Tensor bậc 3 trong NLP: Batch văn bản $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$ (Batch size $\times$ Sequence length $\times$ Embedding dimension).

---

## 4.5 Định lượng tương đồng và Khoảng cách (Norms and Similarity)

### 4.5.1 Khái niệm Chuẩn vector (Vector Norms)
Chuẩn $\|\cdot\|$ là hàm ánh xạ từ $\mathbb{R}^n \to [0, \infty)$ thỏa mãn 3 tiên đề:
1. Không âm và xác định: $\|\mathbf{x}\| \ge 0$, và $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$.
2. Thuần nhất tuyệt đối: $\|c\mathbf{x}\| = |c| \|\mathbf{x}\|$.
3. Bất đẳng thức tam giác: $\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$.

### 4.5.2 Các chuẩn $L_p$ kinh điển trong Học máy
- **Chuẩn $L_1$ (Manhattan Norm):**
  $$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$$
  Ứng dụng: Tạo tính chất thưa (Sparsity) trong chính quy hóa Lasso ($L_1$ Regularization), giúp triệt tiêu các trọng số dư thừa về 0.
- **Chuẩn $L_2$ (Euclidean Norm):**
  $$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}$$
  Ứng dụng: Đo khoảng cách hình học thông thường, dùng trong Ridge Regularization (Weight Decay) để ép trọng số nhỏ đều.
- **Chuẩn $L_\infty$ (Max Norm):**
  $$\|\mathbf{x}\|_\infty = \max_{1 \le i \le n} |x_i|$$
  Ứng dụng: Trong kiểm thử đối kháng (Adversarial Robustness), giới hạn mức độ thay đổi tối đa trên từng pixel ảnh bị tấn công.

### 4.5.3 Độ tương đồng Cosine (Cosine Similarity)
Đo lường góc giữa hai vector bất chấp độ lớn (magnitude) của chúng:
$$\text{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \cos(\theta) \in [-1, 1]$$
- **Ứng dụng:** Đo lường độ tương đồng ngữ nghĩa giữa các vector nhúng (Text Embeddings) trong tìm kiếm ngữ nghĩa (Semantic Search), Vector Database, và cơ chế Attention trong kiến trúc Transformer.

---

## 4.6 Trị riêng, Vector riêng và SVD

### 4.6.1 Định nghĩa Trị riêng và Vector riêng (Eigenvalues & Eigenvectors)
Cho ma trận vuông $\mathbf{A} \in \mathbb{R}^{n \times n}$. Nếu tồn tại vector khác không $\mathbf{v} \neq \mathbf{0}$ và số vô hướng $\lambda \in \mathbb{C}$ thỏa mãn:
$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$
thì $\mathbf{v}$ được gọi là **vector riêng (eigenvector)** và $\lambda$ là **trị riêng (eigenvalue)** tương ứng.
- **Ý nghĩa hình học:** Thông thường, một ma trận biến đổi sẽ làm vector vừa đổi hướng vừa đổi độ dài. Tuy nhiên, các vector riêng là những hướng đặc biệt trong không gian mà phép biến đổi $\mathbf{A}$ **không làm thay đổi phương hướng**, mà chỉ kéo giãn hoặc thu hẹp theo hệ số tỉ lệ $\lambda$.

```
          y ▲                     Av = λv
            │                    /
            │                   /
            │                  /  v
            │                 / 
            │                /
            └───────────────┴────────► x
```

### 4.6.2 Phân rã phổ (Eigendecomposition)
Nếu ma trận $\mathbf{A} \in \mathbb{R}^{n \times n}$ có $n$ vector riêng độc lập tuyến tính $\{\mathbf{v}_1, \dots, \mathbf{v}_n\}$, ta có thể phân tích:
$$\mathbf{A} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^{-1}$$
Trong đó $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_n]$ và $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_n)$.
- **Định lý phổ cho ma trận đối xứng thực (Spectral Theorem):** Mọi ma trận đối xứng thực $\mathbf{A} = \mathbf{A}^T$ đều có các trị riêng là số thực và luôn tồn tại một hệ cơ sở trực chuẩn gồm các vector riêng:
  $$\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

### 4.6.3 Tính xác định dương (Positive Definiteness)
Một ma trận đối xứng $\mathbf{A} \in \mathbb{R}^{n \times n}$ được gọi là:
- **Xác định dương (Positive Definite - PD):** $\mathbf{x}^T \mathbf{A} \mathbf{x} > 0, \quad \forall \mathbf{x} \neq \mathbf{0} \iff \text{Mọi } \lambda_i > 0$.
- **Nửa xác định dương (Positive Semi-Definite - PSD):** $\mathbf{x}^T \mathbf{A} \mathbf{x} \ge 0, \quad \forall \mathbf{x} \iff \text{Mọi } \lambda_i \ge 0$.
- **Ý nghĩa trong học máy:** Ma trận Hessian $\mathbf{H}$ của hàm mất mát tại điểm cực tiểu địa phương phải là ma trận bán xác định dương ($\mathbf{H} \succeq 0$). Ma trận hiệp phương sai luôn là ma trận bán xác định dương.

### 4.6.4 Phân tích giá trị suy biến (Singular Value Decomposition - SVD)
Phân tích trị riêng chỉ áp dụng cho ma trận vuông. SVD mở rộng phân tích này cho **mọi ma trận chữ nhật bất kỳ** $\mathbf{A} \in \mathbb{R}^{m \times n}$:
$$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
Trong đó:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$ là ma trận trực giao chứa các vector suy biến trái (Left singular vectors), chính là các vector riêng của $\mathbf{AA}^T$.
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ là ma trận trực giao chứa các vector suy biến phải (Right singular vectors), chính là các vector riêng của $\mathbf{A}^T\mathbf{A}$.
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ là ma trận đường chéo chứa các **giá trị suy biến (singular values)** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ sắp xếp giảm dần, với $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^T\mathbf{A})}$.

#### Định lý Xấp xỉ Hạng thấp Eckart-Young-Mirsky:
Ma trận xấp xỉ hạng $k < r$ tối ưu nhất của $\mathbf{A}$ theo chuẩn Frobenius là giữ lại $k$ giá trị suy biến lớn nhất:
$$\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$
Đây là nền tảng toán học của **Phân tích thành phần chính (PCA)**, nén ảnh, khử nhiễu tín hiệu và phân rã ma trận trong hệ gợi ý (Collaborative Filtering).

---

## 4.7 Chuẩn ma trận và Sự ổn định của các mô hình AI

### 4.7.1 Chuẩn ma trận Frobenius và Chuẩn phổ
- **Chuẩn Frobenius:** Đo lường tổng độ lớn của tất cả các phần tử:
  $$\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2} = \sqrt{\text{Tr}(\mathbf{A}^T \mathbf{A})} = \sqrt{\sum_{i=1}^r \sigma_i^2}$$
- **Chuẩn phổ (Spectral Norm / $L_2$ Matrix Norm):** Hệ số khuếch đại tối đa của ma trận:
  $$\|\mathbf{A}\|_2 = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|\mathbf{Ax}\|_2}{\|\mathbf{x}\|_2} = \sigma_{\max}(\mathbf{A})$$

### 4.7.2 Số điều kiện (Condition Number) và Độ ổn định số học
Số điều kiện của ma trận khả nghịch $\mathbf{A}$:
$$\kappa(\mathbf{A}) = \|\mathbf{A}\|_2 \|\mathbf{A}^{-1}\|_2 = \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}$$
- Nếu $\kappa(\mathbf{A}) \approx 1$: Ma trận được điều kiện hóa tốt (Well-conditioned), thuật toán nghịch đảo và Gradient Descent hội tụ nhanh, ổn định.
- Nếu $\kappa(\mathbf{A}) \gg 1$: Ma trận điều kiện xấu (Ill-conditioned), một nhiễu rất nhỏ ở đầu vào $\Delta \mathbf{x}$ sẽ bị khuếch đại thành sai số khổng lồ ở đầu ra:
  $$\frac{\|\Delta \mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A}) \frac{\|\Delta \mathbf{b}\|}{\|\mathbf{b}\|}$$
- **Kết nối với Deep Learning:** Trong mạng nơ-ron sâu $L$ tầng $\mathbf{h}_L = \mathbf{W}_L \mathbf{W}_{L-1} \dots \mathbf{W}_1 \mathbf{x}$, nếu các ma trận trọng số có $\sigma_{\max} > 1$, gradient sẽ bùng nổ (Exploding gradient); nếu $\sigma_{\max} < 1$, gradient sẽ triệt tiêu về 0 (Vanishing gradient). Kỹ thuật **Spectral Normalization** chuẩn hóa $\|\mathbf{W}\|_2 = 1$ để ổn định quá trình huấn luyện GANs và Transformer.

---

## 4.8 Tài liệu tham khảo (References)

1. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Cambridge University Press. [Tham khảo: `resources/Mathematics for Machine learning.pdf`] (Chương 2: Linear Algebra; Chương 3: Analytic Geometry; Chương 4: Matrix Decompositions).
2. **Vũ Hữu Tiệp (2018).** *Machine Learning Cơ Bản*. NXB Khoa học và Kỹ thuật. [Tham khảo: `resources/Machine Learning Co Ban.pdf`] (Chương 2: Đại số tuyến tính; Chương 14: Singular Value Decomposition).
3. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo: `resources/Deep Learning.pdf`] (Chương 2: Linear Algebra).
4. **Strang, G. (2016).** *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. (Giáo trình kinh điển về hình học và phân rã ma trận).
5. **Eckart, C., & Young, G. (1936).** The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3), 211-218. (Định lý nền tảng của SVD xấp xỉ hạng thấp).
6. **Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018).** Spectral normalization for generative adversarial networks. *ICLR 2018*.

---

## 4.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta cài đặt và trực quan hóa các ứng dụng đại số tuyến tính cốt lõi trong Học máy chỉ bằng `numpy` và `matplotlib`:

1. **Thành phần 1 - Phép biến đổi tuyến tính trong không gian 2D:**
   - Trực quan hóa hình học cách các ma trận biến đổi (Scale, Rotate, Shear, Reflect) làm biến dạng lưới tọa độ phẳng $\mathbb{R}^2$.
2. **Thành phần 2 - Đo lường tương đồng và Tìm kiếm vector nhúng (Cosine Similarity):**
   - Xây dựng hệ thống tìm kiếm tài liệu tương đồng dựa trên vector đặc trưng bằng chuẩn $L_1, L_2$ và Cosine Similarity.
3. **Thành phần 3 - Trị riêng và Vector riêng từ đầu:**
   - Cài đặt thuật toán Power Iteration để tìm trị riêng trội (Dominant Eigenvalue/Eigenvector).
   - Kiểm chứng tính chất $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ và trực quan hóa các hướng không đổi của vector riêng.
4. **Thành phần 4 - Nén dữ liệu và giảm chiều biểu diễn bằng SVD (Low-Rank Matrix Approximation):**
   - Tự cài đặt phân tích SVD và tái tạo ma trận xấp xỉ hạng $k$: $\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.
   - Thực nghiệm nén một bức ảnh 2D ở các mức hạng $k \in \{5, 20, 50\}$, phân tích tỷ lệ dung lượng tiết kiệm và tỷ lệ năng lượng phổ được bảo toàn $\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$.
5. **Thành phần 5 - Phân tích tính ổn định số học (Condition Number):**
   - Mô phỏng hệ phương trình điều kiện tốt vs. điều kiện xấu; minh họa tác động của việc chuẩn hóa dữ liệu lên tốc độ hội tụ.
