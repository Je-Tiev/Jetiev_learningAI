# Chương 8: Học máy Cổ điển (Classical Machine Learning)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Làm chủ thuật toán $k$-Nearest Neighbors ($k$-NN):** Hiểu rõ bản chất phi tham số (Non-parametric) và học lười (Lazy learning); phân tích tác động của siêu tham số $k$, khoảng cách có trọng số và lời nguyền số chiều (Curse of Dimensionality).
2. **Xây dựng Cây quyết định (Decision Tree):** Tính toán độ thuần khiết qua **Entropy**, **Information Gain (ID3)** và **Chỉ số Gini (CART)**; hiểu cơ chế phân chia không gian trực giao và kỹ thuật tỉa cành (Pruning).
3. **Phân tích toán học Rừng ngẫu nhiên (Random Forest):** Thấu hiểu cơ chế lấy mẫu Bootstrap và không gian đặc trưng con ngẫu nhiên (Random Feature Subspace); chứng minh khả năng triệt tiêu phương sai mà không làm tăng thiên kiến.
4. **Hiểu sâu sắc Hồi quy Tuyến tính và Hồi quy Logistic:** Nắm vững hàm liên kết Logit/Sigmoid, hàm mất mát Binary Cross-Entropy và đạo hàm gradient.
5. **Mở rộng sang Hồi quy Softmax (Multinomial Logistic Regression):** Làm chủ bài toán phân loại đa lớp, hàm kích hoạt Softmax và hàm mất mát Categorical Cross-Entropy.
6. **Hệ thống hóa các phương pháp Học tổ hợp (Ensemble Learning):** Phân biệt bản chất toán học giữa **Bagging** (giảm phương sai), **Boosting** (AdaBoost, Gradient Boosting - giảm thiên kiến) và **Stacking** (Meta-learning).

---

## 8.1 Thuật toán $k$-Láng giềng gần nhất ($k$-Nearest Neighbors - $k$-NN)

### 8.1.1 Bản chất Thuật toán và Bầu chọn đa số (Majority Voting)
$k$-NN là một thuật toán **học lười (Lazy Learner / Instance-based Learning)**: Mô hình không có pha huấn luyện tạo tham số tường minh mà chỉ đơn giản lưu trữ toàn bộ tập dữ liệu huấn luyện $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$.
- **Quy trình dự đoán cho điểm mới $\mathbf{x}^*$:**
  1. Tính khoảng cách (thường là khoảng cách Euclidean $d(\mathbf{x}^*, \mathbf{x}_i) = \|\mathbf{x}^* - \mathbf{x}_i\|_2$) từ $\mathbf{x}^*$ đến mọi điểm trong tập huấn luyện.
  2. Xác định tập hợp $k$ điểm gần $\mathbf{x}^*$ nhất: $\mathcal{N}_k(\mathbf{x}^*)$.
  3. **Bầu chọn đa số (Majority Voting):**
     $$\hat{y} = \arg\max_{c \in \{1, \dots, C\}} \sum_{i \in \mathcal{N}_k(\mathbf{x}^*)} \mathbb{I}(y_i = c)$$

```
                         y ▲
                           │       ▲ Lớp A
                           │     ▲   ▲
                           │    ┌───┐
                           │  ▲ │ ● │ ■
                           │    └───┘   ■ Lớp B
                           │       ■  ■
                           └─────────────────► x
                                  x*
```

### 8.1.2 Đánh trọng số theo khoảng cách (Distance-Weighted $k$-NN)
Các láng giềng ở gần điểm truy vấn hơn xứng đáng có tiếng nói lớn hơn trong quyết định:
$$w_i = \frac{1}{d(\mathbf{x}^*, \mathbf{x}_i)^2 + \epsilon}, \qquad \hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_k(\mathbf{x}^*)} w_i \mathbb{I}(y_i = c)$$

### 8.1.3 Lựa chọn siêu tham số $k$ và Lời nguyền số chiều
- **$k = 1$:** Mô hình cực kỳ phức tạp, ranh giới phân chia gấp khúc bao quanh từng điểm dữ liệu $\implies$ Quá khớp (Overfitting), nhạy cảm cực độ với nhiễu.
- **$k$ quá lớn ($k \to N$):** Mô hình luôn dự đoán lớp chiếm đa số trong tập dữ liệu $\implies$ Thiếu khớp (Underfitting), làm mờ ranh giới quyết định.
- **Lời nguyền số chiều (Curse of Dimensionality):** Khi số chiều $D$ lớn, toàn bộ các điểm trong không gian đều trở nên "xa nhau" và khoảng cách giữa các điểm có xu hướng tiệm cận nhau, khiến khái niệm "láng giềng gần nhất" mất đi ý nghĩa phân biệt.

---

## 8.2 Cây quyết định (Decision Trees)

Cây quyết định phân chia không gian đặc trưng nhiều chiều thành các khối hộp chữ nhật trực giao (Axis-aligned hyperrectangles) thông qua chuỗi các câu hỏi điều kiện dạng `IF feature_j <= threshold THEN ...`.

```
                  [ Tuổi <= 30? ]
                    ╱         ╲
              Có  ╱             ╲  Không
                ▼                 ▼
        [ Thu nhập <= 15tr? ]   [ Phê duyệt (y=1) ]
           ╱              ╲
     Có  ╱                  ╲  Không
       ▼                      ▼
  [ Từ chối (y=0) ]     [ Phê duyệt (y=1) ]
```

### 8.2.1 Độ thuần khiết và Tiêu chuẩn phân chia (Splitting Criteria)
Mục tiêu tại mỗi nút là chọn đặc trưng $j$ và ngưỡng $\theta$ sao cho tập dữ liệu con sinh ra có **độ thuần khiết (Purity)** cao nhất (các mẫu cùng nhãn nằm chung một nhóm).

#### 1. Entropy và Độ lợi thông tin (Information Gain - Thuật toán ID3):
- **Entropy của tập dữ liệu $S$ có $C$ lớp:**
  $$H(S) = -\sum_{c=1}^C p_c \log_2(p_c)$$
- **Độ lợi thông tin khi phân chia tập $S$ theo thuộc tính $A$:**
  $$\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
  Thuật toán ID3 chọn thuộc tính $A^*$ làm cực đại hóa $\text{IG}(S, A)$.

#### 2. Chỉ số vẩn đục Gini (Gini Impurity - Thuật toán CART):
- Đo xác suất một phần tử bị gán nhãn sai nếu nhãn được gán ngẫu nhiên theo phân phối của tập dữ liệu:
  $$\text{Gini}(S) = 1 - \sum_{c=1}^C p_c^2$$
  - $\text{Gini}(S) = 0$: Nút hoàn toàn thuần khiết (tất cả các mẫu thuộc cùng 1 lớp).
  - Với phân loại nhị phân cân bằng, $\text{Gini}_{\max} = 0.5$, trong khi $H_{\max} = 1.0$.

### 8.2.2 Cắt tỉa cây (Tree Pruning) để chống Quá khớp
Nếu để cây phát triển tối đa cho đến khi mọi nút lá đều thuần khiết ($H = 0$), cây quyết định sẽ có phương sai khổng lồ và bị quá khớp nghiêm trọng.
- **Tiền tỉa cành (Pre-pruning):** Dừng phân chia sớm dựa trên các ngưỡng khống chế: `max_depth` (độ sâu tối đa), `min_samples_split` (số mẫu tối thiểu để phân chia), `min_samples_leaf`.
- **Hậu tỉa cành (Post-pruning / Cost-Complexity Pruning):** Cho cây mọc tối đa rồi cắt bỏ các nhánh phụ làm tăng hàm mục tiêu phạt độ phức tạp: $R_\alpha(T) = R(T) + \alpha |T|$.

---

## 8.3 Rừng ngẫu nhiên (Random Forest)

Random Forest (Breiman, 2001) là thuật toán học tổ hợp kết hợp một số lượng lớn các cây quyết định cá lẻ ($B$ cây) để tạo ra mô hình có năng lực dự đoán vượt trội và độ ổn định cao.

### 8.3.1 Hai cơ chế ngẫu nhiên hóa cốt lõi:
1. **Lấy mẫu có hoàn lại (Bootstrap Aggregating / Bagging):**
   Mỗi cây $b \in \{1, \dots, B\}$ được huấn luyện trên một tập dữ liệu con $\mathcal{D}_b$ kích thước $N$ được rút ngẫu nhiên có hoàn lại từ tập dữ liệu gốc $\mathcal{D}$.
   - Trung bình một tập Bootstrap chỉ chứa khoảng $1 - \frac{1}{e} \approx 63.2\%$ số mẫu ban đầu.
   - $36.8\%$ số mẫu còn lại không được chọn gọi là **Out-Of-Bag (OOB)**, được sử dụng để kiểm thử độc lập mà không cần tập validation riêng!
2. **Không gian đặc trưng con ngẫu nhiên (Random Subspace):**
   Tại mỗi nút phân chia của từng cây, thuật toán không tìm kiếm trên toàn bộ $D$ đặc trưng mà chỉ chọn ngẫu nhiên một tập con gồm:
   $$m \approx \sqrt{D} \quad (\text{cho phân loại}) \qquad \text{hoặc} \qquad m \approx \frac{D}{3} \quad (\text{cho hồi quy})$$
   *Ý nghĩa:* Cơ chế này ép các cây phải học các đặc trưng khác nhau, làm giảm tối đa sự tương quan giữa các cây trong rừng.

### 8.3.2 Phân tích toán học giảm Phương sai
Xét trung bình dự đoán của $B$ cây ngẫu nhiên, mỗi cây có phương sai $\sigma^2$ và hệ số tương quan cặp đôi giữa hai cây bất kỳ là $\rho$:
$$\text{Var}\left( \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}) \right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
- Khi số lượng cây $B \to \infty$, thành phần thứ hai triệt tiêu về $0$:
  $$\lim_{B \to \infty} \text{Var} = \rho \sigma^2$$
- Bằng cách ngẫu nhiên hóa không gian đặc trưng, Random Forest làm giảm mạnh hệ số tương quan $\rho$, từ đó kéo sập phương sai tổng thể của mô hình mà **không làm tăng thiên kiến (Bias)**!

---

## 8.4 Hồi quy Tuyến tính (Linear Regression)

- **Mô hình:** $\hat{y} = \mathbf{w}^T \mathbf{x} + b$.
- **Hàm mất mát bình phương tối thiểu:**
  $$\mathcal{L}(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N (y_i - (\mathbf{w}^T \mathbf{x}_i + b))^2$$
- **Nghiệm giải tích (Normal Equation):** Khi gom hệ số tự do $b$ vào vector trọng số:
  $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

## 8.5 Hồi quy Logistic (Logistic Regression)

### 8.5.1 Tại sao Hồi quy Tuyến tính không phù hợp cho Phân loại?
1. Đầu ra của hồi quy tuyến tính $\mathbf{w}^T \mathbf{x} + b \in (-\infty, +\infty)$, không thể xem như một xác suất chuẩn hóa trong $[0, 1]$.
2. Rất nhạy cảm với các điểm ngoại lai nằm rất xa đường biên phân chia, làm lệch ranh giới quyết định một cách vô lý.

### 8.5.2 Tỷ số Chênh (Odds Ratio) và Hàm Sigmoid
- Tỷ số chênh (Odds) của biến cố xác suất $p$: $\text{Odds} = \frac{p}{1 - p} \in (0, \infty)$.
- Lấy logarit tự nhiên (Logit / Log-Odds) và giả định nó là một hàm tuyến tính của $\mathbf{x}$:
  $$\log\left( \frac{p}{1 - p} \right) = \mathbf{w}^T \mathbf{x} + b \implies p = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Hàm Sigmoid $\sigma(z)$:** Ánh xạ mọi giá trị $z \in \mathbb{R}$ về khoảng xác suất $(0, 1)$, với tính chất đạo hàm đẹp đẽ:
  $$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

```
            σ(z) ▲
               1 ┼───────────────────────╭────────
                 │                      ╱
             0.5 ┼─────────────────────●─────────  (z = 0 => p = 0.5)
                 │                   ╱
               0 ┼─────────╮────────┴────────────► z
                          -4          0          4
```

### 8.5.3 Hàm mất mát Binary Cross-Entropy (Log-Loss)
Từ nguyên lý Hợp lý cực đại (MLE) trên phân phối Bernoulli:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- Đạo hàm Gradient để tối ưu hóa:
  $$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
  Công thức đạo hàm có dạng giống hệt Hồi quy tuyến tính, nhưng $\hat{\mathbf{y}} = \sigma(\mathbf{X}\mathbf{w})$ là một hàm phi tuyến tính.

---

## 8.6 Hồi quy Softmax (Multinomial Logistic Regression)

Mở rộng Hồi quy Logistic cho bài toán phân loại đa lớp $C > 2$:
Mỗi lớp $c \in \{1, \dots, C\}$ sở hữu một vector tham số $\mathbf{w}_c$. Điểm số tuyến tính (Logit): $z_c = \mathbf{w}_c^T \mathbf{x} + b_c$.

### 8.6.1 Hàm kích hoạt Softmax
Biến đổi vector điểm số $\mathbf{z} \in \mathbb{R}^C$ thành phân phối xác suất hợp lệ:
$$P(y = c \mid \mathbf{x}) = \hat{p}_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}, \quad \text{thỏa mãn } \hat{p}_c > 0 \text{ và } \sum_{c=1}^C \hat{p}_c = 1$$

### 8.6.2 Hàm mất mát Categorical Cross-Entropy
$$\mathcal{L}(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(\hat{p}_{ic})$$
Trong đó $y_{ic} \in \{0, 1\}$ là biểu diễn One-hot của nhãn thực tế.

---

## 8.7 Các phương pháp Học tổ hợp (Ensemble Learning)

```
                              ┌─────────────────────────────┐
                              │  HỌC TỔ HỢP (ENSEMBLE)      │
                              └──────────────┬──────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
    │     Bagging     │             │    Boosting     │             │    Stacking     │
    ├─────────────────┤             ├─────────────────┤             ├─────────────────┤
    │ • Huấn luyện    │             │ • Huấn luyện    │             │ • Dùng Meta-    │
    │   song song độc │             │   tuần tự sửa   │             │   learner để    │
    │   lập           │             │   lỗi           │             │   học cách phối │
    │ • Giảm Variance │             │ • Giảm Bias     │             │   hợp mô hình   │
    │ • Random Forest │             │ • XGBoost, GBDT │             │ • Multi-model   │
    └─────────────────┘             └─────────────────┘             └─────────────────┘
```

### 8.7.1 Bagging (Bootstrap Aggregating)
- Huấn luyện song song nhiều mô hình cơ sở độc lập (Base learners có dung lượng cao, High Variance) trên các tập mẫu Bootstrap.
- Tổng hợp kết quả bằng biểu quyết đa số (Phân loại) hoặc tính trung bình cộng (Hồi quy).
- Đại diện tiêu biểu: **Random Forest**.

### 8.7.2 Boosting
- Huấn luyện tuần tự một chuỗi các mô hình yếu (Weak learners có dung lượng thấp, High Bias, như cây quyết định nông - Decision Stumps).
- Mô hình sau tập trung tăng trọng số cho các mẫu bị mô hình trước dự đoán sai:
  1. **AdaBoost (Adaptive Boosting):** Tăng trọng số dữ liệu của các ca khó.
  2. **Gradient Boosting Machine (GBM):** Mỗi cây mới được huấn luyện để khớp trực tiếp với **phần dư (Residual / Âm gradient của hàm mất mát)** của toàn bộ các cây trước đó:
     $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$
  3. **Các framework chuẩn công nghiệp:** **XGBoost** (áp dụng xấp xỉ Taylor bậc 2 và chính quy hóa độ phức tạp cây), **LightGBM** (histogram-based và Leaf-wise split), **CatBoost** (xử lý đặc trưng danh mục vượt trội).

### 8.7.3 Stacking (Stacked Generalization)
- Huấn luyện nhiều mô hình nền tảng khác loại (ví dụ: Random Forest, SVM, k-NN, Logistic Regression).
- Dự đoán của các mô hình nền tảng được gom lại thành một tập đặc trưng mới. Một mô hình cấp cao (**Meta-Learner**, thường là Logistic Regression) được huấn luyện trên tập đặc trưng này để đưa ra quyết định tổng hợp cuối cùng.

---

## 8.8 Tài liệu tham khảo (References)

1. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer. [Tham khảo: `resources/The Elements of Statistical Learning.pdf`] (Chương 9: Additive Models, Trees, and Related Methods; Chương 10: Boosting and Additive Trees; Chương 15: Random Forests).
2. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. [Tham khảo: `resources/Pattern Recognition and Machine Learning.pdf`] (Chương 4: Linear Models for Classification; Chương 14: Combining Models).
3. **Vũ Hữu Tiệp (2018).** *Machine Learning Cơ Bản*. NXB Khoa học và Kỹ thuật. [Tham khảo: `resources/Machine Learning Co Ban.pdf`] (Chương 3: Linear Regression; Chương 10: Logistic Regression; Chương 13: Softmax Regression).
4. **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5-32. (Bài báo gốc khai sinh thuật toán Random Forest).
5. **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *ACM SIGKDD*, 785-794.

---

## 8.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng bài thực hành toàn diện đối chiếu các thuật toán học máy cổ điển:

1. **Thành phần 1 - Cài đặt $k$-NN từ đầu (from scratch):**
   - Tự viết hàm tính ma trận khoảng cách Euclidean và bầu chọn nhãn đa số.
   - Khảo sát sự thay đổi của ranh giới quyết định (Decision Boundary) khi $k$ tăng từ $1 \to 5 \to 30$.
2. **Thành phần 2 - Cài đặt Cây quyết định (Decision Tree Classifier from scratch):**
   - Cài đặt hàm tính chỉ số Gini Impurity và hàm tìm điểm cắt tối ưu (Best Split).
   - Huấn luyện cây phân loại nhị phân trên dữ liệu 2 chiều.
3. **Thành phần 3 - Cài đặt Hồi quy Logistic từ đầu bằng Gradient Descent:**
   - Huấn luyện hàm Sigmoid với Binary Cross-Entropy Loss.
   - Trực quan hóa đường biên phân chia xác suất $P(y=1 \mid \mathbf{x}) = 0.5$.
4. **Thành phần 4 - Đối chiếu sức mạnh Học tổ hợp (Single Tree vs. Random Forest vs. Gradient Boosting):**
   - Huấn luyện một cây quyết định đơn lẻ (Single Tree), mô hình Rừng ngẫu nhiên (Random Forest) và mô hình Boosting (AdaBoost/GBDT).
   - Vẽ ranh giới phân chia của cả 3 mô hình cạnh nhau để thấy rõ khả năng làm mượt và ổn định ranh giới của Ensemble Learning.
