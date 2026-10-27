# Day 01: Random Forest — Random Forests

> **Bài báo gốc:** *Random Forests*  
> **Tác giả:** Leo Breiman  
> **Xuất bản:** Tạp chí *Machine Learning*, Tập 45, Số 1, Tháng 10 năm 2001, Trang 5–32.  
> **Tài liệu PDF gốc:** [`resources/Day 1 Random Forest.pdf`](./resources/Day%201%20Random%20Forest.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Sự Đánh Đổi Bias - Variance](#2-bối-cảnh-lịch-sử--sự-đánh-đổi-bias---variance)
3. [Kiến Trúc & Cơ Chế Hoạt Động Của Random Forest](#3-kiến-trúc--cơ-chế-hoạt-động-của-random-forest)
4. [Đột Phá Kỹ Thuật & Cơ Sở Toán Học Về Phương Sai](#4-đột-phá-kỹ-thuật--cơ-sở-toán-học-về-phương-sai)
5. [Các Đột Phá Bổ Trợ Của Breiman (OOB & Feature Importance)](#5-các-đột-phá-bổ-trợ-của-breiman-oob--feature-importance)
6. [Quy Trình Huấn Luyện & Tham Số Tối Ưu](#6-quy-trình-huấn-luyện--tham-số-tối-ưu)
7. [Kết Quả Thực Nghiệm Trong Bài Báo](#7-kết-quả-thực-nghiệm-trong-bài-báo)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Vào đầu thế kỷ 21, Giáo sư **Leo Breiman** (nhà thống kê lỗi lạc tại Đại học California, Berkeley, tác giả của thuật toán phân loại cây CART kinh điển năm 1984) đã công bố một công trình mang tính định hình toàn bộ lĩnh vực Học máy hiện đại: **Random Forests**.

Xuất bản năm 2001 trên tạp chí *Machine Learning*, bài báo đã mở ra kỷ nguyên vàng của phương pháp **Học kết hợp (Ensemble Learning)**:
- Thay vì cố gắng xây dựng một mô hình duy nhất thật phức tạp và hoàn hảo, ta tập hợp **hàng trăm "chuyên gia yếu" (Weak Learners - các cây quyết định ngẫu nhiên)** để đưa ra quyết định tập thể.
- Thuật toán kết hợp một cách tao nhã giữa hai kỹ thuật: **Bagging (Bootstrap Aggregating)** và **Không gian con ngẫu nhiên (Random Subspace Method)**.
- Trở thành một trong những thuật toán học máy đa năng, mạnh mẽ và ổn định nhất thế giới, là chuẩn mực mặc định (strong baseline) cho hầu hết các bài toán dữ liệu dạng bảng (Tabular Data) trong suốt hơn hai thập kỷ qua.

```text
                  DỮ LIỆU HUẤN LUYỆN (N mẫu, M đặc trưng)
               ┌──────────────────────┴──────────────────────┐
               │                                             │
      Bootstrap Sample 1                            Bootstrap Sample B
 (N mẫu có hoàn lại, m đặc trưng)              (N mẫu có hoàn lại, m đặc trưng)
               │                                             │
               ▼                                             ▼
        [Cây Quyết Định 1]      ... ... ...          [Cây Quyết Định B]
               │                                             │
               ▼                                             ▼
          Dự đoán y_1                                   Dự đoán y_B
               └──────────────────────┬──────────────────────┘
                                      ▼
                        BỎ PHIẾU ĐA SỐ / TRUNG BÌNH
                                      ▼
                               KẾT QUẢ CUỐI CÙNG
```

### Đóng góp cốt lõi của nghiên cứu:
1. **Triệt tiêu phương sai mà không làm tăng độ chệch**: Bằng cách ngẫu nhiên hóa việc lựa chọn đặc trưng tại mỗi điểm rẽ nhánh, Breiman đã kéo giảm hệ số tương quan giữa các cây ($\rho$) xuống mức tối thiểu, giúp phương sai của toàn bộ quần thể suy giảm đột phá.
2. **Ước lượng lỗi không cần tập kiểm định (Out-of-Bag - OOB Error)**: Khám phá ra việc tận dụng ~36.8% dữ liệu bị bỏ sót trong mỗi mẫu bootstrap để kiểm định chất lượng mô hình miễn phí mà không cần Cross-Validation.
3. **Độ đo tầm quan trọng của đặc trưng (Feature Importance)**: Phát minh ra độ đo Permutation Importance và MDI (Mean Decrease Impurity) giúp giải thích mô hình hộp đen một cách toán học.

---

## 2. Bối Cảnh Lịch Sử & Sự Đánh Đổi Bias - Variance

### 2.1. Bản chất của Cây Quyết Định Đơn Lẻ (Single Decision Tree)
Cây quyết định (Decision Tree - CART) là một mô hình trực quan, phi tham số (non-parametric), có khả năng học các mối quan hệ phi tuyến phức tạp:
- **Ưu điểm:** Độ chệch rất thấp (**Low Bias**), có khả năng ghi nhớ chi tiết biên phân chia dữ liệu.
- **Tử huyệt:** Phương sai cực cao (**High Variance**). Một biến động nhỏ trong tập dữ liệu huấn luyện có thể làm thay đổi hoàn toàn cấu trúc rẽ nhánh của cây. Nếu để cây phát triển sâu tối đa (unpruned), mô hình sẽ bị quá khớp (overfitting) trầm trọng.

### 2.2. Hạn chế của Bagging đơn thuần (Breiman, 1996)
Kỹ thuật Bagging lấy mẫu lặp lại có hoàn lại (Bootstrap) để huấn luyện $B$ cây quyết định độc lập. Giả sử mỗi cây có phương sai $\sigma^2$ và hệ số tương quan giữa hai cây bất kỳ là $\rho$:

$$\text{Var}\left(\frac{1}{B} \sum_{b=1}^B T_b(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

- Khi số lượng cây $B \to \infty$, số hạng thứ hai $\frac{1 - \rho}{B}\sigma^2 \to 0$.
- Tuy nhiên, số hạng thứ nhất **$\rho \sigma^2$ vẫn tồn tại**!
- **Vấn đề cốt tử của Bagging:** Nếu trong tập dữ liệu có một vài đặc trưng quá mạnh (dominant features), hầu hết các cây trong rừng sẽ đều chọn các đặc trưng đó để phân chia ở các nút gốc trên cùng. Kết quả là các cây **rất giống nhau (highly correlated, $\rho$ lớn)**. Do đó, việc lấy trung bình nhiều cây không còn mang lại nhiều tác dụng giảm phương sai.

---

## 3. Kiến Trúc & Cơ Chế Hoạt Động Của Random Forest

Breiman đã giải quyết triệt để vấn đề tương quan cao của Bagging bằng cơ chế "Ngẫu nhiên kép" (**Double Randomization**):

### 3.1. Ngẫu nhiên hóa mẫu dữ liệu (Data Bootstrap)
- Cho tập dữ liệu $D$ gồm $N$ mẫu.
- Với mỗi cây $b \in \{1, \dots, B\}$, ta rút trích ngẫu nhiên **$N$ mẫu có hoàn lại (with replacement)** từ $D$.
- Mỗi cây được huấn luyện trên một tập con dữ liệu độc lập $D_b$.

### 3.2. Ngẫu nhiên hóa không gian đặc trưng (Random Subspace / Feature Bagging)
- Tại **mỗi nút rẽ nhánh** trong quá trình xây dựng cây:
  - Thay vì duyệt qua toàn bộ $M$ đặc trưng để tìm điểm cắt tối ưu nhất (như cây CART truyền thống).
  - Thuật toán chỉ chọn ngẫu nhiên một tập con gồm **$m$ đặc trưng** ($m \ll M$).
  - Giá trị khuyến nghị của Breiman:
    - Bài toán phân loại (Classification): $m = \lfloor \sqrt{M} \rfloor$.
    - Bài toán hồi quy (Regression): $m = \lfloor M / 3 \rfloor$.
  - Tìm đặc trưng tốt nhất và ngưỡng phân chia tối ưu chỉ trong phạm vi $m$ đặc trưng này.

### 3.3. Xây dựng cây không cắt tỉa (Unpruned Deep Trees)
- Mỗi cây trong Random Forest được cho phép **phát triển sâu tối đa** cho đến khi các nút lá hoàn toàn thuần khiết (Pure leaf) hoặc số mẫu nhỏ hơn ngưỡng tối thiểu (`min_samples_split`).
- Không cần áp dụng các kỹ thuật cắt tỉa hậu kỳ phức tạp (cost-complexity pruning).

### 3.4. Tổng hợp kết quả (Ensemble Aggregation)
- **Phân loại:** Bỏ phiếu đa số (Majority Voting) hoặc trung bình xác suất (Soft Voting):
  $$\hat{y} = \arg\max_c \sum_{b=1}^B \mathbb{I}(T_b(x) = c)$$
- **Hồi quy:** Lấy trung bình cộng dự đoán của $B$ cây:
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x)$$

---

## 4. Đột Phá Kỹ Thuật & Cơ Sở Toán Học Về Phương Sai

### 4.1. Cơ chế hóa giải tương quan $\rho$
Nhờ việc ép mỗi nút chỉ được chọn trong số $m = \sqrt{M}$ đặc trưng:
- Các đặc trưng mạnh nhất không thể xuất hiện ở tất cả các nút gốc của mọi cây.
- Các đặc trưng yếu hơn hoặc thứ cấp có cơ hội được lựa chọn, giúp cây khai phá các góc nhìn tiềm ẩn khác nhau của dữ liệu.
- **Hệ quả toán học:** Hệ số tương quan $\rho$ giữa các cây giảm mạnh. Mặc dù phương sai của từng cây riêng lẻ $\sigma^2$ có thể tăng nhẹ (do cây bị hạn chế đặc trưng), nhưng **mức giảm của $\rho$ vượt trội hơn rất nhiều**, làm giảm phương sai tổng thể $\rho \sigma^2$ xuống mức cực thấp!

### 4.2. Định lý Giới Hạn Sai Số Tổng Quát Hóa (Breiman's Generalization Error Bound)
Breiman đã chứng minh rằng sai số tổng quát hóa của Random Forest bị chặn trên bởi một cận toán học chặt chẽ:

$$\text{PE}^* \le \bar{\rho} \frac{1 - s^2}{s^2}$$

Trong đó:
- $s$: Độ mạnh trung bình (Strength) của từng cây riêng lẻ (đo lường độ chính xác phân loại vượt trên mức ngẫu nhiên).
- $\bar{\rho}$: Hệ số tương quan trung bình giữa các cây trong rừng.
- **Ý nghĩa then chốt:** Khi số lượng cây $B \to \infty$, **Random Forest KHÔNG BAO GIỜ bị quá khớp (overfit)** do thêm nhiều cây! Càng tăng $B$, sai số kiểm định sẽ hội tụ tiệm cận về một hằng số giới hạn thay vì tăng vọt lên như các mô hình khác.

---

## 5. Các Đột Phá Bổ Trợ Của Breiman (OOB & Feature Importance)

### 5.1. Ước Lượng Lỗi Ngoại Mẫu (Out-of-Bag - OOB Error)
Trong quá trình rút mẫu Bootstrap với $N$ lần chọn có hoàn lại từ $N$ phần tử:
- Xác suất để một mẫu cụ thể **không bao giờ được chọn** trong một lần rút là: $1 - \frac{1}{N}$.
- Xác suất để mẫu đó không được chọn trong cả $N$ lần rút:
  $$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.3679 \approx 36.8\%$$

```text
Tập dữ liệu gốc (100% mẫu)
┌──────────────────────────────────────────────┐
│ Huấn luyện Cây b (~63.2% mẫu được chọn)     │  ==> Xây dựng cấu trúc cây
├──────────────────────────────────────────────┤
│ Out-of-Bag (OOB) (~36.8% mẫu KHÔNG được chọn)│  ==> Đánh giá kiểm định độc lập!
└──────────────────────────────────────────────┘
```

- Với mỗi mẫu dữ liệu $x_i$, ta thu thập dự đoán từ **tất cả các cây KHÔNG sử dụng $x_i$ trong tập huấn luyện** của chúng (khoảng $B/3$ cây).
- **Hệ quả tuyệt vời:** OOB Error là một **ước lượng không chệch (unbiased estimate)** của sai số kiểm định thực tế. Người dùng có thể đánh giá chính xác hiệu năng mô hình ngay trong quá trình huấn luyện mà không cần tốn công chia tập Validation hay chạy K-Fold Cross-Validation!

### 5.2. Đo Lường Tầm Quan Trọng Của Đặc Trưng (Feature Importance)
Breiman đề xuất hai phương pháp đo lường tầm quan trọng của đặc trưng:

1. **Mean Decrease Impurity (MDI / Gini Importance):**
   Tổng mức suy giảm độ vẩn đục Gini (hoặc Entropy) mang lại bởi đặc trưng $j$ trên toàn bộ các nút rẽ nhánh của tất cả $B$ cây:
   $$\text{MDI}(j) = \frac{1}{B} \sum_{b=1}^B \sum_{t \in T_b, v(t)=j} \Delta I(t)$$
2. **Mean Decrease Accuracy (MDA / Permutation Importance):**
   Độ đo chuẩn xác và khách quan hơn:
   - Tính toán độ chính xác OOB ban đầu của mô hình.
   - Xáo trộn ngẫu nhiên (shuffle/permute) các giá trị của đặc trưng $j$ trong tập OOB (phá hủy mối quan hệ giữa đặc trưng $j$ và nhãn mục tiêu).
   - Đo mức suy giảm độ chính xác OOB. Nếu độ chính xác tụt dốc thảm hại, đặc trưng $j$ là cực kỳ quan trọng!

---

## 6. Quy Trình Huấn Luyện & Tham Số Tối Ưu

### Bảng Siêu Tham Số Cốt Lõi:

| Tham Số | Ký Hiệu | Khuyến Nghị Chuẩn Của Breiman | Ý Nghĩa Thực Tế |
|---|:---:|:---:|---|
| **Số lượng cây (`n_estimators`)** | $B$ | $100 - 500$ | Càng lớn càng ổn định, không gây overfit nhưng tốn thời gian tính toán. |
| **Số đặc trưng rẽ nhánh (`max_features`)** | $m$ | $\sqrt{M}$ (Phân loại), $M/3$ (Hồi quy) | Giảm $m$ làm giảm tương quan $\rho$ nhưng cũng làm giảm nhẹ độ mạnh $s$. |
| **Độ sâu tối đa (`max_depth`)** | $d$ | `None` (để sâu tối đa) | Cây càng sâu càng bảo đảm độ chệch thấp (Low Bias). |
| **Mẫu tối thiểu rẽ nhánh (`min_samples_split`)** | - | $2$ (hoặc $5$) | Điều kiện dừng phân chia nút. |
| **Tiêu chí phân chia (`criterion`)** | - | `Gini` hoặc `Entropy` | Gini Impurity: $I_G(t) = 1 - \sum_{c} p_c^2$. |

---

## 7. Kết Quả Thực Nghiệm Trong Bài Báo

Trong bài báo năm 2001, Leo Breiman đã so sánh toàn diện Random Forest với **Adaboost** (thuật toán số 1 lúc bấy giờ của Freund & Schapire) và **Bagging chuẩn** trên gần 20 tập dữ liệu thực tế từ UCI Repository:
- **Độ chính xác ngang ngửa hoặc vượt trội Adaboost**: Đạt tỷ lệ lỗi cực thấp trên các bài toán phân loại phức tạp (như Sonar, Breast Cancer, Glass, Ionosphere).
- **Khả năng chống nhiễu vượt trội (Noise Robustness)**: Adaboost rất dễ bị phá hủy khi dữ liệu bị dán nhãn sai (label noise) vì thuật toán boosting sẽ dồn trọng số cực đại vào các mẫu nhiễu. Ngược lại, Random Forest phân bổ trọng số đồng đều, hoạt động cực kỳ bền bỉ và miễn nhiễm với nhiễu.
- **Tốc độ huấn luyện cực nhanh**: Khả năng phân nhánh trên tập con $m = \sqrt{M}$ giúp việc huấn luyện nhanh hơn gấp hàng chục lần so với việc duyệt toàn bộ đặc trưng.

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Hạn chế của Random Forest
1. **Chi phí bộ nhớ và tốc độ suy luận**: Khi rừng có hàng nghìn cây sâu, kích thước mô hình lưu trữ có thể lên tới hàng Gigabyte và tốc độ dự đoán (Inference Latency) chậm hơn so với mạng nơ-ron tuyến tính.
2. **Kém khả năng ngoại suy (No Extrapolation)**: Cây quyết định chỉ dự đoán giá trị trung bình trong các vùng không gian đã thấy. Trong các bài toán chuỗi thời gian hoặc dữ liệu có xu hướng (trend), Random Forest không thể dự đoán vượt ra ngoài biên giá trị của tập huấn luyện.

### 8.2. Sự tiến hóa của các mô hình cây quyết định
- **Extremely Randomized Trees (Extra-Trees - Geurts et al., 2006)**: Tăng thêm một mức ngẫu nhiên hóa bằng cách chọn ngưỡng phân cắt hoàn toàn ngẫu nhiên thay vì tìm ngưỡng tối ưu.
- **Gradient Boosted Decision Trees (GBDT)**: XGBoost (2016), LightGBM (2017), CatBoost (2018) kết hợp kiến trúc tối ưu hóa theo gradient của cây, trở thành đối trọng lớn nhất của Random Forest trên Kaggle.
- **Differentiable Neural Decision Forests (Konta et al., 2015)**: Biến đổi các phép rẽ nhánh cứng thành hàm sigmoid mềm để tích hợp và huấn luyện end-to-end cùng mạng nơ-ron sâu bằng PyTorch.

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục cung cấp 2 tệp mã nguồn hoàn chỉnh:

```text
Day 01 - Random Forest/
├── README.md                  # Báo cáo lý thuyết chuyên sâu 10 mục chuẩn này
├── random_forest_scratch.py   # Cài đặt CART Tree, Random Forest, OOB Error & MDI bằng Python & NumPy
├── random_forest_scratch.ipynb# Notebook thực hành tương tác trực quan (From Scratch)
├── random_forest_pytorch.py   # Triển khai Rừng Nơ-ron Vi Phân (Neural Decision Forest) bằng PyTorch
├── random_forest_pytorch.ipynb# Notebook thực hành tương tác & trực quan hóa PyTorch
└── resources/
    └── Day 1 Random Forest.pdf # Tài liệu bài báo gốc của Leo Breiman
```

### 9.1. Tệp `random_forest_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python chuẩn + NumPy.
- **Tính năng nổi bật**:
  - `DecisionTreeScratch`: Tự viết thuật toán tính độ vẩn đục Gini, tìm ngưỡng cắt tối ưu trên tập con ngẫu nhiên `max_features = sqrt(M)`.
  - `RandomForestScratch`: Lấy mẫu Bootstrap, huấn luyện song song $B$ cây, thực hiện cơ chế bỏ phiếu đa số.
  - Tự động tính toán **Ước lượng sai số ngoại mẫu Out-of-Bag (OOB Error)**.
  - Tự động tính toán và xuất bảng **Tầm quan trọng đặc trưng (Mean Decrease Impurity - MDI)** ra màn hình terminal.

### 9.2. Tệp `random_forest_pytorch.py` (Triển khai bằng PyTorch)
- **Công nghệ**: PyTorch, torchvision, matplotlib, scikit-learn.
- **Tính năng nổi bật**:
  - `NeuralRandomForest`: Triển khai mô hình rừng nơ-ron có thể vi phân (Differentiable Neural Forest) với các cổng rẽ nhánh mềm $\sigma(w \cdot x - b)$, cho phép tối ưu hóa trọng số bằng Gradient Descent và GPU.
  - So sánh trực tiếp hiệu năng với mô hình Random Forest cổ điển.
  - Tự động xuất biểu đồ đường biên phân loại (Decision Boundary) và tầm quan trọng đặc trưng vào thư mục `outputs/random_forest_evaluation.png`.

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Tại sao việc giới hạn số đặc trưng xem xét tại mỗi nút ở mức $m = \sqrt{M}$ lại giúp giảm phương sai của toàn bộ khu rừng mà không làm tăng độ chệch quá mức?**
2. **Chứng minh toán học vì sao tỷ lệ mẫu bị bỏ sót trong quá trình rút mẫu Bootstrap với số lượng mẫu lớn luôn xấp xỉ $1 - 1/e \approx 36.8\%$?**
3. **Phân biệt bản chất và so sánh ưu nhược điểm giữa hai độ đo tầm quan trọng đặc trưng: Mean Decrease Impurity (MDI) và Permutation Importance (MDA)?**
4. **Tại sao theo định lý giới hạn sai số của Breiman, việc tăng thêm số lượng cây $B$ trong Random Forest không bao giờ dẫn tới hiện tượng quá khớp (overfitting)?**
5. **So sánh sự khác biệt triết lý giữa phương pháp Bagging (Random Forest: các cây độc lập, giảm phương sai) và Boosting (AdaBoost/XGBoost: các cây tuần tự, giảm độ chệch)?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
