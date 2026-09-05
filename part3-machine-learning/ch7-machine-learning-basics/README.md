# Chương 7: Các nguyên lý cơ bản của Học máy (Machine Learning Basics)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Nắm vững định nghĩa hình thức của Học máy:** Hiểu sâu sắc định nghĩa bộ ba $(T, P, E)$ theo Tom Mitchell và phân biệt rõ ranh giới giữa lập trình truyền thống và học máy.
2. **Phân loại toàn diện các tác vụ $T$:** Nắm vững bản chất toán học của Hồi quy (Regression), Phân loại (Classification), Phân cụm (Clustering), Giảm chiều dữ liệu (Dimensionality Reduction) và Ước lượng mật độ (Density Estimation).
3. **Thành thạo các độ đo hiệu năng $P$:** Tính toán và biện luận chính xác Confusion Matrix, Accuracy, Precision, Recall, $F_1$-score, ROC-AUC cho bài toán phân loại; MSE, RMSE, MAE, $R^2$ cho bài toán hồi quy.
4. **Phân loại các hình thức học từ kinh nghiệm $E$:** Phân biệt Học có giám sát (Supervised), Tự giám sát (Self-Supervised), Không giám sát (Unsupervised) và Học tăng cường (Reinforcement Learning).
5. **Nắm bắt quy trình xây dựng hệ thống ML chuẩn mực:** Phân chia tập dữ liệu Train / Validation / Test, kỹ thuật K-Fold Cross-Validation, tránh hiện tượng rò rỉ dữ liệu (Data Leakage) và xử lý trôi dạt phân phối (Concept Drift).
6. **Chứng minh toán học sự đánh đổi Thiên kiến - Phương sai (Bias-Variance Tradeoff):** Khảo sát hiện tượng Thiếu khớp (Underfitting), Quá khớp (Overfitting), dung lượng mô hình (Capacity) và Định lý phân rã sai số.
7. **Làm chủ các kỹ thuật Chính quy hóa (Regularization):** Hiểu sâu bản chất hình học và xác suất của chuẩn $L_1$ (Lasso), chuẩn $L_2$ (Ridge / Weight Decay), và cơ chế Early Stopping.

---

## 7.1 Thế nào là “Học máy”? (What is Machine Learning?)

### 7.1.1 Định nghĩa kinh điển của Tom M. Mitchell (1997)
Trong cuốn giáo trình kinh điển *Machine Learning*, Tom Mitchell định nghĩa một cách chặt chẽ:
> *"Một chương trình máy tính được gọi là **học** từ kinh nghiệm $E$ (Experience) đối với một lớp tác vụ $T$ (Task) và một thước đo hiệu năng $P$ (Performance), nếu hiệu năng của nó đối với các tác vụ trong $T$, được đo lường bởi $P$, tự động cải thiện sau khi có thêm kinh nghiệm $E$."*

```
          ┌─────────────────────────────────────────────────────────┐
          │                    TRỤ CỘT HỌC MÁY                      │
          ├────────────────────┬────────────────────┬───────────────┤
          │     Tác vụ (T)     │   Hiệu năng (P)    │ Kinh nghiệm(E)│
          ├────────────────────┼────────────────────┼───────────────┤
          │ • Lọc email spam   │ • Tỷ lệ phân loại  │ • Tập 100.000 │
          │                    │   đúng (Accuracy)  │   email có gán│
          │                    │                    │   nhãn sẵn    │
          │ • Dự đoán giá nhà  │ • Sai số bình      │ • Dữ liệu các │
          │                    │   phương trung     │   giao dịch   │
          │                    │   bình (MSE)       │   quá khứ     │
          │ • Xe tự hành lái xe│ • Quãng đường đi an│ • Video & tín │
          │                    │   toàn không cần can│   hiệu cảm biến│
          │                    │   thiệp của người  │   lái xe thực │
          └────────────────────┴────────────────────┴───────────────┘
```

### 7.1.2 Chuyển dịch hệ chuẩn: Lập trình truyền thống vs. Học máy
- **Lập trình truyền thống (Software 1.0):** $\text{Dữ liệu đầu vào} + \text{Quy tắc thuật toán (Code)} \implies \text{Kết quả đầu ra}$.
- **Học máy (Software 2.0):** $\text{Dữ liệu đầu vào} + \text{Kết quả mẫu mong muốn} \implies \text{Mô hình tự sinh quy tắc}$.

---

## 7.2 Tác vụ $T$ (Machine Learning Tasks)

Học máy không phải là thuật toán đơn lẻ mà là phương tiện để giải quyết các tác vụ tính toán cụ thể:

### 7.2.1 Phân loại (Classification)
Ánh xạ từ không gian vector đặc trưng $\mathbf{x} \in \mathbb{R}^D$ sang một tập hữu hạn các nhãn lớp rời rạc $y \in \{1, 2, \dots, C\}$:
$$f: \mathbb{R}^D \to \{1, \dots, C\}$$
- **Phân loại nhị phân (Binary Classification):** $C = 2$ (ví dụ: Giao dịch gian lận vs. Hợp lệ; Khối u ác tính vs. Lành tính).
- **Phân loại đa lớp (Multi-class Classification):** $C > 2$, mỗi mẫu thuộc chính xác 1 lớp (ví dụ: Nhận dạng 10 chữ số viết tay MNIST).
- **Phân loại đa nhãn (Multi-label Classification):** Mỗi mẫu có thể đồng thời sở hữu nhiều nhãn khác nhau (ví dụ: Một bài báo vừa thuộc chuyên mục *Khoa học*, vừa thuộc *Kinh tế*).

### 7.2.2 Hồi quy (Regression)
Ánh xạ từ không gian vector đặc trưng $\mathbf{x} \in \mathbb{R}^D$ sang một hoặc nhiều giá trị số thực liên tục:
$$f: \mathbb{R}^D \to \mathbb{R}$$
- **Ví dụ:** Dự đoán giá cổ phiếu, nhiệt độ ngày mai, thời gian giao hàng dự kiến (ETA), nồng độ protein trong máu.

### 7.2.3 Phân cụm (Clustering)
Nhóm tập dữ liệu không nhãn $\{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ thành $K$ cụm sao cho các mẫu trong cùng một cụm có độ tương đồng tối đa, và các mẫu ở các cụm khác nhau có độ tương đồng tối thiểu.
- **Ví dụ:** Phân khúc khách hàng mục tiêu trong Marketing, nén bảng màu ảnh.

### 7.2.4 Giảm chiều dữ liệu (Dimensionality Reduction)
Ánh xạ dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn ($d \ll D$):
$$f: \mathbb{R}^D \to \mathbb{R}^d$$
sao cho bảo toàn tối đa thông tin quan trọng hoặc cấu trúc hình học lân cận của dữ liệu.
- **Mục đích:** Khử nhiễu, tiết kiệm bộ nhớ, tăng tốc độ huấn luyện mô hình và trực quan hóa dữ liệu trên mặt phẳng 2D/3D (PCA, t-SNE, UMAP).

---

## 7.3 Đo lường Hiệu năng $P$ (Performance Measures)

Việc chọn sai thước đo hiệu năng $P$ có thể dẫn tới các quyết định kinh doanh hoặc lâm sàng thảm họa.

### 7.3.1 Các độ đo cho Bài toán Phân loại (Classification Metrics)

#### Ma trận Nhầm lẫn (Confusion Matrix) cho Phân loại nhị phân:
| | Thực tế Dương tính ($Y=1$) | Thực tế Âm tính ($Y=0$) |
| :--- | :---: | :---: |
| **Dự đoán Dương tính ($\hat{Y}=1$)** | **True Positive (TP)** | **False Positive (FP)** *(Báo động giả)* |
| **Dự đoán Âm tính ($\hat{Y}=0$)** | **False Negative (FN)** *(Bỏ sót nguy hiểm)* | **True Negative (TN)** |

1. **Độ chính xác tổng thể (Accuracy):**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
   *Cảnh báo tử huyệt:* Vô dụng khi dữ liệu mất cân bằng nghiêm trọng (Imbalanced Data). Ví dụ: Trong bộ dữ liệu có $99\%$ giao dịch bình thường và $1\%$ giao dịch lừa đảo, một mô hình ngớ ngẩn luôn đoán "Bình thường" vẫn đạt $\text{Accuracy} = 99\%$ nhưng bắt trượt toàn bộ các vụ lừa đảo!
2. **Độ chuẩn xác (Precision):** Trong số những trường hợp mô hình khẳng định là Dương tính, có bao nhiêu phần trăm thực sự đúng?
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   *Ứng dụng:* Cần cực đại hóa Precision khi chi phí của False Positive quá đắt (ví dụ: Lọc thư rác – không được phép đẩy nhầm email đối tác quan trọng vào hòm thư rác).
3. **Độ nhạy / Độ phủ (Recall / Sensitivity / True Positive Rate):** Trong toàn bộ các ca thực tế là Dương tính, mô hình đã tìm ra được bao nhiêu phần trăm?
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   *Ứng dụng:* Cần cực đại hóa Recall khi chi phí của False Negative quá đắt (ví dụ: Chẩn đoán ung thư, phát hiện bom mìn – thà báo nhầm còn hơn bỏ sót người bệnh).
4. **Điểm $F_1$ ($F_1$-Score):** Trung bình điều hòa (Harmonic Mean) giữa Precision và Recall:
   $$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
5. **Đường cong ROC và Diện tích dưới đường cong (ROC-AUC):**
   - Vẽ đồ thị tỷ lệ bắt đúng $\text{TPR} = \text{Recall}$ theo tỷ lệ báo động nhầm $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ ở mọi ngưỡng phân ngưỡng (Threshold $\in [0, 1]$).
   - $\text{AUC} \in [0.5, 1.0]$: Đo lường năng lực xếp hạng xác suất của mô hình độc lập với ngưỡng quyết định.

### 7.3.2 Các độ đo cho Bài toán Hồi quy (Regression Metrics)
1. **Sai số bình phương trung bình (Mean Squared Error - MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   Nhạy cảm cao với các mẫu ngoại lai (Outliers) vì sai số bị bình phương lên.
2. **Căn bậc hai sai số bình phương trung bình (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
   Có cùng đơn vị đo lường với biến mục tiêu $y$, dễ giải thích trực quan.
3. **Sai số tuyệt đối trung bình (Mean Absolute Error - MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   Chống chịu mẫu ngoại lai tốt hơn (Robust to outliers) so với MSE.
4. **Hệ số xác định ($R^2$ Score / Coefficient of Determination):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
   Đo lường tỷ lệ phần trăm phương sai của $y$ được giải thích bởi các đặc trưng đầu vào $\mathbf{X}$. $R^2 = 1$ là hoàn hảo; $R^2 = 0$ nghĩa là mô hình chỉ tương đương việc đoán giá trị trung bình $\bar{y}$.

---

## 7.4 Kinh nghiệm $E$ (Forms of Experience)

```
                          ┌──────────────────────────┐
                          │   CÁC HÌNH THỨC HỌC (E)  │
                          └────────────┬─────────────┘
                                       │
         ┌──────────────────┬──────────┴──────────┬──────────────────┐
         ▼                  ▼                     ▼                  ▼
┌──────────────────┐ ┌──────────────┐   ┌──────────────────┐ ┌──────────────┐
│  Có giám sát     │ │ Tự giám sát  │   │  Không giám sát  │ │Học tăng cường│
│  (Supervised)    │ │(Self-Superv.)│   │  (Unsupervised)  │ │(Reinforcement│
├──────────────────┤ ├──────────────┤   ├──────────────────┤ ├──────────────┤
│ • Dữ liệu kèm    │ │ • Tự sinh    │   │ • Dữ liệu thô    │ │ • Thử & Sai  │
│   nhãn: (x, y)   │ │   nhãn từ cấu│   │   không nhãn: {x}│ │ • Hành động  │
│ • Học hàm ánh xạ │ │   trúc dữ liệu│  │ • Học mật độ P(x)│ │ • Phần thưởng│
└──────────────────┘ └──────────────┘   └──────────────────┘ └──────────────┘
```

1. **Học có giám sát (Supervised Learning):** Dữ liệu bao gồm các cặp vector đặc trưng và nhãn mong muốn: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$. Mô hình học cách ánh xạ $\mathbf{x} \mapsto y$.
2. **Học không giám sát (Unsupervised Learning):** Dữ liệu chỉ có các quan sát thô không có nhãn: $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$. Mục tiêu là giải mã cấu trúc phân phối tiềm ẩn $P(\mathbf{x})$, giảm chiều hoặc gom cụm dữ liệu.
3. **Học tự giám sát (Self-Supervised Learning - SSL):** Dữ liệu không nhãn được biến đổi tự động để tạo ra bài toán có giám sát: che một phần dữ liệu và yêu cầu mô hình dự đoán phần bị che (Masked Language Modeling trong BERT, Next-token prediction trong GPT, Contrastive Learning trong CLIP).
4. **Học tăng cường (Reinforcement Learning - RL):** Tác tử học thông qua quá trình tương tác trực tiếp với môi trường theo các chu kỳ Hành động $\to$ Quan sát trạng thái mới $\to$ Nhận phần thưởng (Reward), với mục tiêu cực đại hóa tổng phần thưởng tích lũy lâu dài.

---

## 7.5 Quá trình xây dựng mô hình AI hoàn chỉnh (End-to-End Pipeline)

```
[Thu thập dữ liệu] ──► [Tiền xử lý & Kỹ thuật đặc trưng] ──► [Phân chia Train/Val/Test]
                                                                      │
  ┌───────────────────────────────────────────────────────────────────┘
  ▼
[Huấn luyện trên Train] ──► [Tinh chỉnh trên Validation] ──► [Đánh giá trên Test]
                                                                      │
                                                                      ▼
                                                            [Deploy & Monitoring]
```

### 7.5.1 Chiến lược phân chia dữ liệu và Tránh rò rỉ (Data Leakage)
- **Train Set ($60 - 70\%$):** Dùng để cập nhật các tham số trọng số $\mathbf{W}, \mathbf{b}$ thông qua tối ưu hóa hàm mất mát.
- **Validation Set ($15 - 20\%$):** Dùng để đánh giá độc lập, lựa chọn kiến trúc mô hình và tinh chỉnh các siêu tham số (Hyperparameters: Learning rate, số tầng nơ-ron, hệ số chính quy hóa $\lambda$).
- **Test Set ($15 - 20\%$):** Tuyệt đối "khóa kín", chỉ được đem ra đánh giá duy nhất một lần cuối cùng để ước lượng hiệu năng thực tế khách quan ngoài đời.
- **Rò rỉ dữ liệu (Data Leakage):** Hiện tượng thông tin từ tập Validation/Test vô tình rò rỉ vào quá trình huấn luyện (ví dụ: Chuẩn hóa dữ liệu `StandardScaler` tính $\mu, \sigma$ trên toàn bộ tập dữ liệu trước khi phân chia Train/Test). Đây là sai lầm phổ biến khiến mô hình đạt điểm cao ngất ngưởng trong phòng thí nghiệm nhưng thất bại hoàn toàn khi triển khai thực tế.

### 7.5.2 K-Fold Cross-Validation
Khi tập dữ liệu có quy mô vừa hoặc nhỏ, ta chia tập dữ liệu thành $K$ phần bằng nhau. Mô hình được huấn luyện $K$ lần; mỗi lần chọn 1 phần làm Validation và $K-1$ phần còn lại làm Training. Điểm hiệu năng là trung bình cộng của $K$ lần kiểm thử, giúp triệt tiêu độ chệch do phân chia ngẫu nhiên.

---

## 7.6 Quá khớp (Overfitting), Thiếu khớp (Underfitting) và Sự đánh đổi Thiên kiến - Phương sai

### 7.6.1 Rủi ro kinh nghiệm vs. Rủi ro tổng quát hóa
- **Sai số huấn luyện (Training Error):** Sai số tính trên tập dữ liệu đã học.
- **Sai số tổng quát hóa (Generalization Error):** Sai số kỳ vọng trên các mẫu dữ liệu mới rút từ phân phối thực tế.
- **Thiếu khớp (Underfitting / High Bias):** Mô hình không thể đạt được sai số thấp ngay trên chính tập huấn luyện do dung lượng mô hình quá thấp (ví dụ: dùng đường thẳng tuyến tính để khớp dữ liệu hình sin uốn lượn).
- **Quá khớp (Overfitting / High Variance):** Mô hình đạt sai số gần bằng 0 trên tập huấn luyện, nhưng sai số trên tập kiểm tra tăng vọt. Mô hình đã "học vẹt" ghi nhớ cả các nhiễu ngẫu nhiên thay vì học quy luật cốt lõi.

### 7.6.2 Phân rã Thiên kiến - Phương sai (Bias-Variance Decomposition)
Cho bài toán hồi quy với nhãn thực tế $y = f(\mathbf{x}) + \epsilon$, với nhiễu ngẫu nhiên độc lập $\mathbb{E}[\epsilon] = 0$ và $\text{Var}(\epsilon) = \sigma^2$.  
Kỳ vọng sai số bình phương của một mô hình học $\hat{f}(\mathbf{x}; \mathcal{D})$ trên các tập dữ liệu huấn luyện khác nhau có thể được phân rã thành chính xác 3 thành phần:
$$\mathbb{E}_{\mathcal{D}}\left[ (y - \hat{f}(\mathbf{x}; \mathcal{D}))^2 \right] = \underbrace{\left( f(\mathbf{x}) - \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x})] \right)^2}_{\mathbf{\text{Bias}}^2 \text{ (Thiên kiến)}} + \underbrace{\mathbb{E}_{\mathcal{D}}\left[ \left( \hat{f}(\mathbf{x}) - \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x})] \right)^2 \right]}_{\mathbf{\text{Variance}} \text{ (Phương sai)}} + \underbrace{\sigma^2}_{\mathbf{\text{Nhiễu bất khả quy}}}$$

```
    Sai số
      ▲
      │             \         Tổng sai số (Generalization Error)
      │              \       ╱
      │               \_____╱  ◄── Điểm đánh đổi tối ưu (Sweet Spot)
      │               ╱     \
      │   Bias^2     ╱       \   Variance
      │   (Thiên    ╱         \  (Phương sai - Nhạy cảm dữ liệu)
      │    kiến)   ╱           \
      │───────────╱─────────────\────────────────────────► Dung lượng mô hình
               Thấp              Cao
            (Underfitting)    (Overfitting)
```

1. **Thiên kiến (Bias):** Sai số do những giả định sai lầm hoặc sự giản lược hóa quá mức của thuật toán học. Thiên kiến cao $\implies$ Thiếu khớp.
2. **Phương sai (Variance):** Mức độ nhạy cảm của mô hình trước những biến động ngẫu nhiên trong tập dữ liệu huấn luyện. Phương sai cao $\implies$ Quá khớp.
3. **Nhiễu bất khả quy ($\sigma^2$ - Irreducible Error):** Giới hạn vật lý của bài toán do sự thiếu hụt đặc trưng hoặc tính ngẫu nhiên nội tại của thế giới thực; không một mô hình nào có thể vượt qua giới hạn này.

---

## 7.7 Các kỹ thuật Chính quy hóa (Regularization)

Chính quy hóa là bất kỳ sửa đổi nào áp dụng vào thuật toán học nhằm **giảm sai số tổng quát hóa (Variance) mà không làm tăng quá mức sai số huấn luyện (Bias)**.

### 7.7.1 Ý tưởng cốt lõi: Phạt độ phức tạp của mô hình
Bổ sung một số hạng phạt $\Omega(\mathbf{w})$ vào hàm mất mát gốc:
$$J_{\text{reg}}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \Omega(\mathbf{w})$$
Trong đó $\lambda \ge 0$ là siêu tham số kiểm soát mức độ phạt.

### 7.7.2 Chính quy hóa $L_2$ (Ridge Regression / Weight Decay)
- Số hạng phạt: $\Omega(\mathbf{w}) = \frac{1}{2} \|\mathbf{w}\|_2^2 = \frac{1}{2} \sum_{j=1}^D w_j^2$.
- **Cơ chế hoạt động:** Ép tất cả các trọng số co cụm về gần 0 một cách mượt mà, ngăn cản mô hình phụ thuộc quá mức vào một vài đặc trưng cá biệt.
- Trong cập nhật Gradient Descent:
  $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta (\nabla \mathcal{L} + \lambda \mathbf{w}_t) = (1 - \eta \lambda) \mathbf{w}_t - \eta \nabla \mathcal{L}$$
  Hệ số $(1 - \eta \lambda) < 1$ làm tiêu hao dần trọng số ở mỗi bước, do đó có tên gọi là **Weight Decay**.

### 7.7.3 Chính quy hóa $L_1$ (Lasso Regression)
- Số hạng phạt: $\Omega(\mathbf{w}) = \|\mathbf{w}\|_1 = \sum_{j=1}^D |w_j|$.
- **Tính chất thưa (Sparsity Property):** Do các đường mức của chuẩn $L_1$ có các góc nhọn nằm ngay trên các trục tọa độ, điểm tiếp xúc tối ưu thường nằm chính xác tại các đỉnh góc. Điều này làm cho nhiều trọng số bị **triệt tiêu về đúng bằng 0**, biến $L_1$ thành một công cụ **tự động chọn lọc đặc trưng (Automatic Feature Selection)** cực kỳ mạnh mẽ.

---

## 7.8 Tài liệu tham khảo (References)

1. **Mitchell, T. M. (1997).** *Machine Learning*. McGraw-Hill. [Tham khảo: `resources/Machine Learning - Tom Mitchell.pdf`] (Chương 1: Introduction; Chương 5: Evaluating Hypotheses).
2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo: `resources/Deep Learning.pdf`] (Chương 5: Machine Learning Basics; Chương 7: Regularization for Deep Learning).
3. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer. [Tham khảo: `resources/The Elements of Statistical Learning.pdf`] (Chương 2: Overview of Supervised Learning; Chương 3: Linear Methods for Regression - Ridge and Lasso; Chương 7: Model Assessment and Selection).
4. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. [Tham khảo: `resources/Pattern Recognition and Machine Learning.pdf`] (Chương 1: Introduction - Polynomial Curve Fitting, The Curse of Dimensionality, Decision Theory; Chương 3: Linear Models for Regression - Bias-Variance Decomposition).
5. **Vũ Hữu Tiệp (2018).** *Machine Learning Cơ Bản*. NXB Khoa học và Kỹ thuật. [Tham khảo: `resources/Machine Learning Co Ban.pdf`] (Chương 1: Giới thiệu; Chương 11: Overfitting; Chương 12: Đánh giá mô hình phân loại).

---

## 7.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng bài thực nghiệm hoàn chỉnh để làm sáng tỏ các nguyên lý nền tảng của Học máy:

1. **Thành phần 1 - Mô phỏng Hiện tượng Khớp đa thức (Polynomial Curve Fitting):**
   - Sinh dữ liệu theo hàm sóng sin: $y = \sin(2\pi x) + \epsilon$ với số lượng mẫu hạn chế ($N = 15$).
   - Huấn luyện các mô hình đa thức bậc $d \in \{1, 3, 9\}$.
   - Trực quan hóa 3 hiện tượng: Bậc 1 bị Thiếu khớp (Underfitting), Bậc 3 khớp hoàn hảo (Sweet Spot), Bậc 9 bị Quá khớp dao động dữ dội (Overfitting).
2. **Thành phần 2 - Khảo sát Thực nghiệm Đánh đổi Thiên kiến - Phương sai (Bias-Variance Simulation):**
   - Sinh 100 tập dữ liệu huấn luyện độc lập từ cùng một phân phối.
   - Huấn luyện 100 mô hình đa thức bậc thấp và 100 mô hình đa thức bậc cao; trực quan hóa bó đường cong dự đoán để thấy rõ: Mô hình bậc thấp có Bias cao nhưng Variance thấp; mô hình bậc cao có Bias thấp nhưng Variance khổng lồ.
3. **Thành phần 3 - Khắc phục Overfitting bằng Chính quy hóa $L_1$ và $L_2$:**
   - Áp dụng Ridge ($L_2$) và Lasso ($L_1$) lên mô hình đa thức bậc 9.
   - Khảo sát sự suy giảm biên độ của vector trọng số $\mathbf{w}$ theo siêu tham số $\lambda$.
   - Chứng minh thực nghiệm tính chất tạo độ thưa (Sparsity) của $L_1$ (các trọng số dư thừa bị ép về chính xác 0.0).
4. **Thành phần 4 - Cài đặt Kỹ thuật K-Fold Cross-Validation và Đánh giá Phân loại:**
   - Tự viết hàm K-Fold Cross-Validation từ đầu bằng NumPy.
   - Tính toán đầy đủ Confusion Matrix, Precision, Recall, $F_1$-Score trên tập dữ liệu mất cân bằng nhãn.
