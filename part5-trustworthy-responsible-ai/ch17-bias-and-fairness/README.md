# Chương 17: Thiên kiến và Công bằng trong Học máy (Bias & Fairness in AI)

## Mục tiêu chương học
1. **Hiểu bản chất của nghịch lý công bằng**: Phân tích vì sao khái niệm "công bằng" không thể quy về một công thức toán học duy nhất mà luôn chứa đựng mâu thuẫn xã hội và triết học.
2. **Nhận diện toàn diện các nguồn gốc thiên kiến**: Phân tích thiên kiến dữ liệu (lịch sử, lấy mẫu, đại diện), thiên kiến gán nhãn, và thiên kiến hàm mục tiêu thiết kế.
3. **Làm chủ các thước đo công bằng toán học**: Định nghĩa và dẫn xuất chặt chẽ các chỉ số: Demographic Parity, Disparate Impact (luật $80\%$), Equal Opportunity, Equalized Odds, Predictive Parity, và Individual / Counterfactual Fairness.
4. **Nắm vững Định lý Bất khả thi về Công bằng (Impossibility Theorem)**: Chứng minh mâu thuẫn toán học không thể dung hòa giữa các tiêu chuẩn công bằng khi tỷ lệ cơ sở (base rates) giữa các nhóm có sự chênh lệch.
5. **Thành thạo các chiến lược giảm thiểu thiên kiến**: Nắm vững kỹ thuật Tiền xử lý (Re-weighing), Trong huấn luyện (Adversarial Debiasing), và Hậu xử lý (Threshold-Moving).
6. **Tích hợp yếu tố con người và quản trị**: Vai trò của Human-in-the-loop và quy trình kiểm toán độc lập.

---

## 1. Vì sao bài toán công bằng lại khó?

Trí tuệ nhân tạo không tự sinh ra từ hư không; nó phản chiếu trực tiếp thế giới thông qua dữ liệu mà nó được huấn luyện. Khi các hệ thống máy học được triển khai để tự động hóa các quyết định phân bổ nguồn lực xã hội (tuyển dụng nhân sự, xét hạn mức tín dụng, tính điểm thi đại học, dự báo nguy cơ tái phạm tội), mô hình có xu hướng:
1. **Khuếch đại định kiến lịch sử**: Thay vì đưa ra quyết định khách quan, mô hình học cách tái tạo và thậm chí làm trầm trọng hơn những định kiến xã hội vốn có trong dữ liệu quá khứ.
2. **Bẫy "mù thuộc tính nhạy cảm" (Fairness through Unawareness)**: Việc đơn thuần loại bỏ cột Giới tính hay Chủng tộc khỏi bảng dữ liệu hoàn toàn không giải quyết được vấn đề, vì các thuộc tính gián tiếp (Proxy features) như Mã bưu chính, Trường học, Sở thích có thể khôi phục lại gần như hoàn hảo thuộc tính nhạy cảm đó.
3. **Mâu thuẫn giá trị**: Khái niệm công bằng phụ thuộc vào góc nhìn triết học: Bình đẳng về kết quả đầu ra (Equality of Outcome) hay Bình đẳng về cơ hội cạnh tranh (Equality of Opportunity)?

---

## 2. Nguồn gốc của thiên kiến trong học máy (Sources of Bias)

```
                       CÁC NGUỒN GỐC THIÊN KIẾN
                                  |
         +------------------------+------------------------+
         |                        |                        |
  THIÊN KIẾN DỮ LIỆU       THIÊN KIẾN GÁN NHÃN      THIÊN KIẾN THIẾT KẾ
  +-- Lịch sử (Historical) +-- Đo lường chủ quan    +-- Lựa chọn hàm mất mát
  +-- Lấy mẫu (Sampling)   +-- Thiếu chuyên môn     +-- Biến đại diện (Proxy)
  +-- Thiếu đại diện       +-- Định kiến xã hội     +-- Tối ưu hóa trung bình
```

### 2.1. Thiên kiến dữ liệu (Data Bias)
- **Thiên kiến lịch sử (Historical Bias)**: Dữ liệu phản ánh chính xác thế giới quá khứ, nhưng quá khứ đó chứa đựng bất công (ví dụ: dữ liệu tuyển dụng 20 năm qua trong ngành công nghệ có $85\%$ nam giới).
- **Thiên kiến lấy mẫu (Sampling Bias)**: Dữ liệu thu thập không đại diện đúng cho phân phối thực tế của toàn bộ dân số (ví dụ: hệ thống nhận diện khuôn mặt huấn luyện trên $80\%$ ảnh người da trắng).
- **Thiên kiến đại diện (Representation Bias)**: Số lượng mẫu của nhóm thiểu số quá ít, khiến thuật toán học máy (vốn tối ưu hóa sai số trung bình) sẵn sàng "hy sinh" độ chính xác trên nhóm này để đổi lấy độ chính xác cao trên nhóm đa số.

### 2.2. Thiên kiến gán nhãn (Measurement & Label Bias)
Nhãn mục tiêu $Y$ được gán bởi con người thường mang tính thiên vị tiềm thức. Ví dụ: trong bài toán tư pháp, nhãn "có hành vi phạm tội" thường bị thay thế bằng biến số đo lường "bị cảnh sát bắt giữ" (Arrest rate) — vốn bị ảnh hưởng nặng nề bởi tần suất tuần tra dày đặc hơn ở các khu dân cư nghèo.

### 2.3. Thiên kiến mục tiêu và thiết kế (Objective & Design Bias)
Việc sử dụng các hàm mất mát truyền thống như Cross-Entropy hoặc Mean Squared Error trên toàn bộ tập dữ liệu đối xử bình đẳng với mỗi điểm dữ liệu đơn lẻ. Do đó, một mô hình đạt độ chính xác $90\%$ tổng thể hoàn toàn có thể đạt $99\%$ trên nhóm đa số nhưng chỉ đạt $50\%$ (đoán mò) trên nhóm yếu thế.

---

## 3. Các tiêu chuẩn công bằng toán học (Fairness Criteria)

Ký hiệu quy ước:
- $X \in \mathbb{R}^d$: Vector đặc trưng thông thường.
- $A \in \{0, 1\}$: Thuộc tính nhạy cảm (ví dụ $A=0$: Nhóm yếu thế / thiểu số, $A=1$: Nhóm đa số).
- $Y \in \{0, 1\}$: Nhãn thực tế (1: Tích cực - được nhận việc/duyệt vay, 0: Tiêu cực).
- $\hat{Y} \in \{0, 1\}$: Quyết định dự đoán của mô hình.
- $R = f(X, A) \in [0, 1]$: Điểm số rủi ro hoặc xác suất dự đoán liên tục.

---

### 3.1. Tiêu chuẩn công bằng nhóm (Group Fairness)

#### 3.1.1. Bình đẳng nhân khẩu học (Demographic Parity / Statistical Parity)
Tỷ lệ nhận được kết quả tích cực phải độc lập hoàn toàn với thuộc tính nhạy cảm $A$:
$$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$

- **Chỉ số Disparate Impact (Quy tắc $80\%$ của Ủy ban Cơ hội Việc làm Bình đẳng Mỹ - EEOC)**:
  $$\text{DI} = \frac{P(\hat{Y} = 1 \mid A = 0)}{P(\hat{Y} = 1 \mid A = 1)} \ge 0.8$$
  Nếu $\text{DI} < 0.8$, hệ thống bị coi là có sự phân biệt đối xử bất hợp pháp trên thực tế.

#### 3.1.2. Bình đẳng cơ hội (Equal Opportunity - Hardt et al., 2016)
Yêu cầu tỷ lệ dương tính thật (True Positive Rate - TPR) phải bằng nhau giữa các nhóm đối với những cá nhân thực sự đủ điều kiện ($Y=1$):
$$P(\hat{Y} = 1 \mid A = 0, Y = 1) = P(\hat{Y} = 1 \mid A = 1, Y = 1)$$
Nghĩa là: Nếu một ứng viên thực sự có năng lực, cơ hội được tuyển dụng của họ không bị phụ thuộc vào giới tính hay chủng tộc.

#### 3.1.3. Cân bằng tỷ lệ cược (Equalized Odds)
Yêu cầu cả Tỷ lệ dương tính thật (TPR) và Tỷ lệ dương tính giả (FPR) đều phải bằng nhau giữa hai nhóm:
$$P(\hat{Y} = 1 \mid A = 0, Y = y) = P(\hat{Y} = 1 \mid A = 1, Y = y), \quad \forall y \in \{0, 1\}$$
Đồng nghĩa với việc bộ phân loại $\hat{Y}$ độc lập có điều kiện với $A$ khi biết nhãn thực tế $Y$: $\hat{Y} \perp A \mid Y$.

#### 3.1.4. Tính tương đương tiên đoán (Predictive Parity / Sufficiency)
Yêu cầu giá trị tiên đoán dương (Precision) phải đồng nhất giữa các nhóm:
$$P(Y = 1 \mid A = 0, \hat{Y} = 1) = P(Y = 1 \mid A = 1, \hat{Y} = 1)$$
Nghĩa là: Khi mô hình đã dự đoán một người là "có rủi ro vỡ nợ cao", xác suất họ thực sự vỡ nợ là như nhau bất kể họ thuộc nhóm nhân khẩu học nào.

---

### 3.2. Tiêu chuẩn công bằng cá nhân (Individual Fairness - Dwork et al., 2012)
Nguyên tắc: "Những cá nhân tương tự nhau phải nhận được đối xử tương tự nhau".
Định nghĩa qua điều kiện liên tục Lipschitz:
$$D(f(\mathbf{x}_1), f(\mathbf{x}_2)) \le L \cdot d(\mathbf{x}_1, \mathbf{x}_2)$$
Trong đó $d(\cdot, \cdot)$ là hàm khoảng cách công bằng (Fair Metric) được chuyên gia xã hội học định nghĩa để đo lường độ tương đồng giữa hai cá nhân bỏ qua các đặc tính nhạy cảm.

### 3.3. Tiêu chuẩn công bằng nhân quả (Counterfactual Fairness - Kusner et al., 2017)
Một mô hình đạt chuẩn công bằng phản thực tế nếu đối với một cá nhân cụ thể với các đặc trưng $X = \mathbf{x}$ và $A = a$, quyết định của mô hình sẽ không thay đổi nếu trong một thế giới song song phản thực tế, thuộc tính nhạy cảm của người đó được hoán đổi thành $A = a'$ trong khi mọi yếu tố nhân quả không phụ thuộc vào $A$ được giữ nguyên:
$$P(\hat{Y}_{A \leftarrow a} = 1 \mid X = \mathbf{x}, A = a) = P(\hat{Y}_{A \leftarrow a'} = 1 \mid X = \mathbf{x}, A = a)$$

---

## 4. Định lý bất khả thi về công bằng (Impossibility Theorem of Fairness)

Được chứng minh độc lập bởi Kleinberg et al. (2016) và Chouldechova (2017), định lý toán học này đặt dấu chấm hết cho tham vọng tạo ra một hệ thống "công bằng tuyệt đối về mọi mặt".

### Phát biểu định lý:
Giả sử tỷ lệ nền cơ sở (Base Rate) giữa hai nhóm là khác nhau:
$$P(Y = 1 \mid A = 0) \ne P(Y = 1 \mid A = 1)$$
Và bộ phân loại không đạt độ chính xác hoàn hảo $100\%$.
Khi đó, **không thể đồng thời thỏa mãn cả ba tiêu chuẩn sau**:
1. **Bình đẳng nhân khẩu học (Demographic Parity)**.
2. **Cân bằng tỷ lệ cược (Equalized Odds)**.
3. **Tính tương đương tiên đoán (Predictive Parity)**.

> [!CAUTION]
> Bất kỳ quyết định tối ưu hóa cho một tiêu chuẩn công bằng này sẽ tất yếu dẫn tới việc vi phạm một tiêu chuẩn công bằng khác. Đây không phải là thiếu sót về mặt kỹ thuật thuật toán, mà là một định lý toán học bất khả xâm phạm. Lựa chọn tiêu chuẩn nào là một quyết định chính sách và đạo đức của con người.

---

## 5. Các chiến lược kỹ thuật giảm thiểu thiên kiến (Mitigation Strategies)

```
                     CHIẾN LƯỢC GIẢM THIỂU THIÊN KIẾN
                                     |
         +---------------------------+---------------------------+
         |                                                       |
  TIỀN XỬ LÝ (PRE-PROCESSING)                             HẬU XỬ LÝ (POST-PROCESSING)
  +-- Re-weighing (Gán trọng số)                          +-- Threshold-moving theo nhóm
  +-- Disparate Impact Remover                            +-- Reject Option Classification
  +-- Tái lấy mẫu (Re-sampling)                                  |
                                                                 v
                                                 TRONG HUẤN LUYỆN (IN-PROCESSING)
                                                 +-- Adversarial Debiasing (Mạng đối kháng)
                                                 +-- Ràng buộc Lagrangian tối ưu hóa
```

### 5.1. Kỹ thuật tiền xử lý: Tái gán trọng số (Re-weighing - Kamiran & Calders, 2012)
Điều chỉnh trọng số của từng mẫu trong tập huấn luyện sao cho thuộc tính nhạy cảm $A$ độc lập với nhãn thực tế $Y$:
$$W(x) = \frac{P(A = a) \cdot P(Y = y)}{P(A = a, Y = y)}$$
Bằng cách gán trọng số lớn hơn cho các mẫu thuộc nhóm thiểu số có nhãn tích cực $(A=0, Y=1)$, mô hình bị phạt nặng hơn nếu dự đoán sai trên nhóm này.

### 5.2. Kỹ thuật trong huấn luyện: Huấn luyện đối kháng loại bỏ thiên kiến (Adversarial Debiasing - Zhang et al., 2018)
Mạng gồm 2 thành phần:
1. **Mô hình dự đoán chính (Predictor)**: Dự đoán nhãn $\hat{Y}$ từ $X$.
2. **Mô hình đối kháng (Adversary)**: Cố gắng dự đoán thuộc tính nhạy cảm $A$ từ biểu diễn tầng ẩn hoặc xác suất $\hat{Y}$.
Predictor được tối ưu hóa đồng thời để tối đa hóa độ chính xác dự đoán nhãn $Y$, đồng thời cực tiểu hóa khả năng suy đoán thuộc tính $A$ của Adversary.

### 5.3. Kỹ thuật hậu xử lý: Dịch chuyển ngưỡng quyết định (Threshold-Moving - Hardt et al., 2016)
Thay vì áp dụng cùng một ngưỡng cắt cố định $\tau = 0.5$ cho tất cả mọi người, mô hình giữ nguyên điểm số dự đoán $R = f(X)$, nhưng thiết lập hai ngưỡng quyết định riêng biệt $\tau_0$ và $\tau_1$ cho hai nhóm:
$$\hat{Y} = \begin{cases} 1, & \text{nếu } (A = 0 \land R \ge \tau_0) \lor (A = 1 \land R \ge \tau_1) \\ 0, & \text{ngược lại} \end{cases}$$
Các ngưỡng $(\tau_0, \tau_1)$ được giải bằng bài toán quy hoạch tuyến tính trên đường cong ROC để thỏa mãn chính xác tiêu chuẩn Equal Opportunity hoặc Equalized Odds.

---

## 6. Yếu tố con người và Quản trị đạo đức (Human Factors)
- **Con người trong vòng lặp (Human-in-the-loop - HITL)**: Đối với các quyết định rủi ro cao (ví dụ: từ chối bảo hiểm y tế, sa thải nhân viên), AI chỉ đóng vai trò khuyến nghị; quyền quyết định cuối cùng và trách nhiệm giải trình phải thuộc về chuyên gia con người.
- **Tính đa dạng trong đội ngũ nghiên cứu**: Đội ngũ phát triển AI thiếu đa dạng (về giới, sắc tộc, vùng miền) là nguyên nhân hàng đầu khiến các lỗ hổng thiên kiến không được phát hiện từ giai đoạn thiết kế sản phẩm.

---

## 7. Tài liệu tham khảo chính thống
1. **Barocas, S., Hardt, M., & Narayanan, A. (2023)**. *Fairness and Machine Learning: Limitations and Opportunities*, MIT Press.
2. **Hardt, M., Price, E., & Srebro, N. (2016)**. *Equality of Opportunity in Supervised Learning*. NeurIPS 2016.
3. **Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016)**. *Inherent Trade-Offs in the Fair Determination of Risk Scores*. arXiv:1609.05807.
4. **Dwork, C., et al. (2012)**. *Fairness Through Awareness*. Innovations in Theoretical Computer Science (ITCS 2012).
5. **Chouldechova, A. (2017)**. *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*. Big Data 5(2).
6. **Zhang, B. H., Lemoine, B., & Mitchell, M. (2018)**. *Mitigating Unwanted Biases with Adversarial Learning*. AIES 2018.

---

## 8. Định hướng thực hành (`demo.ipynb`)
Trong `demo.ipynb`, chúng ta sẽ trực tiếp xây dựng:
1. **Bộ công cụ tính toán các chỉ số công bằng (Fairness Metrics Suite)**: Lập trình từ đầu các hàm tính Demographic Parity, Disparate Impact Ratio, Equal Opportunity Difference, và Equalized Odds trên một tập dữ liệu phân loại tín dụng nhân tạo có chứa thiên kiến lịch sử.
2. **Kỹ thuật tiền xử lý Re-weighing**: Tính toán trọng số hiệu chỉnh $W(A, Y)$ để cân bằng lại phân phối và đánh giá sự thay đổi của Disparate Impact.
3. **Kỹ thuật hậu xử lý Threshold-Moving**: Tối ưu hóa cặp ngưỡng quyết định riêng biệt $(\tau_0, \tau_1)$ trên đường cong ROC để triệt tiêu hoàn toàn chênh lệch True Positive Rate giữa hai nhóm nhân khẩu học.
