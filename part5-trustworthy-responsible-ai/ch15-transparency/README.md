# Chương 15: Tính minh bạch và Khả năng giải thích của AI (Transparency & Explainable AI - XAI)

## Mục tiêu chương học
1. **Thấu hiểu hệ sinh thái AI đáng tin cậy**: Nhận thức vị trí trung tâm của tính minh bạch (Transparency) bên cạnh tính cứng cáp (Robustness), tính công bằng (Fairness), và đạo đức (Ethics).
2. **Phân định rạch ròi các khái niệm**: Phân biệt mô hình Hộp kính (Glass-box) và Hộp đen (Black-box); làm rõ sự khác biệt bản chất giữa Khả năng diễn giải nội tại (Interpretability) và Khả năng giải thích hậu nghiệm (Explainability).
3. **Phân loại các phương pháp XAI**: Nắm vững trục phân loại Toàn cục (Global) vs Cục bộ (Local); Độc lập mô hình (Model-agnostic) vs Phụ thuộc mô hình (Model-specific); Hậu nghiệm (Post-hoc) vs Bản chất (Ante-hoc).
4. **Làm chủ các thuật toán cốt lõi**: Nắm vững nền tảng toán học và dẫn xuất chi tiết của LIME, SHAP (dựa trên Lý thuyết trò chơi hợp tác của Lloyd Shapley), Bản đồ nổi bật (Saliency Maps), và Đạo hàm tích phân (Integrated Gradients).
5. **Khảo sát XAI nâng cao**: Nghiên cứu phương pháp giải thích phản thực tế (Counterfactuals), vector kích hoạt khái niệm (TCAV), và giải thích cơ học (Mechanistic Interpretability) trên các mô hình nơ-ron nền tảng.

---

## 1. Hệ sinh thái AI đáng tin cậy (Trustworthy AI)

Khi trí tuệ nhân tạo rời khỏi phòng thí nghiệm để đưa vào các quyết định có tính sống còn của con người (chẩn đoán ung thư, duyệt khoản vay tín dụng, xét ân xá tư pháp, điều khiển xe tự hành), độ chính xác danh nghĩa (Accuracy) trên tập kiểm thử không còn là tiêu chí duy nhất. 

Khung tiêu chuẩn AI Đáng tin cậy (NIST & Liên minh Châu Âu EU) được nâng đỡ bởi 4 trụ cột tương hỗ:
1. **Tính minh bạch và Khả năng giải thích (Transparency & Explainability)**: Con người phải hiểu được *tại sao* và *dựa trên cơ sở nào* mô hình đưa ra quyết định đó.
2. **Tính cứng cáp và Bảo mật (Robustness & Privacy)**: Mô hình phải ổn định trước nhiễu, chống chịu được tấn công đối kháng và không làm rò rỉ dữ liệu nhạy cảm.
3. **Tính công bằng và Không thiên kiến (Fairness & Bias-free)**: Quyết định không được phân biệt đối xử dựa trên các đặc tính nhạy cảm (chủng tộc, giới tính, tôn giáo).
4. **Trách nhiệm và Đạo đức (Accountability & Governance)**: Tuân thủ quy định pháp luật, có con người giám sát (Human-in-the-loop) và quy trách nhiệm rõ ràng khi hệ thống gặp sự cố.

---

## 2. Bản chất của Tính minh bạch

### 2.1. Tầm quan trọng trong các lĩnh vực rủi ro cao (High-Stakes Domains)
- **Y tế (Healthcare)**: Một mạng nơ-ron sâu chẩn đoán bệnh nhân bị u phổi ác tính không thể chỉ trả về con số $0.98$. Bác sĩ cần biết vùng tổn thương nào trên phim chụp X-quang dẫn đến kết luận đó để quyết định phẫu thuật.
- **Tài chính và Tín dụng (Finance)**: Các đạo luật bảo vệ người tiêu dùng (như Đạo luật Cơ hội Tín dụng Bình đẳng ECOA của Mỹ) yêu cầu tổ chức tài chính phải cung cấp lý do cụ thể từ chối hồ sơ vay của khách hàng.
- **Tư pháp hình sự (Criminal Justice)**: Các hệ thống đánh giá nguy cơ tái phạm (như COMPAS) nếu là hộp đen sẽ triệt tiêu quyền được xét xử công bằng của bị cáo.

### 2.2. Mô hình Hộp kính (Glass-box) vs Mô hình Hộp đen (Black-box)

| Đặc tính | Mô hình Hộp kính (Glass-Box / White-Box) | Mô hình Hộp đen (Black-Box) |
| :--- | :--- | :--- |
| **Kiến trúc tiêu biểu** | Hồi quy tuyến tính, Cây quyết định nông ($depth \le 4$), Generalized Additive Models (GAMs), Bảng luật (Rule lists). | Rừng ngẫu nhiên sâu, Gradient Boosted Trees ($1000$ cây), Deep CNNs, Transformers tỷ tham số. |
| **Tính minh bạch** | Minh bạch nội tại hoàn toàn (Simulatability / Decomposability). | Hoàn toàn mờ đục; các biểu diễn bị vướng víu phi tuyến qua hàng trăm tầng ẩn. |
| **Năng lực biểu diễn** | Giới hạn ở các mối quan hệ tuyến tính hoặc cộng tính đơn giản; dễ underfitting với dữ liệu phi cấu trúc. | Cực kỳ mạnh mẽ; xấp xỉ vạn năng mọi hàm phi tuyến phức tạp trong không gian đa chiều. |
| **Sự đánh đổi (Trade-off)** | *Hiệu năng thấp $\leftrightarrow$ Khả năng diễn giải tối đa*. | *Hiệu năng đỉnh cao $\leftrightarrow$ Độ mờ đục tối đa*. |

### 2.3. Khả năng diễn giải (Interpretability) vs Khả năng giải thích (Explainability)
- **Khả năng diễn giải (Interpretability / Ante-hoc)**: Bản thân mô hình được thiết kế có cấu trúc đủ đơn giản và tường minh để con người có thể tự lần theo từng bước tính toán từ đầu vào đến đầu ra mà không cần thêm bất kỳ công cụ phụ trợ nào.
- **Khả năng giải thích (Explainability / Post-hoc)**: Mô hình dự đoán ban đầu vẫn là một hộp đen cực kỳ phức tạp. Sau khi nó đưa ra kết quả, ta sử dụng một kỹ thuật phân tích độc lập (như LIME hoặc SHAP) để xấp xỉ và diễn giải hành vi của nó xung quanh điểm dự đoán.

---

## 3. Phân loại các phương pháp XAI

```
                              CÁC PHƯƠNG PHÁP XAI
                                      |
         +----------------------------+----------------------------+
         |                                                         |
  PHẠM VI GIẢI THÍCH                                     TÍNH ĐỘC LẬP MÔ HÌNH
  +-- Cục bộ (Local): Giải thích từng mẫu riêng lẻ       +-- Độc lập mô hình (Model-Agnostic): LIME, KernelSHAP
  +-- Toàn cục (Global): Giải thích toàn bộ hành vi       +-- Phụ thuộc mô hình (Model-Specific): TreeSHAP, Saliency
```

### 3.1. Cục bộ (Local) vs Toàn cục (Global)
- **Giải thích cục bộ (Local Explanation)**: Trả lời câu hỏi *"Tại sao khách hàng Nguyễn Văn A cụ thể này lại bị từ chối khoản vay vào thời điểm này?"*.
- **Giải thích toàn cục (Global Explanation)**: Trả lời câu hỏi *"Nhìn chung trên toàn bộ thị trường, mô hình đánh giá biến số nào là quan trọng nhất để cấp tín dụng?"*.

### 3.2. Độc lập mô hình (Model-Agnostic) vs Phụ thuộc mô hình (Model-Specific)
- **Model-Agnostic**: Xem mô hình gốc như một hàm hộp đen thuần túy $f(\mathbf{x})$. Kỹ thuật chỉ tương tác bằng cách đưa đầu vào $\mathbf{x}$ và nhận nhãn dự đoán $\hat{y}$. Áp dụng được cho bất kỳ thuật toán nào (SVM, Random Forest, Mạng nơ-ron).
- **Model-Specific**: Khai thác trực tiếp cấu trúc nội tại của mô hình (ví dụ: dùng ma trận trọng số và đồ thị tính toán đạo hàm của mạng nơ-ron, hoặc ngưỡng chia nhánh của cây quyết định).

---

## 4. Các thuật toán tiêu chuẩn (Core Standard Algorithms)

### 4.1. LIME: Local Interpretable Model-agnostic Explanations (Ribeiro et al., 2016)

#### 4.1.1. Trực giác hình học
Dù một hàm mục tiêu $f(\mathbf{x})$ trên toàn không gian là cực kỳ phi tuyến và phức tạp, nhưng trong một lân cận vi mô xung quanh một điểm cụ thể $\mathbf{x}$, mặt quyết định có thể xấp xỉ rất tốt bằng một siêu phẳng tuyến tính (tương tự như khai triển Taylor bậc 1).

#### 4.1.2. Công thức toán học
Ta tìm một mô hình giải thích đơn giản $g \in G$ (ví dụ: mô hình hồi quy tuyến tính thưa $g(\mathbf{z}') = w_0 + \mathbf{w}^T \mathbf{z}'$) bằng cách giải bài toán tối ưu hóa:
$$\xi(\mathbf{x}) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_{\mathbf{x}}) + \Omega(g)$$
Trong đó:
- $f$: Mô hình phức tạp ban đầu.
- $\mathbf{x}$: Mẫu dữ liệu gốc cần giải thích.
- $\mathbf{z}' \in \{0, 1\}^d$: Biểu diễn nhị phân đơn giản hóa (ví dụ: từ có xuất hiện hay không, siêu điểm ảnh super-pixel có bị che hay không).
- $\pi_{\mathbf{x}}(\mathbf{z})$: Nhân khoảng cách hàm mũ đo lường độ gần giữa điểm nhiễu $\mathbf{z}$ và điểm gốc $\mathbf{x}$:
  $$\pi_{\mathbf{x}}(\mathbf{z}) = \exp\left( - \frac{D(\mathbf{x}, \mathbf{z})^2}{\sigma^2} \right)$$
- $\mathcal{L}$: Hàm mất mát bình phương sai số có trọng số:
  $$\mathcal{L}(f, g, \pi_{\mathbf{x}}) = \sum_{\mathbf{z}, \mathbf{z}' \in \mathcal{Z}} \pi_{\mathbf{x}}(\mathbf{z}) \left( f(\mathbf{z}) - g(\mathbf{z}') \right)^2$$
- $\Omega(g)$: Số hạng phạt độ phức tạp (ví dụ: số lượng thuộc tính khác 0 $\|\mathbf{w}\|_0 \le K$).

---

### 4.2. SHAP: Shapley Additive exPlanations (Lundberg & Lee, 2017)

SHAP thống nhất các phương pháp giải thích dựa trên thuộc tính thông qua **Lý thuyết trò chơi hợp tác (Cooperative Game Theory)** do nhà kinh tế học đoạt giải Nobel Lloyd Shapley đề xuất (1953).

#### 4.2.1. Đặt bài toán
- **Trò chơi**: Quá trình mô hình dự đoán đầu ra cho điểm dữ liệu $\mathbf{x}$.
- **Người chơi (Players)**: Tập hợp $M$ thuộc tính đặc trưng $F = \{1, 2, \dots, M\}$.
- **Tổng tiền thưởng (Payout)**: Giá trị dự đoán $f(\mathbf{x}) - \mathbb{E}[f(X)]$.
- **Mục tiêu**: Phân chia tiền thưởng công bằng cho từng thuộc tính $\phi_i \in \mathbb{R}$ sao cho tổng đóng góp bằng đúng độ lệch dự đoán so với kỳ vọng nền:
  $$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^M \phi_i$$
  với $\phi_0 = \mathbb{E}[f(X)]$.

#### 4.2.2. Công thức giá trị Shapley (Shapley Value)
Đóng góp biên của thuộc tính $i$ được tính bằng trung bình trọng số trên tất cả các tập hợp con liên minh có thể $S \subseteq F \setminus \{i\}$:
$$\phi_i(\mathbf{x}) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! \, (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
Trong đó:
- $|S|$: Kích thước của liên minh hiện tại.
- $|F|!$: Tổng số thứ tự hoán vị có thể của tất cả các người chơi.
- $f(S)$: Giá trị dự đoán của mô hình khi chỉ có tập thuộc tính $S$ tham gia (các thuộc tính ngoài $S$ được tích phân hoặc thay thế bằng giá trị kỳ vọng từ phân phối biên).

#### 4.2.3. Bốn tiên đề duy nhất (Shapley Axioms)
Lloyd Shapley đã chứng minh rằng giá trị $\phi_i$ là cách phân chia **duy nhất** thỏa mãn đồng thời cả 4 tiên đề:
1. **Tính hiệu quả (Efficiency)**: $\sum_{i=1}^M \phi_i = f(\mathbf{x}) - \mathbb{E}[f(X)]$.
2. **Tính đối xứng (Symmetry)**: Nếu hai thuộc tính $i$ và $j$ có đóng góp biên bằng nhau trong mọi liên minh ($f(S \cup \{i\}) = f(S \cup \{j\}), \forall S \subseteq F \setminus \{i, j\}$) thì $\phi_i = \phi_j$.
3. **Người chơi bù nhìn (Dummy Player)**: Nếu thuộc tính $i$ không làm thay đổi dự đoán trong bất kỳ liên minh nào ($f(S \cup \{i\}) = f(S), \forall S$), thì $\phi_i = 0$.
4. **Tính cộng tính (Additivity)**: Nếu một mô hình là tổng của hai mô hình $f = f_1 + f_2$, thì $\phi_i(f) = \phi_i(f_1) + \phi_i(f_2)$.

---

### 4.3. Bản đồ nổi bật (Saliency Maps)
Áp dụng cho các mô hình khả vi như Mạng nơ-ron sâu trong xử lý ảnh (Simonyan et al., 2013).
Xét hàm dự đoán điểm số lớp $c$ là $S_c(\mathbf{x})$. Khai triển Taylor bậc 1 xung quanh ảnh đầu vào $\mathbf{x}_0$:
$$S_c(\mathbf{x}) \approx \mathbf{w}^T \mathbf{x} + b, \quad \mathbf{w} = \left. \frac{\partial S_c}{\partial \mathbf{x}} \right|_{\mathbf{x}_0}$$

Bản đồ nổi bật tại pixel $(i, j)$ là độ lớn của đạo hàm riêng theo từng kênh màu:
$$M_{ij} = \max_{c \in \{R, G, B\}} \left| \frac{\partial S_c}{\partial x_{ij}^{(c)}} \right|$$
Pixel nào có độ dốc (gradient) lớn nhất nghĩa là việc thay đổi giá trị của pixel đó sẽ tạo ra sự biến thiên mạnh mẽ nhất lên quyết định phân loại của mạng.

---

### 4.4. Đạo hàm tích phân (Integrated Gradients - Sundararajan et al., 2017)
Bản đồ nổi bật truyền thống vi phạm tiên đề **Completeness** do hiện tượng bão hòa gradient (khi hàm kích hoạt Sigmoid hoặc ReLU đạt trạng thái bão hòa, $\nabla_{\mathbf{x}} f \approx \mathbf{0}$ mặc dù đặc trưng đó cực kỳ quan trọng).

Integrated Gradients giải quyết triệt để vấn đề này bằng cách tích phân gradient dọc theo đoạn thẳng nối từ một điểm mốc chuẩn (baseline) $\mathbf{x}'$ (ví dụ: ảnh toàn màu đen) đến điểm dữ liệu thực tế $\mathbf{x}$:
$$\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{0}^1 \frac{\partial f(\mathbf{x}' + \alpha (\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

Theo Định lý cơ bản của Giải tích, tổng giá trị IG trên mọi thuộc tính bằng chính xác độ chênh lệch đầu ra:
$$\sum_{i=1}^M \text{IG}_i(\mathbf{x}) = f(\mathbf{x}) - f(\mathbf{x}')$$
Trong thực nghiệm tính toán, tích phân được xấp xỉ bằng tổng Riemann với $m$ bước vi phân (thường $m \in [50, 300]$):
$$\text{IG}_i^{\text{approx}}(\mathbf{x}) = (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^m \frac{\partial f\left( \mathbf{x}' + \frac{k}{m}(\mathbf{x} - \mathbf{x}') \right)}{\partial x_i}$$

---

## 5. Các phương pháp XAI nâng cao

### 5.1. Giải thích phản thực tế (Counterfactual Explanations - Wachter et al., 2017)
Tìm kiếm sự thay đổi tối thiểu lên đầu vào $\mathbf{x}$ sao cho mô hình thay đổi quyết định sang nhãn mong muốn $y^*$:
$$\mathbf{x}^* = \arg\min_{\mathbf{x}'} d(\mathbf{x}, \mathbf{x}') + \lambda \left( f(\mathbf{x}') - y^* \right)^2$$
Ví dụ dễ hành động cho người dùng: *"Nếu thu nhập hàng tháng của bạn tăng thêm 5 triệu đồng và điểm tín dụng tăng 30 điểm, khoản vay sẽ được phê duyệt."*

### 5.2. Vector kích hoạt khái niệm (Concept Activation Vectors - TCAV - Kim et al., 2018)
Thay vì giải thích ở mức pixel hay token đơn lẻ, TCAV cho phép người dùng đặt câu hỏi ở mức khái niệm trừu tượng của con người (Human Concepts).
- Cho một tập ảnh mang khái niệm $C$ (ví dụ: ảnh có hoa văn sọc vằn).
- Huấn luyện một siêu phẳng tuyến tính trong không gian kích hoạt tầng ẩn để tìm vector pháp tuyến $\mathbf{v}_C$.
- Tính đạo hàm có hướng (directional derivative) của dự đoán lớp theo vector $\mathbf{v}_C$.

### 5.3. Giải thích cơ học (Mechanistic Interpretability)
Nỗ lực kỹ nghệ đảo ngược (reverse-engineer) mạng nơ-ron nền tảng thành các mạch điện toán học (circuits) có thể hiểu được:
- Nghiên cứu cơ chế Induction Heads trong Transformer (nhận diện quy luật lặp từ ngữ cảnh).
- Phân tích hiện tượng đa nghĩa (Polysemanticity) của nơ-ron và sử dụng Sparse Autoencoders (SAE) để phân rã không gian ẩn thành các khái niệm trực giao rõ ràng.

---

## 6. Tài liệu tham khảo chính thống
1. **Bishop, C. M. (2006)**. *Pattern Recognition and Machine Learning*, Springer.
2. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016)**. *\"Why Should I Trust You?\": Explaining the Predictions of Any Classifier* (LIME). ACM KDD 2016.
3. **Lundberg, S. M., & Lee, S. I. (2017)**. *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS 2017.
4. **Sundararajan, M., Taly, A., & Yan, Q. (2017)**. *Axiomatic Attribution for Deep Networks* (Integrated Gradients). ICML 2017.
5. **Kim, B., et al. (2018)**. *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)*. ICML 2018.
6. **Wachter, S., Mittelstadt, B., & Russell, C. (2017)**. *Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR*. Harvard Journal of Law & Technology.

---

## 7. Định hướng thực hành (`demo.ipynb`)
Trong `demo.ipynb`, chúng ta sẽ trực tiếp xây dựng từ đầu 3 phương pháp giải thích cốt lõi:
1. **Cài đặt LIME từ con số không (LIME from scratch)**: Lấy mẫu nhiễu Gaussian quanh một điểm dữ liệu, gán trọng số khoảng cách bằng hàm nhân Exponential Kernel, huấn luyện một mô hình hồi quy tuyến tính cục bộ có trọng số (Weighted Ridge Regression), và trích xuất hệ số tầm quan trọng của thuộc tính.
2. **Tính toán giá trị Shapley giải tích**: Duyệt qua toàn bộ $2^M$ tập hợp con liên minh trên một bài toán mô hình đồ chơi, kiểm chứng chặt chẽ tính bảo toàn của 4 tiên đề Shapley (Efficiency, Symmetry, Dummy, Additivity).
3. **Integrated Gradients trên mô hình khả vi**: Cài đặt xấp xỉ tích phân Riemann $\frac{1}{m} \sum \nabla f$ dọc theo đường cơ sở, so sánh trực quan với Gradient thô (Saliency Map) và kiểm chứng tiên đề Completeness ($f(x) - f(x') = \sum \text{IG}_i$).
