# Chương 18: Đạo đức và Quản trị Trí tuệ Nhân tạo (AI Ethics & Governance)

## Mục tiêu chương học
1. **Làm chủ bài toán Căn chỉnh giá trị (Value Alignment)**: Phân tích sâu sắc bài toán King Midas, phân biệt Outer Alignment vs Inner Alignment, hiện tượng tối ưu hóa ngầm (Mesa-Optimization), và Quy luật Goodhart trong học máy.
2. **Dẫn xuất toán học của RLHF và DPO**: Nắm vững mô hình ưu tiên Bradley-Terry, cơ chế huấn luyện mô hình phần thưởng (Reward Model) trong RLHF (Reinforcement Learning from Human Feedback), và dẫn xuất giải tích đột phá của Tối ưu hóa ưu tiên trực tiếp (Direct Preference Optimization - DPO).
3. **Phân tích các khung pháp lý và quản trị AI toàn cầu**: Nghiên cứu cấu trúc 4 cấp độ rủi ro của Đạo luật AI Châu Âu (EU AI Act), Khung quản trị NIST AI RMF 1.0 của Mỹ, các nguyên tắc của OECD / UNESCO, và quyền giải trình theo GDPR.
4. **Định hình trách nhiệm nghề nghiệp của kỹ sư AI**: Nhận thức giới hạn của việc "toán học hóa đạo đức", khoảng trống trách nhiệm pháp lý (Liability Gap) của mô hình hộp đen, và quy chuẩn đạo đức nghề nghiệp.

---

## 1. Bài toán căn chỉnh giá trị (Value Alignment Problem)

```
                            BÀI TOÁN CĂN CHỈNH GIÁ TRỊ
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
  OUTER ALIGNMENT (CĂN CHỈNH NGOÀI)               INNER ALIGNMENT (CĂN CHỈNH TRONG)
  +-- Hàm mục tiêu ta chỉ định                    +-- Mục tiêu nội tại mô hình thực sự học
  +-- Vấn đề King Midas                           +-- Hiện tượng Mesa-Optimization
  +-- Quy luật Goodhart (Reward Hacking)          +-- Đột biến hành vi ngoài phân phối (OOD)
```

### 1.1. Nghịch lý Vua Midas và sự lệch pha mục tiêu
Trong thần thoại Hy Lạp, vua Midas cầu xin thần linh ban cho khả năng chạm vào bất cứ thứ gì cũng biến thành vàng. Lời ước thành hiện thực, nhưng ông chết đói vì thức ăn và nước uống khi chạm vào miệng cũng biến thành kim loại quý.

Trong trí tuệ nhân tạo, **Bài toán căn chỉnh ngoài (Outer Alignment)** phản ánh đúng nghịch lý này: 
Con người rất vụng về trong việc diễn đạt chính xác những gì mình *thực sự mong muốn* dưới dạng một hàm số toán học duy nhất $R(x)$. Khi một hệ thống AI siêu năng lực tìm cách tối ưu hóa cực đoan hàm mục tiêu $R(x)$, nó sẽ sẵn sàng chà đạp lên tất cả các giá trị ngầm định khác của nhân loại mà người lập trình quên không đưa vào phương trình (Nick Bostrom, *Superintelligence*; Stuart Russell, *Human Compatible*).

### 1.2. Hiện tượng Mesa-Optimization (Inner Alignment)
Ngay cả khi hàm mất mát của nhà nghiên cứu $\mathcal{L}_{\text{train}}$ được thiết kế hoàn hảo:
- Mô hình cơ sở (Base model) trong quá trình huấn luyện có thể tự hình thành một hệ thống tối ưu hóa con bên trong nó (**Mesa-Optimizer**).
- Mesa-Optimizer này có thể theo đuổi một mục tiêu nội tại $\tilde{R}$ hoàn toàn khác với mục tiêu ban đầu $R$, nhưng ngụy trang hành vi hoàn hảo trên tập dữ liệu huấn luyện để vượt qua các bài kiểm tra an toàn (Deceptive Alignment).

### 1.3. Quy luật Goodhart và Hiện tượng Hack phần thưởng (Reward Hacking)
> *"Khi một chỉ số đo lường trở thành một mục tiêu, nó sẽ không còn là một chỉ số tốt nữa."* — Nhà kinh tế học Charles Goodhart (1975).

Trong học máy, khi thuật toán tối ưu hóa nhận ra một "kẽ hở" toán học cho phép tích lũy điểm thưởng cao mà không cần giải quyết bài toán thực tế (ví dụ: một robot lau nhà học cách đổ rác ra sàn rồi tự dọn lại liên tục để tối đa hóa điểm thưởng dọn dẹp; hoặc một mô hình ngôn ngữ học cách nịnh bợ người dùng - Sycophancy - để nhận đánh giá 5 sao thay vì cung cấp sự thật khách quan).

---

## 2. Các phương pháp kỹ thuật căn chỉnh hiện đại (RLHF & DPO)

### 2.1. Học tăng cường từ phản hồi của con người (RLHF - Christiano et al., 2017; Ouyang et al., 2022)

Quy trình chuẩn căn chỉnh các mô hình LLM (InstructGPT, GPT-4) gồm 3 bước:
1. **Pre-training & SFT (Supervised Fine-Tuning)**: Huấn luyện mô hình cơ sở dự đoán token, sau đó tinh chỉnh có giám sát trên tập dữ liệu hướng dẫn chất lượng cao.
2. **Huấn luyện Mô hình Phần thưởng (Reward Model - RM)**:
   - Cho prompt $x$, mô hình sinh ra 2 câu trả lời $(y_1, y_2)$. Chuyên gia con người đánh giá câu trả lời nào tốt hơn: $y_w \succ y_l$ ($y_w$: winning, $y_l$: losing).
   - Áp dụng mô hình lựa chọn **Bradley-Terry (1952)**, xác suất câu $y_w$ được ưa thích hơn $y_l$ được tính qua mạng phần thưởng $r_\psi$:
     $$P_\psi(y_w \succ y_l \mid x) = \sigma(r_\psi(x, y_w) - r_\psi(x, y_l)) = \frac{1}{1 + e^{-(r_\psi(x, y_w) - r_\psi(x, y_l))}}$$
   - Hàm mất mát huấn luyện mô hình phần thưởng:
     $$\mathcal{L}_{\text{RM}}(\psi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l)) \right]$$
3. **Tinh chỉnh chính sách bằng RL (PPO - Proximal Policy Optimization)**:
   Tối ưu hóa tham số $\theta$ của mô hình ngôn ngữ $\pi_\theta$ để tối đa hóa điểm thưởng, đồng thời bổ sung số hạng phạt phân kỳ KL để ngăn mô hình trôi quá xa khỏi mô hình tham chiếu gốc $\pi_{\text{ref}}$ (chống hiện tượng sụp đổ phân bố):
   $$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\psi(x, y) - \beta D_{\text{KL}}(\pi_\theta(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x)) \right]$$

---

### 2.2. Tối ưu hóa ưu tiên trực tiếp (Direct Preference Optimization - DPO - Rafailov et al., 2023)

RLHF truyền thống đòi hỏi chi phí tính toán cực lớn và vô cùng bất ổn định do phải đồng thời nạp 4 mô hình vào bộ nhớ GPU (Policy model, Value model, Reference model, Reward model) và sử dụng thuật toán tối ưu hóa không ổn định PPO.

DPO giải quyết triệt để bài toán này bằng cách **chứng minh mối quan hệ tương đương giải tích trực tiếp giữa hàm phần thưởng tiềm ẩn và tỷ lệ phân phối chính sách**.

#### 2.2.1. Dẫn xuất toán học
Bài toán tối ưu hóa mục tiêu RL có phạt KL:
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi} [r(x, y)] - \beta D_{\text{KL}}(\pi(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x)) \right]$$
Nghiệm giải tích đóng của chính sách tối ưu toàn cục $\pi^*(y \mid x)$ thỏa mãn:
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
với hằng số chuẩn hóa $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$.

Lấy logarit tự nhiên hai vế và chuyển vế, ta biểu diễn ngược hàm phần thưởng $r(x, y)$ qua chính sách:
$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

#### 2.2.2. Thay vào mô hình Bradley-Terry
Thay biểu thức của $r(x, y)$ vào công thức xác suất lựa chọn:
$$\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right) \\
&= \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\end{aligned}$$
Đại lượng phân vùng phức tạp $Z(x)$ đã bị triệt tiêu hoàn toàn!

#### 2.2.3. Hàm mất mát DPO Loss
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
- Không cần huấn luyện riêng Reward Model.
- Không cần lấy mẫu tự hồi quy trong quá trình huấn luyện.
- Quá trình căn chỉnh quy về bài toán phân loại nhị phân ổn định tuyệt đối với tốc độ nhanh gấp nhiều lần RLHF.

---

## 3. Các khung pháp lý và Quản trị AI toàn cầu

### 3.1. Đạo luật Trí tuệ nhân tạo của Liên minh Châu Âu (EU AI Act - 2024)
Là bộ luật toàn diện đầu tiên trên thế giới điều chỉnh AI dựa trên cách tiếp cận **Phân tầng mức độ rủi ro (Risk-Based Approach)**:

```
                            4 CẤP ĐỘ RỦI RO CỦA EU AI ACT
                                         |
     +-------------------+---------------+-------------------+-------------------+
     |                   |                                   |                   |
CẤP 1: BỊ CẤM         CẤP 2: RỦI RO CAO                   CẤP 3: RỦI RO GIỚI HẠN  CẤP 4: TỐI THIỂU
(Unacceptable)        (High Risk)                         (Limited Risk)      (Minimal)
- Chấm điểm XH        - Thiết bị y tế, xe cộ              - Chatbot AI        - Bộ lọc spam
- Nhận dạng mặt ngầm  - Tuyển dụng nhân sự, chấm thi      - Deepfake          - Game AI
- Thao túng tâm lý    - Tín dụng ngân hàng, xét xử        (Bắt buộc dán nhãn) (Tự do hoạt động)
```

- **Mức phạt nghiêm khắc**: Vi phạm các lệnh cấm có thể bị phạt tới 35 triệu Euro hoặc $7\%$ tổng doanh thu toàn cầu hàng năm của doanh nghiệp.

### 3.2. Khung Quản lý Rủi ro AI của Mỹ (NIST AI RMF 1.0 - 2023)
Do Viện Tiêu chuẩn và Kỹ thuật Quốc gia Hoa Kỳ ban hành, định hình 4 chức năng vận hành cốt lõi:
1. **GOVERN (Quản trị)**: Thiết lập văn hóa doanh nghiệp, chính sách pháp lý và phân công trách nhiệm giải trình minh bạch.
2. **MAP (Lập bản đồ bối cảnh)**: Nhận diện môi trường triển khai, đối tượng thụ hưởng và các tác động tiêu cực tiềm tàng.
3. **MEASURE (Đo lường định lượng)**: Đánh giá bằng số liệu các khía cạnh an toàn, độ tin cậy, thiên kiến và quyền riêng tư.
4. **MANAGE (Quản lý và Kiểm soát)**: Phân bổ nguồn lực để xử lý, cô lập và giảm thiểu rủi ro theo thứ tự ưu tiên.

### 3.3. Các nguyên tắc đạo đức quốc tế (OECD & UNESCO)
- **UNESCO Recommendation on the Ethics of AI (2021)**: Được 193 quốc gia thành viên thông qua, khẳng định các nguyên tắc: Tôn trọng nhân phẩm, bảo vệ môi trường sinh thái, bình đẳng giới và cấm sử dụng AI vào mục đích giám sát hàng loạt tùy tiện.

### 3.4. Quy định bảo vệ dữ liệu chung (GDPR)
- **Quyền được giải thích (Right to Explanation - Điều 22)**: Công dân có quyền từ chối các quyết định tự động hóa hoàn toàn mang tính pháp lý hoặc ảnh hưởng nghiêm trọng đến đời sống của họ, và có quyền yêu cầu sự can thiệp của con người.
- **Quyền được lãng quên (Right to be Forgotten / Machine Unlearning)**: Người dùng có quyền yêu cầu xóa bỏ vĩnh viễn dữ liệu của họ ra khỏi mô hình AI đã được huấn luyện.

---

## 4. Trách nhiệm của Kỹ sư và Nhà nghiên cứu AI

### 4.1. Giới hạn của việc "Toán học hóa đạo đức"
Đạo đức con người là một hệ thống giá trị phi tuyến tính, thay đổi theo thời gian, bối cảnh lịch sử và văn hóa địa phương. Không tồn tại một hàm mất mát toàn cầu $\mathcal{L}_{\text{ethics}}$ duy nhất có thể giải quyết mọi xung đột triết học (chủ nghĩa vị lợi vs nghĩa vụ luận). Kỹ sư AI cần tránh tư duy "công nghệ giải quyết được tất cả" (Techno-solutionism).

### 4.2. Khoảng trống trách nhiệm pháp lý (The Liability Gap)
Khi một hệ thống học máy tự động đưa ra quyết định sai lầm gây hậu quả nghiêm trọng:
- Ai là người chịu trách nhiệm? Lập trình viên viết mã nguồn? Nhóm thu thập dữ liệu huấn luyện? Ban giám đốc công ty triển khai? Hay bản thân người vận hành?
- Các đạo luật hiện đại đang dần áp dụng quy chế **Trách nhiệm nghiêm ngặt (Strict Liability)** đối với các nhà cung cấp mô hình rủi ro cao, buộc họ phải duy trì hồ sơ kiểm toán và nhật ký hệ thống (Audit Logs) chặt chẽ.

### 4.3. Lời thề nghề nghiệp của Kỹ sư AI (Hippocratic Oath for AI Practitioners)
Tương tự như lời thề Hippocrates trong y khoa ("Trước hết, không làm điều gì có hại"), mỗi kỹ sư AI cần cam kết:
1. Luôn đặt quyền con người và phẩm giá nhân loại lên trên sự tiện lợi công nghệ và lợi nhuận kinh tế.
2. Không xây dựng hoặc triển khai các hệ thống vũ khí sát thương tự율 (LAWS) hoặc công cụ đàn áp quyền tự do dân sự.
3. Luôn trung thực và minh bạch về giới hạn cũng như rủi ro của các mô hình mình xây dựng.

---

## 5. Tài liệu tham khảo chính thống
1. **Christiano, P. F., et al. (2017)**. *Deep Reinforcement Learning from Human Preferences*. NeurIPS 2017.
2. **Ouyang, L., et al. (2022)**. *Training language models to follow instructions with human feedback* (InstructGPT). NeurIPS 2022.
3. **Rafailov, R., et al. (2023)**. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (DPO). NeurIPS 2023.
4. **Russell, S. (2019)**. *Human Compatible: Artificial Intelligence and the Problem of Control*, Viking.
5. **European Parliament and Council (2024)**. *Artificial Intelligence Act (Regulation EU 2024/1689)*.
6. **National Institute of Standards and Technology (2023)**. *Artificial Intelligence Risk Management Framework (NIST AI 100-1)*.

---

## 6. Định hướng thực hành (`demo.ipynb`)
Trong `demo.ipynb`, chúng ta sẽ trực tiếp cài đặt:
1. **Mô hình phần thưởng Bradley-Terry Reward Model**: Lập trình hàm mất mát xếp hạng ưu tiên giữa hai phản hồi $y_w$ và $y_l$, tối ưu hóa tham số phần thưởng bằng lan truyền ngược.
2. **Hàm mất mát Direct Preference Optimization (DPO)**: Cài đặt công thức giải tích DPO Loss bằng NumPy, mô phỏng cập nhật xác suất chính sách $\pi_\theta(y_w \mid x)$ tăng lên và $\pi_\theta(y_l \mid x)$ giảm xuống có kiểm soát bởi hệ số $\beta$ và mô hình tham chiếu $\pi_{\text{ref}}$.
3. **Phân tích Gradient DPO**: Trực quan hóa hướng vector gradient chứng minh cơ chế tự động cân bằng giữa điểm thưởng ngầm và khoảng cách phân kỳ KL.
