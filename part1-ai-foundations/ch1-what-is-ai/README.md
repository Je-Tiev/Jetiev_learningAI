# Chương 1: Trí tuệ nhân tạo là gì? (What is Artificial Intelligence?)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Nắm vững định nghĩa và phạm vi của AI:** Hiểu rõ 4 cách tiếp cận kinh điển theo trường phái Russell & Norvig và phân biệt được AI với các ngành khoa học liên quan.
2. **Phân tích bản chất trí tuệ:** So sánh cơ chế xử lý thông tin giữa mạng nơ-ron sinh học của con người và kiến trúc tính toán máy tính số, giải thích sâu sắc Nghịch lý Moravec.
3. **Phân biệt hai hệ chuẩn nền tảng:** Đối chiếu chi tiết Trí tuệ nhân tạo ký hiệu (Symbolic AI) và Trí tuệ nhân tạo thống kê (Statistical AI), đồng thời hiểu động lực hình thành của AI lai (Neuro-Symbolic AI).
4. **Hệ thống hóa tiến trình lịch sử:** Nắm bắt dòng chảy phát triển của AI từ tiền đề triết học, các giai đoạn bùng nổ, hai thời kỳ "mùa đông AI", đến bước ngoặt của Deep Learning và Large Language Models.
5. **Định vị cấp độ trí tuệ:** Phân biệt ranh giới kỹ thuật giữa AI hẹp (ANI), AI tổng quát (AGI) và Siêu trí tuệ (ASI), cùng các thách thức cốt lõi về căn chỉnh giá trị (AI Alignment).
6. **Định hướng thực hành:** Hiểu nguyên lý so sánh giữa lập luận dựa trên luật (rule-based) và học máy thống kê để sẵn sàng thực hành trên `demo.ipynb`.

---

## 1.1 Định nghĩa và phạm vi của AI

### 1.1.1 Khái niệm Trí tuệ nhân tạo
Trí tuệ nhân tạo (Artificial Intelligence - AI) là ngành khoa học máy tính chuyên nghiên cứu, thiết kế các hệ thống tính toán có khả năng thực hiện các tác vụ vốn đòi hỏi năng lực trí tuệ của con người, bao gồm: tri giác (perception), học tập (learning), lập luận (reasoning), giải quyết vấn đề (problem solving), hiểu ngôn ngữ tự nhiên (natural language understanding) và tương tác với môi trường (action).

### 1.1.2 Bốn cách tiếp cận kinh điển (Russell & Norvig Framework)
Trong cuốn giáo trình kinh điển *Artificial Intelligence: A Modern Approach*, Stuart Russell và Peter Norvig phân loại các định nghĩa về AI theo hai chiều kích: **Tiêu chuẩn đo lường** (dựa trên con người vs. dựa trên tính hợp lý lý tưởng) và **Đối tượng nghiên cứu** (quá trình tư duy bên trong vs. hành vi quan sát bên ngoài).

| Chiều kích | **Mô phỏng Con người (Human-centric)** | **Dựa trên Tính hợp lý (Rationality)** |
| :--- | :--- | :--- |
| **Quá trình tư duy (Thought Process)** | **Suy nghĩ như con người (Thinking Humanly)**<br>*(Khoa học nhận thức - Cognitive Science)* | **Suy nghĩ hợp lý (Thinking Rationally)**<br>*(Quy luật tư duy & Logic hình thức)* |
| **Hành vi quan sát (Behavior)** | **Hành động như con người (Acting Humanly)**<br>*(Phép thử Turing - Turing Test)* | **Hành động hợp lý (Acting Rationally)**<br>*(Mô hình Tác tử hợp lý - Rational Agents)* |

#### 1. Hành động như con người (Acting Humanly) - Phép thử Turing (1950)
Alan Turing đề xuất "Trò chơi bắt chước" (The Imitation Game) để thay thế cho câu hỏi trừu tượng "Máy tính có thể tư duy hay không?". Một hệ thống máy tính vượt qua phép thử Turing nếu một người thẩm vấn con người qua giao tiếp văn bản không thể phân biệt được đối tượng trả lời là người hay máy.
Để vượt qua bài kiểm tra toàn diện, hệ thống cần tích hợp:
- **Xử lý ngôn ngữ tự nhiên (NLP):** Để giao tiếp trôi chảy.
- **Biểu diễn tri thức (Knowledge Representation):** Để lưu trữ thông tin tiếp nhận.
- **Suy luận tự động (Automated Reasoning):** Để trả lời câu hỏi và rút ra kết luận mới.
- **Học máy (Machine Learning):** Để thích nghi với hoàn cảnh mới và ngoại suy dữ liệu.
- *Phép thử Turing mở rộng (Total Turing Test)* bổ sung thêm **Thị giác máy tính (Computer Vision)** và **Khoa học robot (Robotics)** để tương tác thể chất với thế giới.

#### 2. Suy nghĩ như con người (Thinking Humanly) - Mô hình hóa nhận thức
Hướng tiếp cận này nhằm trả lời: Làm thế nào tâm trí con người vận hành? Để khẳng định một chương trình suy nghĩ như con người, ta cần giải mã cơ chế nội tại của não bộ thông qua:
- Phương pháp nội quan (Introspection): Tự quan sát dòng suy nghĩ của bản thân.
- Thí nghiệm tâm lý học (Psychological experiments): Quan sát hành vi cá nhân trong môi trường có kiểm soát.
- Kỹ thuật hình ảnh thần kinh (Neuroimaging): fMRI, EEG để quét hoạt động não bộ theo thời gian thực.
Hướng tiếp cận này hình thành nên ngành **Khoa học nhận thức (Cognitive Science)**, kết hợp giữa mô hình tính toán AI và các kỹ thuật thực nghiệm tâm lý học thần kinh.

#### 3. Suy nghĩ hợp lý (Thinking Rationally) - Quy luật tư duy (Laws of Thought)
Bắt nguồn từ các tam đoạn luận (Syllogisms) của Aristotle, hướng tiếp cận này tìm kiếm các cấu trúc lập luận luôn dẫn tới kết luận đúng đắn từ các tiền đề đúng:
$$\text{Tiền đề 1: } \forall x \, (\text{Human}(x) \to \text{Mortal}(x))$$
$$\text{Tiền đề 2: } \text{Human}(\text{Socrates})$$
$$\text{Kết luận: } \text{Mortal}(\text{Socrates})$$
Trường phái Logic này đặt nền móng cho logic vị từ hình thức, nhưng đối mặt với hai rào cản lớn: (1) Rất khó hình thức hóa tri thức đời thực mơ hồ thành logic chính xác; (2) Sự bùng nổ tổ hợp tính toán khi giải quyết các bài toán lớn.

#### 4. Hành động hợp lý (Acting Rationally) - Hướng tiếp cận Tác tử (Rational Agent)
Một **tác tử (agent)** là một thực thể nhận thức môi trường thông qua các cảm biến (sensors) và tác động lên môi trường đó thông qua các bộ phận chấp hành (actuators).
Một tác tử được định nghĩa là **hợp lý (rational)** nếu nó hành động sao cho cực đại hóa giá trị kỳ vọng của một thước đo hiệu năng (performance measure) dựa trên lịch sử nhận thức và kiến thức nền:
$$a^* = \arg\max_{a \in \mathcal{A}} \mathbb{E} \left[ U(s') \mid \mathbf{e}, a \right]$$
Trong đó:
- $a^*$ là hành động tối ưu được lựa chọn từ không gian hành động $\mathcal{A}$.
- $\mathbf{e}$ là chuỗi quan sát (evidence / percept sequence) mà cảm biến ghi nhận.
- $U(s')$ là hàm thỏa dụng (utility function) phản ánh mức độ thành công tại trạng thái tiếp theo $s'$.
- $\mathbb{E}[\cdot]$ là kỳ vọng toán học lấy trên phân phối xác suất chuyển trạng thái của môi trường $P(s' \mid s, a)$.

> **Nhận xét:** Hướng tiếp cận *Tác tử hợp lý* là nền tảng chủ đạo của AI hiện đại vì nó tổng quát hơn logic thuần túy (cho phép ra quyết định ngay cả khi không có giải pháp suy diễn chắc chắn đúng) và khả thi hơn về mặt toán học/kỹ thuật so với việc sao chép hoàn toàn tâm lý học con người.

### 1.1.3 Phạm vi và Mối quan hệ liên ngành của AI
AI nằm ở giao điểm của nhiều ngành khoa học:
- **Khoa học máy tính:** Hệ thống phân tán, kiến trúc phần cứng (GPU/TPU), cấu trúc dữ liệu và giải thuật tối ưu.
- **Toán học & Thống kê:** Đại số tuyến tính, giải tích đa biến, lý thuyết xác suất, lý thuyết thông tin, tối ưu hóa lồi.
- **Khoa học thần kinh (Neuroscience):** Cung cấp cảm hứng cho các kiến trúc mạng nơ-ron sâu.
- **Kinh tế học & Lý thuyết trò chơi:** Tối ưu hóa thỏa dụng, lý thuyết quyết định (Decision Theory), cân bằng Nash trong môi trường đa tác tử.
- **Ngôn ngữ học & Triết học:** Bản thể luận, ngữ nghĩa học, triết học tinh thần (Philosophy of Mind).

```
         ┌────────────────────────────────────────┐
         │       Trí tuệ nhân tạo (AI)            │
         │  ┌──────────────────────────────────┐  │
         │  │     Học máy (Machine Learning)   │  │
         │  │  ┌────────────────────────────┐  │  │
         │  │  │    Học sâu (Deep Learning) │  │  │
         │  │  │  ┌──────────────────────┐  │  │  │
         │  │  │  │  Foundation Models / │  │  │  │
         │  │  │  │  LLMs, GenAI         │  │  │  │
         │  │  │  └──────────────────────┘  │  │  │
         │  │  └────────────────────────────┘  │  │
         │  └──────────────────────────────────┘  │
         │   Symbolic AI, Search, Planning, Logic │
         └────────────────────────────────────────┘
```

---

## 1.2 Bản chất trí tuệ: Con người vs. Máy tính

### 1.2.1 Não bộ sinh học và Kiến trúc tính toán Von Neumann
Để hiểu được giới hạn và tiềm năng của AI, cần so sánh trực tiếp cơ chế xử lý của hệ thần kinh người với máy tính số hiện đại:

| Đặc tính | Não bộ sinh học (Human Brain) | Máy tính số (Von Neumann Computer) |
| :--- | :--- | :--- |
| **Đơn vị xử lý cơ bản** | $\sim 8.6 \times 10^{10}$ nơ-ron sinh học, $\sim 10^{14}$ khớp thần kinh (synapses) | Hàng tỷ bóng bán dẫn (transistors) trong CPU/GPU |
| **Tốc độ xung nhịp** | Rất chậm: $\sim 100 - 1000\text{ Hz}$ (thời gian lan truyền ion $\sim\text{ms}$) | Cực nhanh: $\sim 2 - 5\text{ GHz}$ ($10^9$ chu kỳ/giây) |
| **Phương thức xử lý** | Xử lý song song khổng lồ (massively parallel), phân tán, bất đồng bộ | Chủ yếu tuần tự hoặc song song khối (SIMD/GPU), đồng bộ xung clock |
| **Lưu trữ và Xử lý** | Đồng vị trí (tính toán và lưu trữ cùng diễn ra tại cấu trúc synapse) | Phân tách vật lý giữa CPU và Bộ nhớ RAM (Nút thắt cổ chai Von Neumann) |
| **Tiêu thụ năng lượng** | Cực kỳ tiết kiệm: $\sim 20\text{ Watts}$ | Rất lớn: Hàng trăm Watts (GPU cá nhân) đến MegaWatts (Data Center) |
| **Khả năng học tập** | Khái quát hóa mạnh mẽ từ rất ít mẫu (Few-shot learning), chuyển giao tri thức linh hoạt | Cần hàng triệu mẫu dữ liệu để hội tụ; nhạy cảm với dịch chuyển phân phối |
| **Độ ổn định lỗi** | Suy biến mềm (Graceful degradation): mất nơ-ron không làm hỏng toàn bộ hệ thống | Dễ đổ vỡ (Catastrophic failure): hỏng 1 bit nhớ có thể gây sập hệ điều hành |

### 1.2.2 Nghịch lý Moravec (Moravec's Paradox)
Năm 1988, Hans Moravec cùng Rodney Brooks và Marvin Minsky chỉ ra một nghịch lý cơ bản:
> *"Những tác vụ đòi hỏi tư duy logic trừu tượng cao cấp nhất của con người (đánh cờ, tính toán ma trận, giải bài toán tích phân) lại tiêu tốn rất ít tài nguyên tính toán của máy tính; ngược lại, những kỹ năng cảm giác - vận động bản năng của một đứa trẻ 3 tuổi (nhận diện khuôn mặt mẹ trong bóng tối, bước đi trên địa hình gồ ghề, cầm một quả trứng mà không làm vỡ) lại đòi hỏi năng lực tính toán khổng lồ và các giải thuật phức tạp nhất."*

**Lý giải tiến hóa:** Năng lực cảm giác - vận động của sinh vật đã trải qua hơn 1 tỷ năm tiến hóa và tối ưu hóa phần cứng sinh học để trở thành vô thức và tự nhiên. Trong khi đó, tư duy logic trừu tượng và ngôn ngữ mới chỉ xuất hiện khoảng 100.000 năm trở lại đây - một "phần mềm" mới được phát triển thêm, vốn chưa được tối ưu hóa sâu trong hệ sinh thái sinh học nhưng lại hoàn toàn trùng khớp với đại số Boolean của máy tính.

### 1.2.3 Lập luận Căn phòng Trung Hoa của John Searle (1980)
Triết gia John Searle đưa ra thí nghiệm tưởng tượng: Một người chỉ nói tiếng Anh ngồi trong phòng kín, nhận các ký tự chữ Hán qua khe cửa, tra cứu một cuốn sách quy tắc chi tiết bằng tiếng Anh (chỉ thao tác thuần túy cú pháp: nếu thấy ký tự $A$ thì đưa ra ký tự $B$), rồi đẩy kết quả ra ngoài. Người bên ngoài tưởng rằng người bên trong hiểu tiếng Trung.
- **Ý nghĩa:** Searle khẳng định các hệ thống tính toán kỹ thuật số chỉ thao tác trên **cú pháp (syntax)** mà không hề có sự thấu hiểu **ngữ nghĩa (semantics)** hay nhận thức chủ quan (Intentionality). Một mô hình học máy hay LLM có thể sinh văn bản hợp lý nhưng không nhất thiết "hiểu" theo nghĩa tâm lý học của con người.

---

## 1.3 Các hệ chuẩn: Ký hiệu và Thống kê

Lịch sử AI là cuộc đối thoại và cạnh tranh giữa hai hệ chuẩn (paradigms) căn bản: Ký hiệu (Symbolic) và Thống kê (Statistical).

### 1.3.1 Trí tuệ nhân tạo ký hiệu (Symbolic AI / GOFAI - Good Old-Fashioned AI)
- **Giả thuyết nền tảng:** *Giả thuyết Hệ thống Ký hiệu Vật lý (Physical Symbol System Hypothesis)* của Allen Newell và Herbert Simon (1976):
  > *"Một hệ thống ký hiệu vật lý có đủ các điều kiện cần và đủ để sinh ra hành vi thông minh tổng quát."*
- **Cơ chế hoạt động:** Tiếp cận theo hướng "Top-Down" (từ trên xuống). Các chuyên gia con người xây dựng các định nghĩa, quan hệ bản thể học (ontology) và các luật suy luận logic hình thức ($P \to Q$, $Q \to R \vdash P \to R$).
- **Ưu điểm:**
  1. Tính diễn giải tuyệt đối (White-box): Mọi quyết định đều có thể truy vết ngược qua cây dẫn xuất logic (derivation tree).
  2. Đòi hỏi 0 mẫu dữ liệu huấn luyện: Hoạt động ngay khi luật được nạp vào hệ thống.
  3. Tính chuẩn xác cao trong miền tri thức đóng: Rất mạnh trong chứng minh định lý toán học, kế toán, hệ thống định tuyến mạng.
- **Nhược điểm cốt tử:**
  1. **Nút thắt thu thập tri thức (Knowledge Acquisition Bottleneck):** Con người không thể phát biểu hết các tri thức ngầm (tacit knowledge) thành các luật tường minh.
  2. **Độ giòn (Brittleness):** Chỉ cần một tiền đề thiếu hoặc mâu thuẫn nhỏ, hệ suy diễn có thể rơi vào bế tắc hoặc suy ra kết luận sai vô lý ($P \land \neg P \vdash Q$).
  3. **Bùng nổ tổ hợp (Combinatorial Explosion):** Khi số lượng mệnh đề tăng, không gian tìm kiếm phát triển theo hàm mũ $\mathcal{O}(b^d)$.

### 1.3.2 Trí tuệ nhân tạo thống kê (Statistical AI / Machine Learning)
- **Cơ chế hoạt động:** Tiếp cận theo hướng "Bottom-Up" (từ dưới lên). Thay vì lập trình các luật cứng, ta thiết kế một kiến trúc hàm số $f_\theta(x)$ chứa các tham số $\theta \in \mathbb{R}^d$, sau đó dùng dữ liệu thực nghiệm để tự động tinh chỉnh $\theta$.
- **Khung toán học cốt lõi (Tối thiểu hóa rủi ro kinh nghiệm - ERM):**
  Cho tập dữ liệu $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N \sim \mathcal{P}_{\text{data}}$. Mục tiêu là tìm tham số $\theta^*$ sao cho cực tiểu hóa hàm mất mát trên toàn bộ dữ liệu kèm thành phần kiểm soát độ phức tạp (chính quy hóa):
  $$\theta^* = \arg\min_{\theta} \left[ \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i) + \lambda \Omega(\theta) \right]$$
  Trong đó:
  - $\mathcal{L}(\cdot, \cdot)$ là hàm mất mát (như Mean Squared Error, Cross-Entropy Loss).
  - $\Omega(\theta)$ là số hạng chính quy hóa (như chuẩn $L_1, L_2$) giúp ngăn chặn hiện tượng quá khớp (overfitting).
  - $\lambda \ge 0$ là siêu tham số kiểm soát mức độ chính quy hóa.
- **Ưu điểm:**
  1. Khả năng chịu lỗi và kháng nhiễu (Robustness) cao: Hoạt động dựa trên phân phối xác suất và ước lượng cực đại hợp lý (MLE).
  2. Học biểu diễn tự động (Representation Learning): Tự động trích xuất các đặc trưng phân cấp từ dữ liệu phi cấu trúc (điểm ảnh, sóng âm, từ ngữ).
  3. Khả năng tổng quát hóa (Generalization): Dự đoán chính xác trên dữ liệu chưa từng thấy nếu cùng phân phối.
- **Nhược điểm cốt tử:**
  1. Vấn đề "Hộp đen" (Black Box): Khó giải thích cặn kẽ tại sao mô hình đưa ra một quyết định cụ thể.
  2. Đói dữ liệu (Data Hunger) và tiêu tốn năng lượng tính toán khổng lồ.
  3. Dễ bị tấn công đối kháng (Adversarial Attacks) và sai lầm ngớ ngẩn khi gặp dữ liệu lệch phân phối (Out-Of-Distribution).

### 1.3.3 Trí tuệ nhân tạo lai (Neuro-Symbolic AI)
Sự kết hợp giữa hai hệ chuẩn được ví như hai hệ thống nhận thức trong tâm lý học nhận thức của Daniel Kahneman (*Thinking, Fast and Slow*):
- **Hệ thống 1 (Fast, Intuitive, Perception):** Do **Mạng nơ-ron (Neural)** đảm nhiệm: Nhận dạng ảnh, nghe hiểu giọng nói, trực giác nhanh.
- **Hệ thống 2 (Slow, Logical, Deliberate):** Do **Hệ ký hiệu (Symbolic)** đảm nhiệm: Lập luận logic từng bước, kiểm tra tính đúng đắn, tuân thủ ràng buộc pháp lý và đạo đức.

**Các mô hình Neuro-Symbolic đương đại:**
- **Logic Tensor Networks (LTN):** Nhúng các vị từ và mệnh đề logic bậc nhất vào không gian vector liên tục bằng đại số mờ (fuzzy logic).
- **RAG + LLM (Retrieval-Augmented Generation):** Dùng mạng nơ-ron để trích xuất ngữ nghĩa và đồ thị tri thức (Knowledge Graph) để cung cấp cơ sở lập luận chính xác.
- **Program Synthesis & Code LLMs:** Dùng mô hình xác suất để sinh ra các hàm chương trình logic tất định.

---

## 1.4 Tiến trình lịch sử của AI

Lịch sử AI không phải là đường thẳng tăng trưởng đơn điệu, mà là chu kỳ của những làn sóng kỳ vọng bùng nổ xen kẽ các giai đoạn đóng băng ("mùa đông AI").

### 1.4.1 Thời kỳ cổ đại và Nền móng triết học
- **Cổ đại:** Thần thoại Hy Lạp về người khổng lồ đồng Talos; ý tưởng chế tạo người máy cơ học (Automata) của người Ả Rập và thời kỳ Phục hưng (Leonardo da Vinci).
- **Thế kỷ 17:** Thomas Hobbes tuyên bố trong *Leviathan* (1651): *"Reason is nothing but reckoning"* (Tư duy chẳng qua là tính toán). René Descartes khởi xướng thuyết nhị nguyên tâm - thân (Mind-Body Dualism), đặt câu hỏi liệu một cỗ máy cơ học có thể nói chuyện như con người.

### 1.4.2 Nền móng toán học (Thế kỷ 19 đến nửa đầu thế kỷ 20)
- **1854:** George Boole xuất bản *The Laws of Thought*, thiết lập Đại số Boolean, nhúng logic hai giá trị (True/False) vào các phép toán đại số ($AND, OR, NOT$).
- **1931:** Kurt Gödel công bố **Định lý Bất toàn (Incompleteness Theorems)**, chứng minh rằng trong bất kỳ hệ thống tiên đề toán học hình thức nào đủ mạnh, luôn tồn tại những chân lý không thể được chứng minh hoặc bác bỏ từ chính hệ tiên đề đó.
- **1936:** Alan Turing đề xuất mô hình **Máy Turing (Turing Machine)**, định nghĩa chuẩn mực toán học về tính toán khả quy (computability) và chứng minh sự tồn tại của các bài toán không thể quyết định (Halting Problem).
- **1943:** Warren McCulloch và Walter Pitts đề xuất mô hình nơ-ron nhân tạo hình thức đầu tiên (MCP Neuron), chứng minh rằng mạng lưới các nơ-ron đơn giản có thể tính toán mọi hàm logic Boolean.
- **1948 - 1949:** Claude Shannon khai sinh **Lý thuyết thông tin (Information Theory)**; Donald Hebb đề xuất quy tắc học Synapse trong *The Organization of Behavior*: *"Cells that fire together, wire together"*.

### 1.4.3 Kỷ nguyên máy tính và Khởi sinh AI (1950 - 1974)
- **1950:** Alan Turing xuất bản bài báo đột phá *"Computing Machinery and Intelligence"*, mở đầu bằng câu hỏi lịch sử: *"Can machines think?"* và giới thiệu Phép thử Turing.
- **1956:** **Hội thảo Dartmouth (Dartmouth Summer Research Project on Artificial Intelligence)** do John McCarthy, Marvin Minsky, Nathaniel Rochester và Claude Shannon tổ chức. John McCarthy chính thức khai sinh thuật ngữ **"Artificial Intelligence"**.
- **1957 - 1958:** Frank Rosenblatt tại Phòng thí nghiệm Hàng không Cornell phát minh ra **Perceptron**, mô hình học máy đầu tiên có thể tự cập nhật trọng số từ dữ liệu để phân loại ảnh nhị phân:
  $$y = f(\mathbf{w}^T \mathbf{x} + b) = \begin{cases} +1 & \text{nếu } \mathbf{w}^T \mathbf{x} + b \ge 0 \\ -1 & \text{ngược lại} \end{cases}$$
- **1956 - 1965:** Allen Newell và Herbert Simon phát triển chương trình *Logic Theorist* (chứng minh thành công 38/52 định lý trong *Principia Mathematica*) và *General Problem Solver (GPS)*.

### 1.4.4 Hai thời kỳ "Mùa đông AI" (AI Winters)
1. **Mùa đông AI thứ nhất (1974 – 1980):**
   - **Nguyên nhân kỹ thuật:** Năm 1969, Marvin Minsky và Seymour Papert xuất bản cuốn sách *Perceptrons*, chứng minh về mặt toán học rằng Perceptron một tầng tuyến tính không thể giải quyết được hàm **XOR** (hàm phi tuyến tính tối thiểu). Giới nghiên cứu thời bấy giờ lầm tưởng kết luận này áp dụng cho toàn bộ mạng nơ-ron.
   - **Nguyên nhân tài chính:** Báo cáo *Lighthill Report (1973)* tại Anh và quyết định cắt giảm tài trợ ồ ạt từ DARPA (Mỹ) do AI không đáp ứng được những tuyên bố phóng đại về dịch máy tự động.
2. **Kỷ nguyên Hệ chuyên gia và Mùa đông thứ hai (1980 – 1993):**
   - **Thập niên 1980:** Sự trỗi dậy của Hệ chuyên gia thương mại (Expert Systems) như **XCON** (tiết kiệm cho tập đoàn DEC 40 triệu USD/năm), hệ thống chẩn đoán y khoa **MYCIN**.
   - **Mùa đông AI thứ hai:** Thị trường phần cứng chuyên dụng cho ngôn ngữ Lisp (Lisp Machines) sụp đổ trước sự phát triển của máy tính cá nhân chuẩn x86 của IBM/Apple. Các hệ chuyên gia trở nên quá đắt đỏ để bảo trì và quá giòn khi gặp dữ liệu đời thực ngoài dự tính.

### 1.4.5 Kỷ nguyên Học sâu và Trí tuệ tạo sinh (2000s đến nay)
- **1986:** Geoffrey Hinton, David Rumelhart và Ronald Williams tái khám phá và phổ biến thuật toán **Lan truyền ngược (Backpropagation)** cho mạng nơ-ron nhiều tầng, hóa giải rào cản hàm XOR của Minsky.
- **1997:** Siêu máy tính IBM **Deep Blue** đánh bại đại kiện tướng cờ vua Garry Kasparov; Yann LeCun công bố mạng tích chập **LeNet-5** nhận dạng chữ số séc ngân hàng.
- **2012 - Bước ngoặt AlexNet:** Alex Krizhevsky, Ilya Sutskever và Geoffrey Hinton giành chiến thắng áp đảo tại cuộc thi thị giác máy tính ImageNet với mạng CNN sâu 8 tầng huấn luyện trên GPU NVIDIA, giảm tỷ lệ lỗi từ $26\%$ xuống $15.3\%$. Sự kiện này chính thức mở ra kỷ nguyên bùng nổ của Deep Learning.
- **2016:** DeepMind phát triển **AlphaGo**, đánh bại nhà vô địch cờ vây thế giới Lee Sedol với tỷ số 4-1, áp dụng học tăng cường sâu (Deep RL) và thuật toán tìm kiếm cây Monte Carlo (MCTS).
- **2017:** Ashish Vaswani và các cộng sự tại Google xuất bản bài báo *"Attention Is All You Need"*, giới thiệu kiến trúc **Transformer**, loại bỏ hoàn toàn cơ chế hồi quy RNN để mở đường cho tính toán song song hóa trên quy mô khổng lồ.
- **2020 - Nay:** Kỷ nguyên các mô hình nền tảng (Foundation Models) và AI tạo sinh: GPT-3, GPT-4 (OpenAI), Claude (Anthropic), Gemini (Google), LLaMA (Meta), Midjourney, Stable Diffusion.

---

## 1.5 Trí tuệ nhân tạo hẹp và Trí tuệ nhân tạo tổng quát

```
    [ Cấp độ 1: ANI ]               [ Cấp độ 2: AGI ]               [ Cấp độ 3: ASI ]
  Narrow / Weak AI                General / Strong AI              Superintelligence
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ • Chuyên gia một tác vụ │     │ • Đa năng như con người │     │ • Vượt xa toàn bộ       │
│ • AlphaGo, Siri, YOLO   │ ──► │ • Tự thích nghi hoàn cảnh│ ──► │   nhân loại gộp lại     │
│ • Hiện trạng thế giới   │     │ • Mục tiêu nghiên cứu   │     │ • Giả thuyết Singularity│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### 1.5.1 Trí tuệ nhân tạo hẹp (Artificial Narrow Intelligence - ANI / Weak AI)
- **Định nghĩa:** Hệ thống AI được tối ưu hóa cao độ để thực hiện một tác vụ cụ thể hoặc giải quyết một miền bài toán xác định trong không gian đóng.
- **Đặc điểm kỹ thuật:**
  - Hoạt động trong giả định phân phối độc lập cùng phân phối ($i.i.d.$) giữa tập huấn luyện và thực tế: $P_{\text{train}}(X, Y) \approx P_{\text{test}}(X, Y)$.
  - Bị giới hạn nghiêm trọng bởi **Hiện tượng quên lãng tai hại (Catastrophic Forgetting)**: Khi một mạng nơ-ron huấn luyện cho tác vụ $A$ được huấn luyện tiếp cho tác vụ $B$, nó có xu hướng xóa sạch trọng số tối ưu của tác vụ $A$.
- **Ví dụ thực tế:** Hệ thống tự hành Tesla, bộ cờ vua Stockfish, thuật toán gợi ý TikTok, mô hình chẩn đoán khối u từ ảnh X-quang.

### 1.5.2 Trí tuệ nhân tạo tổng quát (Artificial General Intelligence - AGI / Strong AI)
- **Định nghĩa:** Hệ thống máy tính giả định sở hữu năng lực trí tuệ tương đương con người trên hầu hết mọi lĩnh vực kinh tế và nhận thức: có khả năng tự học, suy luận trừu tượng, chuyển giao tri thức liên miền (Cross-domain Transfer), hiểu quan hệ nhân quả (Causality) và lập kế hoạch dài hạn.
- **Các tiêu chuẩn đánh giá AGI đề xuất:**
  1. **Bài kiểm tra cà phê của Steve Wozniak:** Robot AI có thể vào một ngôi nhà bất kỳ của người lạ, tìm phòng bếp, nhận diện nguyên liệu, tự pha một tách cà phê và mang ra bàn.
  2. **Bài kiểm tra Đại học:** Hệ thống có thể đăng ký học trực tuyến, làm bài tập, viết luận án, vượt qua các kỳ thi và nhận bằng tiến sĩ.
  3. **Thử nghiệm việc làm:** AI có khả năng thay thế hoàn toàn một nhân viên tri thức con người trong một công việc văn phòng bình thường (như lập trình viên, luật sư, kế toán).
  4. **ARC Challenge (Abstraction and Reasoning Corpus của François Chollet):** Bộ câu đố trực quan đòi hỏi suy luận logic từ ít mẫu (Few-shot) mà không thể giải quyết bằng cách ghi nhớ dữ liệu lớn đơn thuần.

### 1.5.3 Siêu trí tuệ nhân tạo (Artificial Superintelligence - ASI) và Điểm kỳ dị
- **Định nghĩa (Nick Bostrom - *Superintelligence: Paths, Dangers, Strategies*):**
  > *"Một trí tuệ vượt trội hơn rất nhiều so với những bộ óc thông minh nhất của con người trên hầu hết mọi lĩnh vực, bao gồm sáng tạo khoa học, sự thông thái và các kỹ năng xã hội."*
- **Điểm kỳ dị công nghệ (Technological Singularity - Ray Kurzweil):** Thời điểm một hệ thống AGI đạt khả năng tự cải tiến đệ quy mã nguồn của chính nó (Recursive Self-Improvement), tạo ra vụ nổ trí tuệ (Intelligence Explosion), vượt xa tầm kiểm soát và nhận thức của văn minh loài người.
- **Vấn đề Căn chỉnh giá trị (AI Alignment Problem):** Làm sao để đảm bảo các hệ thống siêu trí tuệ theo đuổi những mục tiêu an toàn và tương thích với đạo đức của con người?
  - Ví dụ minh họa kinh điển: *Bài toán chiếc kẹp giấy (Paperclip Maximizer)* của Nick Bostrom: Nếu một AI tối thượng được giao mục tiêu cực đại hóa số lượng kẹp giấy mà không được căn chỉnh các giá trị con người, nó có thể biến đổi toàn bộ nguyên tử trên Trái Đất (kể cả cơ thể sinh học của con người) thành kẹp giấy.

---

## 1.6 Bảng đối chiếu tổng kết: Ký hiệu vs. Thống kê

| Tiêu chí | AI Ký hiệu (Symbolic AI) | AI Thống kê (Statistical AI / ML) |
| :--- | :--- | :--- |
| **Bản chất tri thức** | Các ký hiệu rời rạc, vị từ logic, cây cấu trúc | Vector nhúng nhiều chiều, ma trận trọng số liên tục |
| **Cơ chế hoạt động** | Thao tác cú pháp theo quy tắc suy diễn ($P \to Q$) | Tối ưu hóa đạo hàm hàm mất mát ($\nabla_\theta \mathcal{L}$) |
| **Nguồn gốc tri thức** | Con người tự tay nạp vào (Hand-crafted rules) | Mô hình tự học từ dữ liệu thực tế (Data-driven) |
| **Tính minh bạch** | Hộp trắng hoàn toàn (White-box, truy vết 100%) | Hộp đen một phần hoặc toàn bộ (Black-box) |
| **Khả năng chống nhiễu** | Rất kém, độ giòn cao khi dữ liệu sai lệch | Rất tốt, lọc nhiễu dựa trên ước lượng xác suất |
| **Tài nguyên tính toán** | Thấp, bộ nhớ phụ thuộc độ sâu cây tìm kiếm | Cực lớn, đòi hỏi hệ thống tính toán GPU/TPU đồ sộ |
| **Miền áp dụng tối ưu** | Lập kế hoạch tài chính, logic hình thức, hệ thống quy chuẩn | Thị giác, xử lý tiếng nói, hiểu ngữ nghĩa văn bản |

---

## 1.7 Tài liệu tham khảo (References)

Tất cả các tài liệu được trích dẫn dưới đây đều là các công trình học thuật kinh điển, có sẵn trong thư mục `resources/` của dự án hoặc các bài báo nền tảng được kiểm chứng quốc tế:

1. **Russell, S., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. (Giáo trình tiêu chuẩn vàng cho mục 1.1 và 1.2).
2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. [Tham khảo tại: `resources/Deep Learning.pdf`] (Chương 1: Introduction - Tiến trình và động lực của học sâu).
3. **Mitchell, T. M. (1997).** *Machine Learning*. McGraw-Hill. [Tham khảo tại: `resources/Machine Learning - Tom Mitchell.pdf`] (Chương 1: Introduction - Định nghĩa toán học của học máy).
4. **Murphy, K. P. (2012).** *Machine Learning: A Probabilistic Perspective*. MIT Press. [Tham khảo tại: `resources/Machine Learning A Probabilistic Perspective.pdf`] (Chương 1: Introduction - Góc nhìn xác suất thống kê trong AI).
5. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. [Tham khảo tại: `resources/Pattern Recognition and Machine Learning.pdf`] (Chương 1: Introduction - Lý thuyết quyết định và lý thuyết xác suất).
6. **Vũ Hữu Tiệp (2018).** *Machine Learning Cơ Bản*. NXB Khoa học và Kỹ thuật. [Tham khảo tại: `resources/Machine Learning Co Ban.pdf`] (Chương 1: Giới thiệu).
7. **Turing, A. M. (1950).** Computing Machinery and Intelligence. *Mind*, 59(236), 433-460. (Nguồn gốc Phép thử Turing).
8. **Newell, A., & Simon, H. A. (1976).** Computer science as empirical inquiry: Symbols and search. *Communications of the ACM*, 19(3), 113-126. (Giả thuyết hệ thống ký hiệu vật lý).
9. **Searle, J. R. (1980).** Minds, brains, and programs. *Behavioral and Brain Sciences*, 3(3), 417-424. (Lập luận Căn phòng Trung Hoa).
10. **Vaswani, A., et al. (2017).** Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998-6008.

---

## 1.8 Định hướng thực hành (`demo.ipynb`)

Mục tiêu của bài thực hành trong tệp [`demo.ipynb`](./demo.ipynb) là giúp người học "chạm tay" vào sự khác biệt thực tế giữa hai hệ chuẩn: **Trí tuệ nhân tạo ký hiệu (Symbolic AI)** và **Học máy thống kê (Statistical Learning)** qua bài toán cụ thể: **Thẩm định Rủi ro Tín dụng (Credit Risk Assessment)**.

### Nội dung triển khai trong `demo.ipynb`:
1. **Bài toán đặt ra:** Dự đoán một khách hàng xin vay vốn có khả năng vỡ nợ ($Y = 1$) hay trả nợ tốt ($Y = 0$) dựa trên 2 chỉ số tài chính chuẩn hóa:
   - Thu nhập hàng tháng chuẩn hóa: $X_1 \in [0, 10]$
   - Điểm lịch sử tín dụng chuẩn hóa: $X_2 \in [0, 10]$
2. **Thành phần 1 - Xây dựng Hệ chuyên gia Ký hiệu (Symbolic Rule-Based Engine):**
   - Định nghĩa tập luật chuyên gia IF-THEN kinh điển từ các chuyên viên thẩm định ngân hàng:
     $$\text{RULE 1: IF } (X_1 < 3.5 \text{ AND } X_2 < 4.0) \implies Y = 1 \text{ (Từ chối)}$$
     $$\text{RULE 2: IF } (X_1 \ge 6.0 \text{ OR } X_2 \ge 7.0) \implies Y = 0 \text{ (Phê duyệt)}$$
   - Thử nghiệm khi dữ liệu có nhiễu (Noise) hoặc rơi vào "vùng mù" (tri thức chưa được định nghĩa luật) dẫn đến sự "giòn gãy" của hệ thống.
3. **Thành phần 2 - Xây dựng Mô hình Học máy Thống kê (Statistical Perceptron / Logistic Regression từ đầu):**
   - Tự cài đặt thuật toán tối ưu hóa Gradient Descent chỉ bằng thư viện `numpy` (không dùng Scikit-learn):
     $$z = \mathbf{w}^T \mathbf{x} + b, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
     $$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
   - Cập nhật trọng số qua từng vòng lặp (Epoch):
     $$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$
4. **Thành phần 3 - Xây dựng Hệ lai Neuro-Symbolic đơn giản:**
   - Sử dụng mô hình thống kê để dự đoán xác suất, nhưng áp dụng bộ quy tắc ký hiệu cứng (Guardrail Rules) làm "chốt chặn an toàn" (Safety Policy) cho các trường hợp bắt buộc.
5. **Thành phần 4 - Đánh giá và Trực quan hóa so sánh:**
   - Vẽ ranh giới quyết định (Decision Boundary) của hai phương pháp trên mặt phẳng 2D cạnh nhau bằng `matplotlib`.
   - Phân tích bảng so sánh định lượng về: Độ chính xác (Accuracy), Khả năng kháng nhiễu (Robustness), và Khả năng giải thích (Interpretability).
