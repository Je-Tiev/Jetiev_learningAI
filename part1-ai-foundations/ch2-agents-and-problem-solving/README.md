# Chương 2: Tác tử và Giải quyết bài toán (Agents and Problem Solving)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Nắm vững mô hình Tác tử và Môi trường:** Định nghĩa hình thức tác tử, hàm tác tử và phân tích toàn diện môi trường bài toán thông qua khuôn mẫu **PEAS** (Performance, Environment, Actuators, Sensors).
2. **Hình thức hóa bài toán tìm kiếm:** Xây dựng mô hình toán học cho bài toán tìm kiếm gồm 5 thành phần cốt lõi: Trạng thái khởi đầu, Tập hành động, Mô hình chuyển đổi, Kiểm tra mục tiêu và Hàm chi phí.
3. **Thấu hiểu và cài đặt các thuật toán tìm kiếm mù (Uninformed Search):** Phân tích tính đầy đủ, tính tối ưu, độ phức tạp không gian và thời gian của BFS, DFS, UCS và IDS.
4. **Làm chủ tìm kiếm có thông tin (Informed Search) và thuật toán A\*:** Hiểu sâu sắc điều kiện chấp nhận được (Admissibility) và tính nhất quán (Consistency) của hàm Heuristic để đảm bảo nghiệm tối ưu toàn cục.
5. **Vận dụng tìm kiếm cục bộ (Local Search):** Hiểu cơ chế vượt qua cực trị địa phương của giải thuật Leo đồi (Hill-Climbing) và Luyện kim mô phỏng (Simulated Annealing).
6. **Kết nối tìm kiếm cổ điển với AI hiện đại:** Hiểu bản chất thuật toán tìm kiếm cây Monte Carlo (MCTS) trong AlphaGo, thuật toán Beam Search trong LLMs, và mô hình Cây suy nghĩ (Tree of Thoughts - ToT) trong Tác tử tự trị.

---

## 2.1 Tác tử và Môi trường (Agents and Environments)

### 2.1.1 Khái niệm Tác tử (Agent)
Một **tác tử (agent)** là bất kỳ thực thể nào có khả năng nhận thức môi trường thông qua các **cảm biến (sensors)** và tác động lên môi trường đó thông qua các **bộ phận chấp hành (actuators)**.

```
                  ┌──────────────────────────────┐
                  │          MÔI TRƯỜNG          │
                  └───────┬──────────────▲───────┘
                          │              │
                Nhận thức │              │ Hành động
                (Percepts)│              │ (Actions)
                          ▼              │
                  ┌──────────────────────┴───────┐
                  │    Cảm biến      Chấp hành   │
                  │   (Sensors)     (Actuators)  │
                  │        \            /        │
                  │         ▼          ▲         │
                  │       ┌──────────────┐       │
                  │       │ Chương trình │       │
                  │       │    tác tử    │       │
                  │       └──────────────┘       │
                  │            TÁC TỬ            │
                  └──────────────────────────────┘
```

- **Chuỗi nhận thức (Percept Sequence):** Lịch sử toàn bộ những gì tác tử đã ghi nhận được cho đến thời điểm hiện tại: $\mathbf{p} = (p_1, p_2, \dots, p_t) \in \mathcal{P}^*$.
- **Hàm tác tử (Agent Function):** Ánh xạ trừu tượng toán học từ mọi chuỗi nhận thức có thể sang một hành động cụ thể:
  $$f: \mathcal{P}^* \to \mathcal{A}$$
- **Chương trình tác tử (Agent Program):** Sự cài đặt cụ thể của hàm tác tử chạy trên một kiến trúc phần cứng vật lý (Architecture): $\text{Agent} = \text{Architecture} + \text{Program}$.

### 2.1.2 Mô hình PEAS
Để thiết kế một tác tử thông minh hoàn chỉnh, kỹ sư AI cần xác định rõ 4 yếu tố PEAS:
- **P (Performance Measure):** Thước đo hiệu năng khách quan đánh giá mức độ thành công của hành vi tác tử.
- **E (Environment):** Môi trường bên ngoài mà tác tử nhúng vào và tương tác.
- **A (Actuators):** Các cơ cấu chấp hành giúp tác tử thực thi thay đổi lên môi trường.
- **S (Sensors):** Hệ thống cảm biến giúp tác tử thu nhận tín hiệu từ môi trường.

#### Bảng phân tích PEAS cho các hệ thống AI điển hình:

| Hệ thống AI | Thước đo hiệu năng (P) | Môi trường (E) | Bộ chấp hành (A) | Cảm biến (S) |
| :--- | :--- | :--- | :--- | :--- |
| **Xe tự hành (Tesla, Waymo)** | An toàn, tốc độ, đúng luật, êm ái, tối ưu tiêu hao năng lượng | Đường xá, xe cộ, người đi bộ, thời tiết, biển báo | Vô lăng, chân ga, chân phanh, còi, đèn tín hiệu | Camera, LiDAR, Radar, GPS, cảm biến siêu âm, IMU |
| **Hệ thống giao dịch thuật toán (Algo-Trading)** | Cực đại hóa lợi nhuận (ROI), cực tiểu hóa rủi ro (VaR, Max Drawdown) | Thị trường tài chính, sàn giao dịch, sổ lệnh (Order book) | Lệnh Mua / Bán / Hủy, lệnh phái sinh phòng ngừa | Dữ liệu giá thời gian thực, tin tức kinh tế, feeds mạng xã hội |
| **Robot dọn dẹp vệ sinh thông minh** | Diện tích làm sạch, thời gian, mức tiêu thụ pin, tránh va chạm | Sàn nhà, phòng ốc, thảm, chướng ngại vật nội thất | Bánh xe di chuyển, chổi quét, động cơ hút, vòi lau | Cảm biến va chạm, LiDAR quét phòng, cảm biến rơi bậc thang |
| **Tác tử Chẩn đoán y tế (Medical AI)** | Độ chính xác chẩn đoán, giảm chi phí xét nghiệm, tỷ lệ cứu sống | Bệnh nhân, phòng khám, hồ sơ bệnh án điện tử (EHR) | Đưa ra chẩn đoán, đề xuất phác đồ điều trị, kê đơn | Bàn phím nhập triệu chứng, dữ liệu ảnh MRI/CT, kết quả xét nghiệm |

### 2.1.3 Phân loại các tính chất của Môi trường
Bản chất của môi trường quyết định trực tiếp thiết kế của giải thuật tìm kiếm và học máy:
1. **Quan sát được toàn phần vs. Một phần (Fully observable vs. Partially observable):** Cảm biến có cung cấp trạng thái hoàn chỉnh của môi trường tại mỗi thời điểm hay không (ví dụ: Cờ vua là toàn phần; Poker hay Lái xe thực tế là một phần).
2. **Đơn tác tử vs. Đa tác tử (Single-agent vs. Multi-agent):** Môi trường chỉ có một tác tử hoạt động (giải Sudoku) hay có nhiều tác tử tương tác cạnh tranh/hợp tác (chơi cờ, đá bóng robot).
3. **Tất định vs. Ngẫu nhiên (Deterministic vs. Stochastic):** Trạng thái tiếp theo được quyết định hoàn toàn bởi trạng thái hiện tại và hành động của tác tử, hay có yếu tố may rủi/bất định.
4. **Từng hồi vs. Tiếp nối (Episodic vs. Sequential):** Mỗi quyết định là độc lập (phân loại ảnh) hay hành động hiện tại ảnh hưởng đến tất cả các trạng thái tương lai (chơi cờ, điều khiển xe).
5. **Tĩnh vs. Động (Static vs. Dynamic):** Môi trường có tự thay đổi trong khi tác tử đang suy nghĩ hay không (Cờ vua có đồng hồ bấm giờ là Semi-dynamic; Lái xe trên đường cao tốc là Dynamic).
6. **Rời rạc vs. Liên tục (Discrete vs. Continuous):** Không gian trạng thái và thời gian là tập rời rạc (bàn cờ) hay liên tục (tọa độ, vận tốc của robot).

### 2.1.4 Tác tử hợp lý (Rational Agent)
Một tác tử hợp lý là tác tử lựa chọn hành động $a^*$ nhằm cực đại hóa giá trị kỳ vọng của thước đo hiệu năng:
$$a^* = \arg\max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) U(s')$$
- **Tính hợp lý $\neq$ Toàn tri (Omniscience):** Toàn tri biết trước kết quả thực tế của hành động; tính hợp lý chỉ đòi hỏi tối ưu hóa **kỳ vọng** dựa trên những gì tác tử có thể nhận thức được.
- **Tính tự trị (Autonomy):** Tác tử có khả năng tự bù đắp những thiếu hụt hoặc sai lệch trong kiến thức tiên nghiệm ban đầu bằng việc học từ trải nghiệm thực tế.

---

## 2.2 Cấu trúc hóa vấn đề: Bài toán tìm kiếm (Search Problem Formulation)

Khi tác tử cần đạt được một mục tiêu nhưng không thể quyết định ngay bằng phản xạ tức thời, nó phải lập kế hoạch thông qua **tìm kiếm (search)** trong không gian trạng thái.

### 2.2.1 Năm thành phần hình thức của một bài toán tìm kiếm
Một bài toán tìm kiếm được định nghĩa chặt chẽ bằng bộ 5 thành phần:
1. **Không gian trạng thái ($S$):** Tập hợp tất cả các cấu hình hợp lệ của bài toán.
2. **Trạng thái khởi đầu ($s_0 \in S$):** Điểm xuất phát của tác tử.
3. **Tập hành động khả thi ($\text{Actions}(s)$):** Cho biết các hành động hợp lệ mà tác tử có thể thực hiện tại trạng thái $s$:
   $$\text{Actions}(s) = \{a \in \mathcal{A} \mid a \text{ hợp lệ tại } s\}$$
4. **Mô hình chuyển đổi trạng thái ($\text{Result}(s, a)$):** Trả về trạng thái kết quả $s'$ khi thực hiện hành động $a$ tại trạng thái $s$:
   $$s' = \text{Result}(s, a)$$
5. **Kiểm tra mục tiêu ($\text{Goal-Test}(s)$):** Hàm kiểm tra xem trạng thái $s$ có phải là đích hay không:
   $$\text{Goal-Test}(s) \in \{\text{True}, \text{False}\}$$
6. **Hàm chi phí đường đi ($c(s, a, s')$ và $g(n)$):** Chi phí bước đi để đi từ $s$ sang $s'$ thông qua hành động $a$. Chi phí tổng lũy kế từ gốc $s_0$ đến nút hiện tại $n$ ký hiệu là $g(n)$:
   $$g(n) = \sum_{i=1}^k c(s_{i-1}, a_i, s_i)$$

### 2.2.2 Cấu trúc nút tìm kiếm (Search Node)
Cần phân biệt rõ giữa **Trạng thái (State)** và **Nút tìm kiếm (Node)**:
- *Trạng thái:* Biểu diễn một cấu hình vật lý của thế giới.
- *Nút:* Một cấu trúc dữ liệu dùng để xây dựng cây tìm kiếm, chứa thông tin kế toán đường đi:
```python
class SearchNode:
    state       # Trạng thái trong không gian bài toán
    parent      # Con trỏ trỏ tới nút cha tạo ra nút này
    action      # Hành động được thực thi từ nút cha để tới nút này
    path_cost   # Chi phí lũy kế g(n) từ nút gốc đến nút này
    depth       # Độ sâu của nút trong cây tìm kiếm
```

---

## 2.3 Tìm kiếm mù (Uninformed / Blind Search)

Tìm kiếm mù là các chiến lược tìm kiếm chỉ sử dụng thông tin định nghĩa trong bài toán (sinh trạng thái kế tiếp và kiểm tra mục tiêu), không có bất kỳ thông tin nào về việc một trạng thái chưa mục tiêu có "gần" đích hơn trạng thái khác hay không.

### 2.3.1 Bốn tiêu chí đánh giá giải thuật
1. **Tính đầy đủ (Completeness):** Giải thuật có luôn tìm thấy lời giải nếu lời giải tồn tại hay không?
2. **Tính tối ưu (Optimality):** Giải thuật có đảm bảo tìm thấy đường đi có chi phí $g(n)$ thấp nhất hay không?
3. **Độ phức tạp thời gian (Time Complexity):** Số lượng nút được sinh ra trước khi tìm thấy đích.
4. **Độ phức tạp không gian (Space Complexity):** Số lượng nút tối đa phải lưu giữ trong bộ nhớ RAM cùng một thời điểm.

*Các tham số đo lường:*
- $b$: Hệ số nhánh tối đa (Branching factor).
- $d$: Độ sâu của nút mục tiêu nông nhất (Shallowest goal depth).
- $m$: Độ sâu tối đa của không gian trạng thái (có thể $m = \infty$).

### 2.3.2 Tìm kiếm theo chiều rộng (Breadth-First Search - BFS)
- **Cơ chế:** Mở rộng nút gốc trước, sau đó mở rộng toàn bộ các nút con ở độ sâu 1, rồi đến độ sâu 2, v.v. Sử dụng cấu trúc hàng đợi **FIFO Queue**.
- **Đánh giá:**
  - *Đầy đủ:* Có (nếu $b$ hữu hạn).
  - *Tối ưu:* Chỉ tối ưu khi chi phí mọi bước đi bằng nhau ($c=1$). Không tối ưu khi chi phí các bước khác nhau.
  - *Thời gian:* $\mathcal{O}(b^d)$.
  - *Không gian:* $\mathcal{O}(b^d)$ — Toàn bộ các nút ở tầng biên (Frontier) phải được lưu trong RAM. Đây là điểm yếu chết người của BFS.

### 2.3.3 Tìm kiếm theo chiều sâu (Depth-First Search - DFS)
- **Cơ chế:** Luôn mở rộng nút sâu nhất hiện tại trên nhánh tìm kiếm. Sử dụng cấu trúc ngăn xếp **LIFO Stack** (hoặc đệ quy hệ thống).
- **Đánh giá:**
  - *Đầy đủ:* Không (bế tắc nếu nhánh cây có độ sâu vô hạn hoặc có chu trình lặp vô tận).
  - *Tối ưu:* Không (có thể trả về lời giải rất sâu trong khi có lời giải nông hơn ở nhánh khác).
  - *Thời gian:* $\mathcal{O}(b^m)$ (có thể tệ hơn nhiều so với BFS nếu $m \gg d$).
  - *Không gian:* $\mathcal{O}(b \cdot m)$ — **Cực kỳ tiết kiệm bộ nhớ**, chỉ cần lưu trữ các nút dọc theo nhánh đang duyệt và các nút anh em chưa duyệt.

> **Giải pháp tối ưu kết hợp: Tìm kiếm sâu dần lặp lại (Iterative Deepening Search - IDS)**  
> Chạy DFS lặp lại với giới hạn độ sâu tăng dần $l = 0, 1, 2, \dots, d$.  
> - *Đầy đủ:* Có; *Tối ưu:* Có (khi chi phí bước đều nhau);  
> - *Bộ nhớ:* $\mathcal{O}(b \cdot d)$ (nhẹ như DFS);  
> - *Thời gian:* $\mathcal{O}(b^d)$ (chỉ tốn thêm khoảng $\frac{b}{b-1}$ phép tính so với BFS do các tầng trên rất nhỏ).

### 2.3.4 Tìm kiếm theo chi phí đồng nhất (Uniform-Cost Search - UCS / Dijkstra)
- **Cơ chế:** Thay vì mở rộng nút theo độ sâu, UCS mở rộng nút có chi phí đường đi lũy kế $g(n)$ nhỏ nhất. Sử dụng **Priority Queue** với khóa là $g(n)$.
- **Đánh giá:**
  - *Đầy đủ:* Có (nếu mọi chi phí bước $c(s, a, s') \ge \epsilon > 0$).
  - *Tối ưu:* **Luôn tối ưu** với mọi hàm chi phí không âm.
  - *Độ phức tạp:* $\mathcal{O}\left(b^{1 + \lfloor C^* / \epsilon \rfloor}\right)$, trong đó $C^*$ là chi phí của đường đi tối ưu.

---

## 2.4 Tìm kiếm có thông tin (Informed / Heuristic Search)

Tìm kiếm có thông tin sử dụng tri thức chuyên biệt của bài toán thông qua một **hàm đánh giá Heuristic $h(n)$**:
$$h(n) = \text{Chi phí ước lượng từ nút } n \text{ đến trạng thái mục tiêu gần nhất}$$
Điều kiện tiên quyết: $h(n) \ge 0$ và nếu $n$ là trạng thái mục tiêu thì $h(n) = 0$.

### 2.4.1 Tìm kiếm Tham lam theo lựa chọn tốt nhất (Greedy Best-First Search)
- **Cơ chế:** Đánh giá nút bằng hàm $f(n) = h(n)$. Luôn mở rộng nút được cho là gần đích nhất.
- **Hạn chế:** Tương tự DFS, giải thuật dễ bị đánh lừa bởi địa hình cục bộ, có thể rơi vào đường vòng vô tận và không đảm bảo tìm được nghiệm tối ưu.

### 2.4.2 Thuật toán tìm kiếm A\* (A* Search)
Thuật toán A\* kết hợp chi phí thực tế đã đi qua $g(n)$ với chi phí ước lượng còn lại $h(n)$:
$$f(n) = g(n) + h(n)$$
- $g(n)$: Chi phí chính xác để đi từ nút khởi đầu $s_0$ đến nút hiện tại $n$.
- $h(n)$: Chi phí ước lượng rẻ nhất từ nút $n$ đến đích.
- $f(n)$: Chi phí ước lượng của đường đi rẻ nhất đi qua nút $n$ để đến đích.

```
       s0 ●─────────────────────────► ● n ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌► ● Goal
          │◄─────── g(n) ────────────►│◄─────── h(n) ──────────►│
          │                           │                         │
          └───────────────────────── f(n) ──────────────────────┘
```

#### Hai điều kiện toán học đảm bảo tính tối ưu của A\*:
1. **Tính chấp nhận được (Admissibility - Dành cho Tìm kiếm Cây):**
   Một hàm heuristic $h(n)$ là chấp nhận được nếu nó **không bao giờ đánh giá cao hơn** chi phí thực tế tối ưu $h^*(n)$:
   $$0 \le h(n) \le h^*(n), \quad \forall n$$
   *Ý nghĩa:* $h(n)$ luôn lạc quan (optimistic). Nếu $h(n)$ chấp nhận được, A\* chạy trên cây tìm kiếm luôn đảm bảo tìm được nghiệm tối ưu.

2. **Tính nhất quán / Đơn điệu (Consistency / Monotonicity - Dành cho Tìm kiếm Đồ thị):**
   Một hàm heuristic $h(n)$ là nhất quán nếu với mọi nút $n$ và nút con $n'$ sinh ra bởi hành động $a$, $h(n)$ thỏa mãn bất đẳng thức tam giác:
   $$h(n) \le c(n, a, n') + h(n')$$
   *Hệ quả:*
   - Giá trị $f(n)$ không bao giờ giảm dọc theo bất kỳ đường đi nào: $f(n') = g(n') + h(n') = g(n) + c(n, a, n') + h(n') \ge g(n) + h(n) = f(n)$.
   - Khi A\* mở rộng một nút $n$, đường đi tới $n$ được bảo đảm là tối ưu tuyệt đối. Do đó, A\* không bao giờ phải mở lại một nút đã đóng (Closed set), giúp tối ưu hiệu năng tính toán trên đồ thị.

---

## 2.5 Tìm kiếm cục bộ và Tối ưu hóa (Local Search & Optimization)

Trong nhiều bài toán thực tế (bài toán xếp $N$-hậu, định tuyến mạch điện tử, lập lịch nhà máy), chuỗi hành động dẫn tới đích không quan trọng; điều duy nhất ta quan tâm là **cấu hình của trạng thái đích tối ưu**. Các giải thuật tìm kiếm cục bộ chỉ duy trì một trạng thái hiện tại và di chuyển sang các trạng thái lân cận mà không cần lưu vết đường đi.

### 2.5.1 Thuật toán Leo đồi (Hill-Climbing Search)
- **Ý tưởng:** Di chuyển liên tục theo hướng có giá trị hàm mục tiêu tăng nhanh nhất (hoặc chi phí giảm nhanh nhất - Gradient Descent trong không gian rời rạc).
- **Các cạm bẫy địa hình:**
  1. *Cực đại địa phương (Local Maxima):* Đỉnh cao hơn mọi điểm xung quanh nhưng thấp hơn đỉnh toàn cục.
  2. *Sống núi (Ridges):* Chuỗi các đỉnh dốc khiến leo đồi dễ bị dao động ziczac chậm chạp.
  3. *Cao nguyên (Plateaux):* Vùng phẳng mà mọi trạng thái láng giềng có giá trị ngang nhau, làm mất phương hướng tìm kiếm.

### 2.5.2 Thuật toán Luyện kim mô phỏng (Simulated Annealing)
Lấy cảm hứng từ quá trình tôi luyện kim loại (nung nóng kim loại rồi làm nguội từ từ để các nguyên tử sắp xếp về trạng thái tinh thể có năng lượng tự do thấp nhất):
- Thay vì luôn chọn bước đi tốt nhất, giải thuật cho phép thực hiện cả các bước đi "xấu hơn" với một xác suất nhất định để thoát khỏi cực trị địa phương.
- Xác suất chấp nhận bước đi làm giảm chất lượng $\Delta E = E(s_{\text{next}}) - E(s_{\text{current}}) < 0$ tại nhiệt độ $T$:
  $$P(\text{Chấp nhận bước xấu}) = e^{\frac{\Delta E}{T}}$$
- Khi nhiệt độ $T$ còn cao (giai đoạn đầu), $P \approx 1$ — thuật toán nhảy tự do để khám phá (Exploration).
- Khi $T \to 0$ theo lịch làm nguội (Cooling schedule: $T_{k+1} = \alpha T_k$), xác suất chấp nhận bước xấu triệt tiêu — thuật toán thu hẹp vào khai thác tối ưu cục bộ (Exploitation).

---

## 2.6 Tìm kiếm trong Trí tuệ nhân tạo hiện đại (Search in Modern AI)

Khái niệm tìm kiếm không hề biến mất trong kỷ nguyên Deep Learning mà đã chuyển hóa thành các cơ chế cốt lõi trong các hệ thống AI tiên tiến nhất:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TÌM KIẾM TRONG AI HIỆN ĐẠI                            │
├───────────────────────┬─────────────────────────────┬───────────────────────┤
│   Monte Carlo Tree    │         Beam Search         │   Tree of Thoughts    │
│     Search (MCTS)     │        (NLP / GenAI)        │      (LLM Agents)     │
├───────────────────────┼─────────────────────────────┼───────────────────────┤
│ • AlphaGo / AlphaZero │ • Giải mã chuỗi ngôn ngữ    │ • Lập luận nhiều bước │
│ • Cân bằng Khám phá & │ • Giữ lại Top-K giả thuyết  │ • Tự đánh giá & quay  │
│   Khai thác bằng UCT  │   có xác suất cao nhất      │   lui (Backtracking)  │
└───────────────────────┴─────────────────────────────┴───────────────────────┘
```

### 2.6.1 Cờ vây và Tìm kiếm Cây Monte Carlo (Monte Carlo Tree Search - MCTS)
Trò chơi cờ vây có hệ số nhánh $b \approx 250$ và độ sâu ván đấu $d \approx 150$, tạo ra không gian trạng thái $\sim 10^{170}$ (vượt xa số nguyên tử trong vũ trụ), khiến giải thuật Minimax/Alpha-Beta truyền thống hoàn toàn bất khả thi. MCTS giải quyết thách thức này qua 4 bước chu kỳ:
1. **Lựa chọn (Selection):** Đi từ gốc xuống nút lá bằng công thức UCT (Upper Confidence Bound for Trees) để cân bằng giữa Khai thác (Exploitation) và Khám phá (Exploration):
   $$\text{UCT}(v_i) = \frac{Q(v_i)}{N(v_i)} + c \sqrt{\frac{\ln N(v)}{N(v_i)}}$$
   Trong đó $Q(v_i)$ là điểm số tích lũy, $N(v_i)$ là số lần thăm nút con $v_i$, $N(v)$ là số lần thăm nút cha, $c$ là hằng số khám phá ($\sqrt{2}$).
2. **Mở rộng (Expansion):** Thêm một hoặc nhiều nút con mới vào cây tìm kiếm.
3. **Mô phỏng (Simulation / Rollout):** Chơi nhanh ván cờ tới kết thúc bằng chính sách ngẫu nhiên hoặc mạng Policy Network để ước lượng kết quả thắng/thua.
4. **Lan truyền ngược (Backpropagation):** Cập nhật kết quả thắng/thua ngược lên toàn bộ các nút tổ tiên trên đường dẫn.

### 2.6.2 Sinh văn bản trong NLP và Tìm kiếm Chùm (Beam Search)
Khi các mô hình ngôn ngữ (như GPT, Transformer) sinh văn bản, không gian chuỗi từ có kích thước khổng lồ $|V|^L$ ($|V| \approx 50.000$ từ vựng, độ dài $L$).
- **Tìm kiếm tham lam (Greedy Search):** Chỉ chọn từ có xác suất cao nhất tại mỗi bước $y_t = \arg\max P(y_t \mid y_{<t})$ $\to$ Dễ rơi vào câu lặp rỗng hoặc văn bản chất lượng kém.
- **Tìm kiếm chùm (Beam Search):** Duy trì một chùm (beam) gồm $K$ chuỗi có xác suất logarit tích lũy cao nhất tại mỗi bước thời gian:
  $$\mathcal{B}_t = \arg\operatorname*{Top-}K \sum_{\tau=1}^t \log P(y_\tau \mid y_{<\tau}, x)$$
  Khi đạt độ dài tối đa hoặc sinh ra token kết thúc `<EOS>`, chuỗi có điểm chuẩn hóa độ dài cao nhất được chọn.

### 2.6.3 Tác tử LLMs và Cây suy nghĩ (Tree of Thoughts - ToT)
- **Chain-of-Thought (CoT):** Mô hình sinh suy luận theo một chuỗi tuyến tính duy nhất, nếu một bước sai thì toàn bộ kết quả sau đó sai hỏng hoàn toàn.
- **Tree of Thoughts (ToT - Yao et al., 2023):** Khung kiến trúc cho phép LLM đóng vai trò đồng thời:
  1. *Bộ sinh trạng thái (Thought Generator):* Đề xuất nhiều hướng suy nghĩ láng giềng khả dĩ.
  2. *Bộ đánh giá trạng thái (State Evaluator):* Tự chấm điểm xác suất thành công của từng hướng suy nghĩ ($h(n)$).
  3. *Bộ máy tìm kiếm cổ điển (Search Engine):* Sử dụng BFS, DFS hoặc A\* để điều phối quá trình suy nghĩ, cho phép **quay lui (backtracking)** khi nhận thấy nhánh suy nghĩ hiện tại đi vào ngõ cụt.

---

## 2.7 Bảng tổng hợp so sánh các giải thuật tìm kiếm

| Giải thuật | Tính đầy đủ (Complete) | Tính tối ưu (Optimal) | Độ phức tạp thời gian | Độ phức tạp bộ nhớ | Cấu trúc dữ liệu chính |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BFS** | Có (nếu $b < \infty$) | Có (khi $c=1$) | $\mathcal{O}(b^d)$ | $\mathcal{O}(b^d)$ | Hàng đợi FIFO |
| **DFS** | Không (trong đồ thị vô hạn) | Không | $\mathcal{O}(b^m)$ | $\mathcal{O}(b \cdot m)$ | Ngăn xếp LIFO |
| **IDS** | Có (nếu $b < \infty$) | Có (khi $c=1$) | $\mathcal{O}(b^d)$ | $\mathcal{O}(b \cdot d)$ | Ngăn xếp đệ quy |
| **UCS** | Có (nếu $c \ge \epsilon$) | **Luôn tối ưu** | $\mathcal{O}(b^{1 + \lfloor C^*/\epsilon \rfloor})$ | $\mathcal{O}(b^{1 + \lfloor C^*/\epsilon \rfloor})$ | Hàng đợi ưu tiên theo $g(n)$ |
| **Greedy** | Không | Không | $\mathcal{O}(b^m)$ | $\mathcal{O}(b^m)$ | Hàng đợi ưu tiên theo $h(n)$ |
| **A\*** | Có | **Tối ưu** (khi $h$ admissible) | $\mathcal{O}(b^{\Delta \cdot d})$ | $\mathcal{O}(b^d)$ | Hàng đợi ưu tiên theo $g(n)+h(n)$ |
| **MCTS** | Có (khi $N \to \infty$) | Tối ưu tiệm cận | Tùy số vòng mô phỏng | Phụ thuộc kích thước cây | Cây UCT |

---

## 2.8 Tài liệu tham khảo (References)

1. **Russell, S., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. (Chương 2: Intelligent Agents; Chương 3: Solving Problems by Searching; Chương 4: Search in Complex Environments).
2. **Mitchell, T. M. (1997).** *Machine Learning*. McGraw-Hill. [Tham khảo: `resources/Machine Learning - Tom Mitchell.pdf`] (Chương 1: Khái niệm không gian giả thuyết và tìm kiếm).
3. **Hart, P. E., Nilsson, N. J., & Raphael, B. (1968).** A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107. (Công trình gốc khai sinh thuật toán A\*).
4. **Silver, D., Huang, A., Maddison, C. J., et al. (2016).** Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489. (Bài báo nền tảng về AlphaGo và MCTS).
5. **Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023).** Tree of Thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.
6. **Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).** Optimization by simulated annealing. *Science*, 220(4598), 671-680.

---

## 2.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta cài đặt và trực quan hóa bài toán kinh điển: **Tìm đường thông minh trong Mê cung 2D (Grid Maze Pathfinding) & Tối ưu hóa Hành trình**:

1. **Thành phần 1 - Xây dựng Môi trường Mê cung (GridWorld Maze):**
   - Biểu diễn bản đồ lưới $20 \times 20$ với các chướng ngại vật ngẫu nhiên, điểm xuất phát $S$ và điểm đích $G$.
   - Chi phí bước đi biến đổi: Địa hình thông thường ($c=1$), địa hình bùn lầy sình lầy ($c=5$).
2. **Thành phần 2 - Cài đặt các thuật toán Tìm kiếm Mù từ đầu:**
   - **BFS (Breadth-First Search):** Tìm đường theo số bước ít nhất.
   - **UCS (Uniform-Cost Search):** Tìm đường tối ưu theo chi phí năng lượng tích lũy $g(n)$.
3. **Thành phần 3 - Cài đặt Thuật toán Heuristic A\*:**
   - Cài đặt hai hàm Heuristic: Khoảng cách Manhattan ($h_{\text{manhattan}}$) và Khoảng cách Euclidean ($h_{\text{euclidean}}$).
   - Kiểm chứng thực nghiệm tính ưu việt của A\* so với BFS/UCS về số lượng nút cần mở rộng (Explored Nodes).
4. **Thành phần 4 - Tìm kiếm cục bộ Luyện kim mô phỏng (Simulated Annealing):**
   - Áp dụng giải bài toán Người du lịch ghé thăm $K$ trạm tiếp nhiên liệu trong mê cung (TSP on Grid).
   - Trực quan hóa đường cong hạ nhiệt $T$ và sự cải thiện của hàm năng lượng theo thời gian.
5. **Thành phần 5 - Mô phỏng cơ chế Beam Search:**
   - Cài đặt mô phỏng giải mã chuỗi với độ rộng chùm $K=3$ để so sánh với giải thuật tham lam (Greedy).
6. **Trực quan hóa và Bảng so sánh định lượng:**
   - Vẽ bản đồ mê cung kèm đường đi tìm được của BFS, UCS, A\*.
   - Đồ thị cột so sánh: Độ dài đường đi, Tổng chi phí năng lượng, Số nút đã duyệt (Explored States).
