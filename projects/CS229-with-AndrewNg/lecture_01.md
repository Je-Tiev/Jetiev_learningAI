# Lecture 01: Giới thiệu về Machine Learning (Introduction)

**Khóa học:** Stanford CS229 - Machine Learning (Autumn 2018)  
**Giảng viên:** Andrew Ng  

---

## 1. Tổng quan và Mục tiêu buổi học

Bài giảng đầu tiên tập trung vào việc giới thiệu tổng quan về khóa học, các quy định về bài tập/dự án cuối kỳ, định nghĩa cốt lõi về Machine Learning (Học máy), và phân loại các nhánh chính trong Machine Learning thông qua các ví dụ thực tiễn.

---

## 2. Machine Learning là gì?

Andrew Ng đưa ra hai định nghĩa kinh điển:

1. **Định nghĩa của Arthur Samuel (1959):**
   > *"Machine Learning là lĩnh vực nghiên cứu giúp máy tính có khả năng học hỏi mà không cần được lập trình một cách tường minh (without being explicitly programmed)."*
   * **Ví dụ:** Arthur Samuel đã viết một chương trình chơi cờ đam (Checkers). Thay vì lập trình cứng mọi nước đi, ông để chương trình tự chơi hàng chục nghìn ván với chính nó để tự nhận diện nước đi nào dẫn đến chiến thắng. Kết quả là chương trình chơi giỏi hơn cả chính tác giả.

2. **Định nghĩa hiện đại của Tom Mitchell (1998):**
   > *"Một chương trình máy tính được coi là học từ kinh nghiệm $E$ đối với một lớp tác vụ $T$ và thước đo hiệu suất $P$, nếu hiệu suất của nó ở tác vụ $T$, đo bằng $P$, được cải thiện thông qua kinh nghiệm $E$."*
   * **Ví dụ với bài toán lọc thư rác (Spam Filter):**
     * **Tác vụ ($T$):** Phân loại email là thư rác (spam) hay không (not spam).
     * **Kinh nghiệm ($E$):** Quan sát người dùng đánh dấu các email là spam hoặc không phải spam.
     * **Thước đo hiệu suất ($P$):** Tỷ lệ phần trăm email được phân loại chính xác.

---

## 3. Phân loại các bài toán trong Machine Learning

Nội dung chính của CS229 xoay quanh 3 nhánh lớn (cùng với phần Lý thuyết học & Lời khuyên thực hành):

### 3.1. Học có giám sát (Supervised Learning)
Là bài toán mà dữ liệu huấn luyện bao gồm cả đầu vào ($x$) và nhãn đầu ra đúng ($y$) tương ứng. Mục tiêu là học một hàm ánh xạ $h: X \rightarrow Y$ để dự đoán đầu ra cho dữ liệu mới.

* **Bài toán Hồi quy (Regression):** Biến mục tiêu $y$ cần dự đoán là một giá trị **liên tục** ($y \in \mathbb{R}$).
  * *Ví dụ:* Dự đoán giá nhà dựa trên diện tích, số phòng ngủ, vị trí.
* **Bài toán Phân loại (Classification):** Biến mục tiêu $y$ nhận một số lượng **rời rạc** các giá trị (ví dụ: $y \in \{0, 1\}$ cho phân loại nhị phân, hoặc $y \in \{0, 1, \dots, k\}$ cho phân loại đa lớp).
  * *Ví dụ:* Chẩn đoán khối u vú là lành tính ($0$) hay ác tính ($1$) dựa trên kích thước khối u và độ tuổi bệnh nhân.
  * *Mở rộng:* Trong thực tế, dữ liệu có thể có vô số đặc trưng (infinite-dimensional features). Để xử lý được không gian vô hạn chiều này, khóa học sẽ giới thiệu thuật toán **Support Vector Machines (SVM)** và **Kernel Trick**.

### 3.2. Học không giám sát (Unsupervised Learning)
Là bài toán mà dữ liệu đầu vào chỉ có $x$, hoàn toàn **không có nhãn $y$**. Thuật toán phải tự tìm ra cấu trúc ẩn (structure) bên trong dữ liệu.

* **Phân cụm (Clustering):** Gom nhóm các điểm dữ liệu có tính chất tương đồng.
  * *Ví dụ:* Google News tự động gom các bài báo cùng chủ đề; phân khúc khách hàng trong marketing; phân tích biểu hiện gen (microarray data).
* **Bài toán nguồn âm thanh hỗn hợp (Cocktail Party Problem):**
  * *Mô tả:* Đặt 2 micro ở 2 vị trí khác nhau trong một căn phòng có 2 người đang nói chuyện cùng lúc. Mỗi micro thu được một bản ghi âm trộn lẫn giọng của cả hai người với cường độ khác nhau.
  * *Giải pháp:* Thuật toán **ICA (Independent Component Analysis)** giúp tách riêng giọng nói của từng người ra khỏi tín hiệu hỗn hợp.

### 3.3. Học tăng cường (Reinforcement Learning)
Áp dụng cho các bài toán ra quyết định tuần tự (sequential decision making) theo thời gian, nơi không có đáp án "đúng/sai" ngay lập tức cho từng bước đi.

* **Cơ chế:** Tác tử (Agent) tương tác với môi trường (Environment), thực hiện hành động (Action), chuyển sang trạng thái mới (State) và nhận về phần thưởng (Reward) hoặc hình phạt.
* **Ví dụ:**
  * Lái trực thăng tự động nhào lộn (Stanford Autonomous Helicopter): Không có "góc nghiêng chuẩn" cho từng mili-giây, mà chỉ có phần thưởng khi bay đẹp và hình phạt cực nặng khi rơi máy bay.
  * Chơi cờ vua, cờ vây (AlphaGo), điều khiển robot.

---

## 4. Ký hiệu toán học chuẩn (Notation) sử dụng trong CS229

Để thống nhất xuyên suốt khóa học, các ký hiệu sau được quy ước:

* $m$: Số lượng mẫu huấn luyện (number of training examples).
* $x$: Biến đầu vào / đặc trưng (input variables / features), thường là vector $x \in \mathbb{R}^n$.
* $y$: Biến đầu ra / biến mục tiêu (output / target variable).
* $(x, y)$: Một mẫu huấn luyện đơn lẻ.
* $(x^{(i)}, y^{(i)})$: Mẫu huấn luyện thứ $i$ trong tập dữ liệu ($i = 1, \dots, m$). *(Lưu ý: chỉ số trên ngoặc đơn không phải là lũy thừa)*.
* $x_j^{(i)}$: Giá trị của đặc trưng thứ $j$ trong mẫu huấn luyện thứ $i$.
* $n$: Số lượng đặc trưng (number of features).
* $\mathcal{X}$: Không gian của các giá trị đầu vào (thường là $\mathbb{R}^n$).
* $\mathcal{Y}$: Không gian của các giá trị đầu ra.
* $h$: Hàm giả thuyết (Hypothesis), ánh xạ $h: \mathcal{X} \rightarrow \mathcal{Y}$.

---

## 5. Câu hỏi ôn tập Lecture 1

1. Sự khác biệt cốt lõi giữa bài toán Regression và Classification trong Supervised Learning là gì?
2. Tại sao bài toán Cocktail Party lại được xếp vào nhóm Unsupervised Learning?
3. Trong ký hiệu $x_3^{(5)}$, số $5$ và số $3$ lần lượt đại diện cho đại lượng nào?