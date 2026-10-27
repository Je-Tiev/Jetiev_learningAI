# Day 03: t-SNE — Visualizing Data using t-SNE

> **Bài báo gốc:** *Visualizing Data using t-SNE*  
> **Tác giả:** Laurens van der Maaten & Geoffrey Hinton  
> **Xuất bản:** Tạp chí *Journal of Machine Learning Research (JMLR)*, Tập 9, Tháng 11 năm 2008, Trang 2579–2605.  
> **Tài liệu PDF gốc:** [`resources/Day 3 Visualizing Data using t-SNE.pdf`](./resources/Day%203%20Visualizing%20Data%20using%20t-SNE.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Bài Toán Trực Quan Hóa Đa Tạp](#2-bối-cảnh-lịch-sử--bài-toán-trực-quan-hóa-đa-tạp)
3. [Kiến Trúc & Cơ Chế Xác Suất Của t-SNE](#3-kiến-trúc--cơ-chế-xác-suất-của-t-sne)
4. [Đột Phá Kỹ Thuật: Đối Xứng Hóa & Phân Phối t-Student](#4-đột-phá-kỹ-thuật-đối-xứng-hóa--phân-phối-t-student)
5. [Cơ Chế Hoạt Động & Phân Tích Giải Thuật Gradient](#5-cơ-chế-hoạt-động--phân-tích-giải-thuật-gradient)
6. [Quy Trình Huấn Luyện & Kỹ Thuật Tối Ưu Thực Chiến](#6-quy-trình-huấn-luyện--kỹ-thuật-tối-ưu-thực-chiến)
7. [Kết Quả Thực Nghiệm Trong Bài Báo](#7-kết-quả-thực-nghiệm-trong-bài-báo)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Trực quan hóa dữ liệu nhiều chiều (High-dimensional Data Visualization) là một trong những thách thức cốt lõi của Khoa học Dữ liệu và Học máy. Trước năm 2008, các nhà khoa học gần như bế tắc khi muốn hiển thị trực quan các cụm dữ liệu phức tạp trên mặt phẳng 2D hoặc 3D:
- **PCA (Principal Component Analysis)** chỉ tìm kiếm siêu phẳng tuyến tính, làm đè bẹp các đa tạp cong uốn lượn.
- **Isomap** và **LLE (Locally Linear Embedding)** bảo toàn cấu trúc hình học tốt hơn nhưng thất bại thảm hại khi dữ liệu phân bố trên các cụm rời rạc hoặc đa tạp có độ cong lớn.
- **SNE (Stochastic Neighbor Embedding - Hinton & Roweis, 2002)** mở ra hướng tiếp cận xác suất đột phá nhưng lại mắc phải "Vấn nạn tập trung quá mức" (**Crowding Problem**) khiến các cụm dữ liệu bị nén chặt vào tâm và không thể tách rời.

Năm 2008, **Laurens van der Maaten** (hiện là Giám đốc Nghiên cứu tại Meta FAIR) cùng với **Geoffrey Hinton** đã công bố thuật toán **t-SNE (t-Distributed Stochastic Neighbor Embedding)** trên *Journal of Machine Learning Research*.

```text
               Không gian dữ liệu gốc D-chiều (D = 784, 1024, ...)
                        [ Điểm dữ liệu x_i, x_j ]
                                   │
                                   ▼
             Đổi khoảng cách hình học thành Phân phối Xác suất
                  P_ij : Xác suất x_i chọn x_j làm láng giềng
                                   │
                                   ▼
           Chiếu xuống không gian hiển thị 2D/3D (y_i, y_j)
            Q_ij : Dùng phân phối t-Student (Heavy-tailed)
                                   │
                                   ▼
       Tối thiểu hóa khoảng cách KL: KL(P || Q) -> Tách cụm hoàn hảo!
```

### Đóng góp cốt lõi của nghiên cứu:
1. **Giải quyết triệt để Crowding Problem**: Sử dụng phân phối t-Student với 1 bậc tự do (phân phối Cauchy với phần đuôi dày - heavy-tailed) ở không gian ít chiều, tạo ra không gian đẩy mạnh mẽ giúp các cụm tách bạch rõ rệt.
2. **Symmetric SNE (SNE Đối xứng)**: Đơn giản hóa việc tính toán xác suất liên hiệp và đạo hàm gradient, giúp thuật toán ổn định và hội tụ nhanh hơn nhiều so với SNE nguyên bản.
3. **Tiêu chuẩn vàng cho trực quan hóa học máy**: Trở thành công cụ mặc định trong suốt hơn 15 năm qua để kiểm tra biểu diễn latent space của Deep Learning, phân cụm tế bào đơn (Single-cell RNA sequencing) và phân tích cụm văn bản/hình ảnh.

---

## 2. Bối Cảnh Lịch Sử & Bài Toán Trực Quan Hóa Đa Tạp

### 2.1. Lời nguyền số chiều & Giả thuyết đa tạp
Trong các bài toán thị giác máy tính, xử lý ngôn ngữ hay sinh học phân tử, số chiều $D$ của dữ liệu có thể lên tới hàng trăm hoặc hàng nghìn. Tuy nhiên, theo **Giả thuyết Đa tạp (Manifold Hypothesis)**, dữ liệu thực tế chỉ nằm trên một đa tạp con phi tuyến tính có số chiều nội tại $d$ nhỏ hơn rất nhiều ($d \ll D$). Mục tiêu của trực quan hóa là ánh xạ các điểm từ không gian $\mathbb{R}^D$ xuống không gian $\mathbb{R}^2$ hoặc $\mathbb{R}^3$ sao cho:
- **Cấu trúc cục bộ (Local structure)** được bảo toàn: Các điểm gần nhau trong không gian gốc phải ở gần nhau trên màn hình.
- **Cấu trúc toàn cục (Global structure)** được thể hiện: Các cụm dữ liệu khác nhau phải được tách biệt rõ ràng.

### 2.2. Vấn nạn tập trung quá mức (The Crowding Problem)
Tại sao việc ánh xạ từ không gian nhiều chiều xuống 2 chiều lại cực kỳ khó khăn?
- Xét một hình cầu bán kính $r$ trong không gian $D$ chiều, thể tích của nó tỷ lệ thuận với $r^D$.
- Khi $D$ lớn, **hầu hết thể tích của hình cầu nằm tập trung sát bề mặt** của nó!
- Giả sử có $N$ điểm dữ liệu phân bố đều quanh một điểm trung tâm trong không gian 10 chiều. Khoảng cách giữa các điểm lân cận tương đối lớn.
- Khi ta cố gắng ép các điểm này vào không gian 2 chiều, **mặt phẳng 2 chiều đơn giản là không có đủ diện tích** để chứa tất cả các điểm đó ở khoảng cách tương ứng!
- Hậu quả: Trong các phương pháp cũ (như SNE nguyên bản), các điểm ở cự ly trung bình bị dồn ứ lại, đè bẹp lên nhau và tạo thành một "cục bông" dày đặc ở trung tâm biểu đồ.

```text
Không gian 10D: Có thể chứa hàng trăm láng giềng ở khoảng cách r mà không chồng lấn
                          ○
                       ○     ○
                    ○     ●     ○   (Rất nhiều không gian trống)
                       ○     ○
                          ○

Ánh xạ xuống 2D (SNE thông thường):
                    ○○○○○○○○○○
                    ○○○●●●●○○○    <--- Các điểm bị ép chặt vào tâm (Crowding Problem),
                    ○○○○○○○○○○         không thể phân biệt được các cụm!
```

---

## 3. Kiến Trúc & Cơ Chế Xác Suất Của t-SNE

Khác với PCA bảo toàn phương sai hay Multidimensional Scaling (MDS) bảo toàn khoảng cách Euclidean, t-SNE chuyển đổi toàn bộ bài toán hình học sang **bài toán khớp phân phối xác suất (Probability Distribution Matching)**.

### 3.1. Phân phối xác suất trong không gian nhiều chiều ($P$)
Trong không gian gốc $\mathbb{R}^D$, độ tương đồng giữa hai điểm $x_i$ và $x_j$ được mô hình hóa bằng xác suất có điều kiện $p_{j|i}$: xác suất để $x_i$ chọn $x_j$ làm láng giềng theo phân phối chuẩn Gaussian:

$$p_{j|i} = \frac{\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)}{\sum_{k \neq i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)}, \quad p_{i|i} = 0$$

- $\sigma_i$ là phương sai của phân phối Gaussian tập trung tại $x_i$.
- Điểm đặc biệt: $\sigma_i$ không cố định mà được tính riêng cho từng điểm $x_i$ thông qua tham số **Perplexity (Độ hỗn loạn)**.

#### Cơ chế Perplexity & Tìm kiếm nhị phân phương sai $\sigma_i$:
Perplexity có thể hiểu nôm na là số lượng láng giềng hiệu dụng mà mỗi điểm tương tác:
$$\text{Perp}(P_i) = 2^{H(P_i)}$$
Trong đó $H(P_i)$ là entropy Shannon của phân phối $P_i$:
$$H(P_i) = -\sum_{j} p_{j|i} \log_2 p_{j|i}$$
Thuật toán sử dụng **tìm kiếm nhị phân (binary search)** để tìm giá trị $\sigma_i$ chính xác sao cho $\text{Perp}(P_i)$ bằng giá trị Perplexity do người dùng thiết lập (thường từ $5$ đến $50$). 
> *Ý nghĩa:* Tại các vùng dữ liệu dày đặc, $\sigma_i$ tự động thu nhỏ lại; tại các vùng dữ liệu thưa thớt, $\sigma_i$ tự động mở rộng ra.

### 3.2. Xác suất liên hiệp đối xứng (Symmetric Probability)
Để tránh các điểm dị biệt (outliers) làm hỏng gradient và giúp công thức đối xứng thuận tiện, t-SNE định nghĩa xác suất đồng thời:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}, \quad p_{ii} = 0$$

Điều này đảm bảo $\sum_{i, j} p_{ij} = 1$ và $p_{ij} = p_{ji}$.

### 3.3. Phân phối xác suất trong không gian ít chiều ($Q$)
Trong không gian hiển thị $\mathbb{R}^2$ hoặc $\mathbb{R}^3$, tọa độ tương ứng của các điểm là $y_i, y_j$. Thay vì dùng phân phối Gaussian, t-SNE sử dụng **phân phối t-Student với 1 bậc tự do (phân phối Cauchy)**:

$$q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\sum_{k} \sum_{l \neq k} \left(1 + \|y_k - y_l\|^2\right)^{-1}}, \quad q_{ii} = 0$$

---

## 4. Đột Phá Kỹ Thuật: Đối Xứng Hóa & Phân Phối t-Student

### 4.1. Sức mạnh của phân phối t-Student (Đuôi Dày - Heavy Tails)
Sự khác biệt cốt lõi giữa phân phối Gaussian và phân phối t-Student:
- Phân phối Gaussian suy giảm cực kỳ nhanh theo quy luật hàm mũ: $\exp(-d^2)$.
- Phân phối t-Student suy giảm theo quy luật nghịch đảo bình phương: $(1 + d^2)^{-1}$.

```text
Mật độ xác suất
  ▲
  │       Gaussian (Đuôi mỏng - suy giảm nhanh về 0)
  │          ╭───╮
  │         │     │         t-Student (Đuôi dày - Heavy tail)
  │        │       │       ╭───────╮
  │       │         │    ╭─╯       ╰─╮
  │───────╯─────────╰───╭╯           ╰╮──────────► Khoảng cách d
```

#### Cơ chế hóa giải Crowding Problem:
- Giả sử hai điểm $x_i, x_j$ có khoảng cách trung bình trong không gian nhiều chiều, tương ứng với xác suất $p_{ij} = 0.05$.
- Trong không gian 2D, để đạt được xác suất $q_{ij} = 0.05$:
  - Nếu dùng Gaussian: $y_i$ và $y_j$ phải nằm rất gần nhau.
  - Nếu dùng t-Student: Vì phần đuôi của t-Student dày hơn nhiều, $y_i$ và $y_j$ **có thể nằm cách xa nhau đáng kể** mà vẫn đạt được xác suất $q_{ij} = 0.05$!
- **Hệ quả kỳ diệu:** Các điểm thuộc các cụm khác nhau được "cho phép" đứng cách xa nhau trên mặt phẳng 2D mà không phải chịu chi phí phạt quá mức. Các cụm tự do bung tỏa và tách biệt rõ ràng!

---

## 5. Cơ Chế Hoạt Động & Phân Tích Giải Thuật Gradient

### 5.1. Hàm mất mát Kullback-Leibler Divergence
t-SNE tối ưu hóa vị trí các điểm $y_i$ sao cho phân phối $Q$ phản ánh trung thực nhất phân phối $P$. Độ sai khác giữa hai phân phối được đo bằng **Kullback-Leibler (KL) Divergence**:

$$C = KL(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

#### Đặc tính bất đối xứng của hàm KL:
- Nếu $p_{ij}$ lớn (hai điểm gần nhau ở không gian gốc) nhưng $q_{ij}$ nhỏ (bị đặt xa nhau ở 2D): Số hạng $p_{ij} \log(p_{ij} / q_{ij})$ **rất lớn** $\implies$ **Hình phạt cực nặng!** Mô hình bắt buộc phải kéo chúng lại gần nhau.
- Nếu $p_{ij}$ nhỏ (hai điểm xa nhau ở không gian gốc) nhưng $q_{ij}$ lớn (bị đặt gần nhau ở 2D): Hình phạt tương đối nhỏ.
- **Kết luận:** t-SNE cực kỳ ưu tiên **bảo toàn tính cục bộ** (cụm láng giềng gần gũi).

### 5.2. Đạo hàm giải tích tuyệt đẹp của t-SNE
Nhờ sử dụng Symmetric SNE kết hợp phân phối t-Student, đạo hàm của hàm mất mát $C$ theo tọa độ $y_i$ có dạng đơn giản và trực quan bất ngờ:

$$\frac{\partial C}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij})(y_i - y_j)\left(1 + \|y_i - y_j\|^2\right)^{-1}$$

### 5.3. Trực giác vật lý: Hệ thống Lò xo Đẩy - Hút
Đạo hàm trên có thể diễn giải như tổng hợp lực tác động lên chất điểm $y_i$ bởi tất cả các chất điểm $y_j$ khác thông qua các "lò xo ảo":

```text
                          Lực hút (Attraction)
                    (p_ij > q_ij) : Kéo y_i về phía y_j
                              y_i ───► ◄─── y_j

                          Lực đẩy (Repulsion)
                    (p_ij < q_ij) : Đẩy y_i ra xa y_j
                              y_i ◄─── ───► y_j
```

1. **Lực hút (Attraction):** Tỷ lệ thuận với $p_{ij}(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$. Chỉ có tác dụng mạnh khi $x_i, x_j$ là láng giềng gần ở không gian gốc.
2. **Lực đẩy (Repulsion):** Tỷ lệ thuận với $q_{ij}(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$. Tác dụng giữa tất cả các cặp điểm để chống co cụm và duy trì khoảng cách không gian.

---

## 6. Quy Trình Huấn Luyện & Kỹ Thuật Tối Ưu Thực Chiến

Thuật toán t-SNE sử dụng phương pháp giảm gradient kết hợp các thủ thuật tối ưu then chốt:

```text
BƯỚC 1: TÍNH P                  BƯỚC 2: KHỞI TẠO              BƯỚC 3: TỐI ƯU HÓA
Tính khoảng cách Euclidean      Khởi tạo Y ~ N(0, 10^-4)      Lặp T vòng:
Tìm sigma_i qua Perplexity      (hoặc dùng PCA 2D)            - Tính Q_ij
Tính ma trận đối xứng P_ij                                     - Tính Gradient dC/dY
                                                               - Early Exaggeration (P * 12)
                                                               - Momentum & Gradient Update
                                                               - Trừ tâm (Zero-centering)
```

### 6.1. Early Exaggeration (Khuếch đại sớm)
- Trong $250$ vòng lặp đầu tiên, nhân toàn bộ các giá trị $p_{ij}$ với một hệ số khuếch đại $\alpha$ (thường là $4$ hoặc $12$).
- **Mục đích:** Ép các cụm dữ liệu tự thu nhỏ thành các "đảo" độc lập cách xa nhau, tạo ra các khoảng trống lớn trên mặt phẳng 2D để các cụm dễ dàng hoán đổi vị trí mà không bị kẹt vào cực tiểu địa phương.

### 6.2. Động lượng (Momentum) & Tốc độ học thích ứng
Cập nhật tọa độ theo quy tắc động lượng:

$$Y^{(t)} = Y^{(t-1)} + \eta \frac{\partial C}{\partial Y} + \alpha(t) \left(Y^{(t-1)} - Y^{(t-2)}\right)$$

- Trong giai đoạn đầu ($t < 250$): dùng động lượng nhỏ $\alpha(t) = 0.5$.
- Sau giai đoạn Early Exaggeration ($t \ge 250$): tăng động lượng lên $\alpha(t) = 0.8$.
- Tốc độ học $\eta$: Khuyến nghị $\eta = 200$ hoặc $\eta = \frac{N}{12}$ cho các tập dữ liệu lớn.

### 6.3. Khử trôi dạt (Zero-Centering)
Sau mỗi bước cập nhật gradient, toàn bộ tọa độ $Y$ được trừ đi giá trị trung bình của chúng ($\sum_i y_i = 0$). Điều này ngăn ngừa toàn bộ đám mây điểm bị trôi dạt ra xa khỏi gốc tọa độ.

---

## 7. Kết Quả Thực Nghiệm Trong Bài Báo

Bài báo kiểm chứng t-SNE trên nhiều tập dữ liệu đa dạng và so sánh trực tiếp với 7 thuật toán giảm chiều hàng đầu thời bấy giờ:

### 7.1. Chữ số viết tay MNIST (60,000 ảnh $28 \times 28$)
- **PCA**: Đè lẫn các chữ số vào một khối mờ duy nhất, chỉ phân tách được sơ sài số 1 và số 0.
- **Isomap & LLE**: Bị vặn xoắn và chồng lấn giữa các lớp chữ số có nét tương đồng (như 3, 5, 8 hoặc 4, 7, 9).
- **Sammon Mapping**: Gặp Crowding Problem nặng, các chữ số bị nén nghẽn ở tâm.
- **t-SNE**: **Tách biệt gần như hoàn hảo 10 cụm chữ số** thành các hòn đảo riêng biệt, rõ ràng với biên giới tự nhiên. Thậm chí các biến thể viết tay khác nhau của cùng một chữ số (ví dụ: số 1 gạch ngang vs 1 thẳng đứng) cũng hình thành các nhánh con sắc nét!

### 7.2. Khuôn mặt Olivetti Faces (400 ảnh)
- t-SNE gom nhóm thành công các bức ảnh của cùng một người bất kể sự thay đổi về góc nghiêng hay đeo kính.

### 7.3. Tập vật thể đa góc COIL-20
- COIL-20 gồm 20 vật thể quay $360^\circ$ (tạo thành các đa tạp vòng tròn khép kín 1D trong không gian nhiều chiều).
- t-SNE khôi phục xuất sắc **20 vòng tròn lặp kín (closed loops)** độc lập trên mặt phẳng 2D, chứng minh khả năng bảo toàn topo học phi tuyến vượt bậc.

---

## 8. Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Các hạn chế của t-SNE nguyên bản
1. **Độ phức tạp tính toán cao $O(N^2)$**: Tính toán toàn bộ ma trận khoảng cách giữa tất cả các cặp điểm khiến t-SNE nguyên bản không thể chạy trực tiếp trên các tập dữ liệu $> 10,000$ mẫu.
2. **Không có tính chất quy nạp (Non-parametric)**: t-SNE không học một hàm ánh xạ $f(x) \to y$. Khi có điểm dữ liệu mới đến, không thể chiếu điểm đó vào bản đồ 2D sẵn có mà phải chạy lại toàn bộ thuật toán từ đầu.
3. **Khoảng cách giữa các cụm xa không có ý nghĩa**: t-SNE chỉ bảo toàn cấu trúc láng giềng cục bộ. Khoảng cách giữa hai cụm xa nhau trên biểu đồ t-SNE không nhất thiết phản ánh khoảng cách thực giữa chúng trong không gian gốc.

### 8.2. Sự tiến hóa của t-SNE và các thuật toán kế cận
| Thuật toán | Năm | Cải tiến then chốt | Ưu điểm & Ứng dụng |
|---|:---:|---|---|
| **Barnes-Hut t-SNE** | 2014 | Dùng cấu trúc cây Quadtree chia không gian xấp xỉ lực đẩy | Giảm độ phức tạp từ $O(N^2)$ xuống $O(N \log N)$, xử lý hàng trăm nghìn mẫu |
| **FIt-SNE (Fast Fourier Transform t-SNE)** | 2019 | Dùng phép biến đổi Fourier nhanh để nội suy lực đẩy trên lưới đều | Tăng tốc gấp hàng chục lần so với Barnes-Hut, chuẩn mực cho Single-Cell Genomics |
| **Parametric t-SNE** | 2009 | Dùng mạng nơ-ron sâu học hàm ánh xạ $y = f_\theta(x)$ | Chiếu tức thì các điểm dữ liệu mới vào không gian trực quan hóa |
| **UMAP (Uniform Manifold Approximation & Projection)** | 2018 | Dựa trên lý thuyết topo Riemann và đồ thị mờ (Fuzzy Simplicial Sets) | Tốc độ nhanh hơn, bảo toàn cấu trúc toàn cục (global structure) tốt hơn t-SNE |

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục này cung cấp 2 tệp mã nguồn độc lập, tự vận hành:

```text
Day 03 - Visualizing Data using t-SNE/
├── README.md                  # Tài liệu phân tích lý thuyết chi tiết này
├── tsne_scratch.py            # Cài đặt thuần Python & NumPy (Binary Search Perplexity, Gradient, Early Exaggeration)
├── tsne_scratch.ipynb         # Notebook thực hành tương tác trực quan (From Scratch)
├── tsne_pytorch.py            # Triển khai tăng tốc bằng PyTorch (Module autograd, so sánh PCA, xuất biểu đồ)
├── tsne_pytorch.ipynb         # Notebook thực hành tương tác & trực quan hóa PyTorch
└── resources/
    └── Day 3 Visualizing Data using t-SNE.pdf
```

### 9.1. Tệp `tsne_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python + NumPy chuẩn (không dùng Scikit-Learn hay PyTorch).
- **Tính năng nổi bật**:
  - Tự cài đặt thuật toán **Binary Search** tìm phương sai $\sigma_i$ khớp với Perplexity mục tiêu.
  - Tự động chuẩn hóa ma trận xác suất đối xứng $P$ và phân phối $Q$ theo t-Student.
  - Tính toán đạo hàm giải tích $\frac{\partial C}{\partial y}$ hoàn toàn bằng các phép toán vector hóa NumPy.
  - Tích hợp **Early Exaggeration ($\alpha = 4$)**, Momentum chuyển pha từ $0.5 \to 0.8$, và Zero-centering.
  - Tự sinh dữ liệu kiểm thử nhân tạo và hiển thị tiến trình hội tụ chi tiết trên màn hình terminal.

### 9.2. Tệp `tsne_pytorch.py` (Triển khai hiện đại bằng PyTorch)
- **Công nghệ**: PyTorch, torchvision, matplotlib, scikit-learn.
- **Tính năng nổi bật**:
  - Xây dựng lớp `PyTorchTSNE` tận dụng đồ thị tính toán động của PyTorch.
  - Hỗ trợ nạp tập dữ liệu chữ số MNIST / Digits tự động, kèm **bộ sinh dữ liệu Fallback an toàn khi chạy offline**.
  - So sánh trực quan song song giữa **PCA Baseline** và **t-SNE Embedding**.
  - Tự động xuất hình ảnh kết quả so sánh độ phân giải cao vào thư mục `outputs/tsne_comparison.png`.

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Tại sao việc thay thế phân phối Gaussian bằng phân phối t-Student ở không gian ít chiều lại giải quyết được triệt để hiện tượng Crowding Problem?**
2. **Trong t-SNE, tham số Perplexity có ý nghĩa bản chất là gì? Điều gì sẽ xảy ra nếu ta đặt Perplexity quá nhỏ (ví dụ: = 2) hoặc quá lớn (gần bằng số lượng mẫu $N$)?**
3. **Kỹ thuật Early Exaggeration hoạt động như thế nào và tại sao nó lại giúp các cụm dữ liệu phân tách rõ rệt hơn trên bản đồ 2D?**
4. **Tại sao hàm mất mát KL-Divergence trong t-SNE lại có xu hướng bảo toàn cấu trúc láng giềng cục bộ (Local structure) mạnh hơn nhiều so với cấu trúc toàn cục (Global structure)?**
5. **So sánh sự khác nhau cốt lõi giữa t-SNE và UMAP về mặt cơ sở toán học và khả năng bảo toàn cấu trúc dữ liệu?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
