# Day 04: ImageNet — A Large-Scale Hierarchical Image Database

> **Bài báo gốc:** *ImageNet: A Large-Scale Hierarchical Image Database*  
> **Tác giả:** Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei  
> **Xuất bản:** Hội nghị *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Miami, Florida, 2009, Trang 248–255.  
> **Tài liệu PDF gốc:** [`resources/Day 4 ImageNet.pdf`](./resources/Day%204%20ImageNet.pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Nền Tảng Cây Phả Hệ WordNet](#2-bối-cảnh-lịch-sử--nền-tảng-cây-phả-hệ-wordnet)
3. [Cấu Trúc Cơ Sở Dữ Liệu & Quy Trình Thu Thập - Xác Thực](#3-cấu-trúc-cơ-sở-dữ-liệu--quy-trình-thu-thập---xác-thực)
4. [Đặc Tính Kỹ Thuật: Quy Mô, Tính Đa Dạng & Cấu Trúc Phân Cấp](#4-đặc-tính-kỹ-thuật-quy-mô-tính-đa-dạng--cấu-trúc-phân-cấp)
5. [Độ Đo Đánh Giá & Bài Toán Phân Loại Phân Cấp](#5-độ-đo-đánh-giá--bài-toán-phân-loại-phân-cấp)
6. [Cuộc Thi ILSVRC & Cú Hích Lịch Sử Khai Sinh Deep Learning](#6-cuộc-thi-ilsvrc--cú-hích-lịch-sử-khai-sinh-deep-learning)
7. [Kết Quả Thực Nghiệm Ban Đầu Trong Bài Báo CVPR 2009](#7-kết-quả-thực-nghiệm-ban-đầu-trong-bài-báo-cvpr-2009)
8. [Di Sản, Thách Thức & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-thách-thức--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)

---

## 1. Tổng Quan & Ý Nghĩa Lịch Sử

Vào những năm 2000, cộng đồng Thị giác Máy tính (Computer Vision) rơi vào một "điểm nghẽn nhận thức" nghiêm trọng:
- Hầu hết các nhà nghiên cứu tin rằng **mô hình và thuật toán** là chìa khóa duy nhất để giải quyết bài toán nhận diện vật thể. Hàng trăm thuật toán trích xuất đặc trưng thủ công phức tạp (SIFT, SURF, HOG) và các biến thể máy học hạt nhân (Kernel SVM) liên tục được phát minh.
- Tuy nhiên, các thuật toán này đều được thử nghiệm trên các tập dữ liệu cực kỳ nhỏ:
  - **Caltech-101 (2004)**: Chỉ gồm 101 lớp và khoảng 9,000 bức ảnh.
  - **Caltech-256 (2007)**: 256 lớp, khoảng 30,000 ảnh.
  - **PASCAL VOC**: Chỉ gồm 20 lớp vật thể với khoảng 10,000 ảnh.
- Kết quả là các mô hình gặp hiện tượng **quá khớp (overfitting)** trầm trọng. Chúng có thể phân biệt rất tốt giữa một con báo đốm chụp ngang góc chuẩn và một chiếc xe máy trên nền trắng, nhưng hoàn toàn sụp đổ khi gặp các hình ảnh thực tế phức tạp ngoài đời thực.

Năm 2006–2007, Giáo sư **Li Fei-Fei** (Lý Phi Phi) tại Đại học Princeton (sau này là Đại học Stanford) cùng nghiên cứu sinh **Jia Deng** và các cộng sự đã đưa ra một giả thuyết mang tính cách mạng:  
> *"Vấn đề không nằm ở thuật toán, mà nằm ở dữ liệu. Để một cỗ máy học cách nhìn thế giới, nó phải được tiếp xúc với sự phong phú và bao la của chính thế giới đó."*

```text
       TIẾP CẬN CŨ (Trước 2009)                     TIẾP CẬN MỚI: IMAGENET (2009)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
│ Dữ liệu nhỏ (10-100 lớp, vài nghìn ảnh)│     │ Dữ liệu khổng lồ (21,841 lớp, 14 triệu ảnh)   │
│                 +                    │ ==> │                      +                       │
│ Thuật toán phức tạp (Feature Eng.)   │     │ Mô hình sâu, tổng quát (Deep Neural Networks)│
│ ==> Overfitting, không dùng thực tế  │     │ ==> Khả năng tổng quát hóa phi thường!       │
└──────────────────────────────────────┘     └──────────────────────────────────────────────┘
```

Công bố tại CVPR 2009, **ImageNet** không chỉ là một bài báo về tập dữ liệu đơn thuần, mà là **bước chuyển dịch hệ tư tưởng (paradigm shift)** mở màn cho kỷ nguyên **AI Lấy Dữ Liệu Làm Trung Tâm (Data-Centric AI)**, trở thành bệ phóng trực tiếp cho sự bùng nổ của AlexNet (2012) và cuộc cách mạng Deep Learning toàn cầu.

---

## 2. Bối Cảnh Lịch Sử & Nền Tảng Cây Phả Hệ WordNet

Khác với các tập dữ liệu phân loại thông thường (vốn chỉ liệt kê một danh sách phẳng các nhãn rời rạc), ImageNet được xây dựng dựa trên cấu trúc ngữ nghĩa đồ sộ của **WordNet** (do Giáo sư tâm lý học George Miller khởi xướng tại Princeton từ năm 1985).

### 2.1. Khái niệm Synset (Tập từ đồng nghĩa)
- Trong WordNet, đơn vị ngữ nghĩa cơ bản là một **Synset (Synonym Set)** — một tập hợp các từ hoặc cụm từ đồng nghĩa biểu diễn cùng một khái niệm cụ thể.
- *Ví dụ:* Synset `{bank, dipping bank}` (bờ đất dốc cạnh bờ sông) hoàn toàn khác biệt với Synset `{bank, financial institution}` (ngân hàng tài chính).
- Việc gắn ảnh vào Synset thay vì từ khóa thuần túy giúp ImageNet giải quyết triệt để vấn đề **đa nghĩa (polysemy)** trong ngôn ngữ.

### 2.2. Quan hệ Cây Phả Hệ Is-A (Hypernym - Hyponym)
Các Synset trong WordNet được liên kết với nhau theo cấu trúc đồ thị có hướng không chu trình (DAG), trong đó nhánh chính là cây phả hệ quan hệ cha-con:
- **Hypernym (Từ bậc trên / Khái niệm cha)**: Mang ý nghĩa tổng quát hơn.
- **Hyponym (Từ bậc dưới / Khái niệm con)**: Mang ý nghĩa cụ thể hơn.

```text
                              [entity]
                                 │
                           [living thing]
                                 │
                            [organism]
                                 │
                             [animal]
                                 │
                            [vertebrate]
                                 │
                             [mammal]
                                 │
                            [carnivore]
                                 │
                             [canine]
                                 │
                          ┌──────┴──────┐
                       [dog]          [wolf]
                          │
                   ┌──────┴──────┐
             [husky]        [golden retriever]
```

Cấu trúc phân cấp này mang lại cho ImageNet một lợi thế chưa từng có: **Không chỉ phân loại nhị phân đúng/sai**, mà cho phép đo lường **khoảng cách ngữ nghĩa (Semantic Tree-Hop Distance)** giữa các phán đoán của mô hình.

---

## 3. Cấu Trúc Cơ Sở Dữ Liệu & Quy Trình Thu Thập - Xác Thực

Xây dựng một cơ sở dữ liệu với hàng chục triệu hình ảnh vào năm 2007–2009 là một nhiệm vụ tưởng chừng bất khả thi. Nếu thuê sinh viên đại học gán nhãn thủ công với tốc độ trung bình, nhóm nghiên cứu ước tính sẽ phải mất **hàng trăm năm**!

Đội ngũ ImageNet đã sáng tạo ra một quy trình tự động hóa kết hợp trí tuệ đám đông (Crowdsourcing) qua 3 giai đoạn:

```text
  GIAI ĐOẠN 1: THU THẬP             GIAI ĐOẠN 2: LÀM SẠCH          GIAI ĐOẠN 3: XÁC THỰC AMT
  Truy vấn nhiều công cụ tìm kiếm    Khử ảnh trùng lặp (Image Hash) Phân tán nhãn cho thợ AMT
  (Google, Flickr, Yahoo, Ask)      Loại bỏ ảnh rác/lỗi format     Thuật toán đa biểu quyết
  với từ khóa Synset đa ngôn ngữ                                   (Majority Voting / Bayes)
           │                                 │                                │
           ▼                                 ▼                                ▼
     ~1,000 ảnh/synset                Ảnh ứng viên sạch             Tập dữ liệu chuẩn xác cao
```

### 3.1. Giai đoạn 1: Khai phá dữ liệu web đa ngôn ngữ
- Với mỗi Synset trong WordNet (chỉ chọn các danh từ cụ thể biểu diễn được bằng hình ảnh - khoảng 80,000 danh từ), thuật toán dịch các từ vựng sang nhiều ngôn ngữ (Trung, Tây Ban Nha, Hà Lan, Đức,...).
- Gửi truy vấn tự động đến các công cụ tìm kiếm hình ảnh hàng đầu thời bấy giờ (Google Images, Flickr, Yahoo Image Search, Ask.com) để thu về khoảng 1,000–2,000 ảnh ứng viên cho mỗi Synset.

### 3.2. Giai đoạn 2: Lọc trùng lặp & Rác kỹ thuật
- Sử dụng thuật toán băm hình ảnh (Perceptual Image Hashing) và so sánh đặc trưng cục bộ để loại bỏ các bức ảnh sao chép hoặc trùng lặp giữa các truy vấn khác nhau.

### 3.3. Giai đoạn 3: Đột phá gán nhãn qua Amazon Mechanical Turk (AMT)
- **Amazon Mechanical Turk** vừa ra đời là một nền tảng cho phép phân chia công việc vi mô (micro-tasks) cho hàng trăm nghìn người lao động tự do trên khắp thế giới.
- Nhóm nghiên cứu đã thuê **gần 50,000 lao động từ 167 quốc gia** trên AMT để kiểm định ảnh.
- **Thách thức chất lượng:** Lao động trên AMT có thể bấm chọn ngẫu nhiên để kiếm tiền nhanh, dẫn đến dữ liệu gán nhãn sai lệch.
- **Giải pháp Kiểm định Đa biểu quyết (Confidence / Majority Voting):**
  - Mỗi bức ảnh ứng viên không chỉ đưa cho 1 người, mà được đưa cho nhiều người kiểm định độc lập: *"Bức ảnh này có chứa vật thể [Synset] hay không?"*
  - Thuật toán dừng thích ứng (Adaptive stopping algorithm): Nếu 3 người đầu tiên đều đồng thuận trả lời "Có", bức ảnh được chấp thuận ngay lập tức. Nếu có ý kiến bất đồng, bức ảnh tiếp tục được gửi cho 5, 7, hoặc 10 người đánh giá cho đến khi độ tin cậy hậu nghiệm (Posterior Probability) vượt qua ngưỡng chuẩn xác 99%.

---

## 4. Đặc Tính Kỹ Thuật: Quy Mô, Tính Đa Dạng & Cấu Trúc Phân Cấp

Khi hoàn thành phiên bản đầy đủ, ImageNet đạt được các thông số kỷ lục:

| Tiêu Chí | ImageNet (Toàn bộ) | Caltech-256 (2007) | PASCAL VOC (2008) |
|---|:---:|:---:|:---:|
| **Tổng số hình ảnh** | **14,197,122 ảnh** | ~30,607 ảnh | ~10,000 ảnh |
| **Số lượng Synsets / Lớp** | **21,841 synsets** | 256 lớp | 20 lớp |
| **Trung bình ảnh / lớp** | **~650 ảnh / synset** | ~119 ảnh / lớp | ~500 ảnh / lớp |
| **Cấu trúc dữ liệu** | **Cây phân cấp WordNet** | Danh sách phẳng | Danh sách phẳng |
| **Độ phủ khái niệm** | Hầu hết danh từ cụ thể | Hạn chế | Hạn chế |

```text
Cấu trúc tổ chức của ImageNet:
├── Subtree: Amphibian (Động vật lưỡng cư) - 94 synsets
├── Subtree: Bird (Chim) - 856 synsets
├── Subtree: Fish (Cá) - 986 synsets
├── Subtree: Mammal (Động vật có vú) - 1,190 synsets
├── Subtree: Plant (Thực vật) - 1,600 synsets
├── Subtree: Vehicle (Phương tiện giao thông) - 863 synsets
└── Subtree: Artifact (Đồ vật nhân tạo) - 7,422 synsets
```

---

## 5. Độ Đo Đánh Giá & Bài Toán Phân Loại Phân Cấp

### 5.1. Top-1 Error vs Top-5 Error
Trong các tập dữ liệu có số lượng lớp khổng lồ (như 1,000 lớp của thử thách ILSVRC), một bức ảnh có thể chứa nhiều vật thể cùng lúc, hoặc một giống chó rất khó phân biệt chính xác tuyệt đối ngay cả với mắt người chuyên gia (ví dụ: nhầm giữa Siberian Husky và Alaskan Malamute). Do đó, bài báo thiết lập chuẩn đánh giá kép:
- **Top-1 Error**: Tỷ lệ phần trăm các ảnh mà nhãn dự đoán có xác suất cao nhất của mô hình **không trùng khớp** với nhãn thực tế ($y_{\text{true}} \neq \hat{y}^{(1)}$).
- **Top-5 Error**: Mô hình được phép đưa ra 5 dự đoán có độ tự tin cao nhất $\{\hat{y}^{(1)}, \dots, \hat{y}^{(5)}\}$. Nếu nhãn thực tế nằm trong top 5 này, mô hình được tính là **đúng**:
  $$\text{Top-5 Error} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(y_i \notin \{\hat{y}_i^{(1)}, \hat{y}_i^{(2)}, \hat{y}_i^{(3)}, \hat{y}_i^{(4)}, \hat{y}_i^{(5)}\}\right)$$

### 5.2. Khoảng cách phân cấp (Tree-Hop Distance) & Chi phí ngữ nghĩa
Trong phân loại phẳng truyền thống (0/1 loss), việc nhầm một con *Husky* thành *Chó sói (Wolf)* bị trừng phạt ngang bằng với việc nhầm con *Husky* thành một *Chiếc xe tải (Truck)*. Điều này phi lý về mặt nhận thức!

ImageNet đề xuất đo lường lỗi phân loại dựa trên **khoảng cách nhảy cây (Tree-Hop Distance)** giữa nút nhãn thực tế $c_i$ và nút dự đoán $c_j$ thông qua **Tổ tiên chung gần nhất (Lowest Common Ancestor - LCA)**:

$$D_{\text{tree}}(c_i, c_j) = \text{Depth}(c_i) + \text{Depth}(c_j) - 2 \times \text{Depth}(\text{LCA}(c_i, c_j))$$

- Khoảng cách giữa *Husky* và *Alaskan Malamute*: $D_{\text{tree}} = 2$ (Chung nút cha *Dog*).
- Khoảng cách giữa *Husky* và *Truck*: $D_{\text{tree}} = 14$ (Chung nút gốc *Entity*).
- Hàm mất mát phân cấp (Hierarchical Loss) phạt tỷ lệ thuận với $D_{\text{tree}}$, khuyến khích mô hình "nếu có đoán sai, hãy đoán sai trong cùng họ hàng ngữ nghĩa gần".

---

## 6. Cuộc Thi ILSVRC & Cú Hích Lịch Sử Khai Sinh Deep Learning

Để thúc đẩy toàn bộ ngành AI bước vào sân chơi dữ liệu lớn, nhóm nghiên cứu đã tổ chức cuộc thi thường niên **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** từ năm 2010 đến 2017:
- Lấy một tập con chuẩn hóa gồm **1,000 lớp vật thể phong phú** với **1.2 triệu ảnh huấn luyện**, **50,000 ảnh kiểm định (validation)**, và **100,000 ảnh thử nghiệm (test)**.

```text
LỊCH SỬ TỶ LỆ LỖI TOP-5 TẠI THỬ THÁCH ILSVRC (2010 - 2017):
28.2% ──┐
        │  25.8% (Phương pháp truyền thống: SIFT + SVM)
        └───┐
            │  16.4% ──► [2012: AlexNet (Krizhevsky, Sutskever, Hinton)] <== CÁCH MẠNG DEEP LEARNING!
            └───┐
                │  11.7% (2013: ZFNet)
                └───┐
                    │  7.3%  (2014: GoogLeNet & VGGNet)
                    └───┐
                        │  5.1% ──► [Ngưỡng nhận diện của mắt người (Human Performance)]
                        └───┐
                            │  3.57% (2015: ResNet - He et al., Vượt mắt người!)
                            └───┐
                                │  2.25% (2017: SENet - Khép lại giải đấu ILSVRC)
```

- **Năm 2012:** **AlexNet** (Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton) sử dụng mạng tích chập sâu (Deep CNN) chạy trên 2 card đồ họa GPU NVIDIA GTX 580 đã **nghiền nát** tất cả các thuật toán thị giác truyền thống, giảm tỷ lệ lỗi Top-5 từ 25.8% xuống 16.4% (cách biệt 10% chưa từng có trong lịch sử). Kể từ giây phút này, Deep Learning chính thức thống trị toàn bộ thế giới AI.

---

## 7. Kết Quả Thực Nghiệm Ban Đầu Trong Bài Báo CVPR 2009

Trong bài báo gốc năm 2009, nhóm tác giả đã tiến hành các thử nghiệm cơ sở (baselines) để chứng minh tính ưu việt của cấu trúc cây phân cấp:
1. **Bag-of-Visual-Words (BoVW) & Spatial Pyramid Matching (SPM)**: Trích xuất đặc trưng SIFT cục bộ, lượng tử hóa thành từ điển thị giác 1,000 từ, và huấn luyện máy học vector hỗ trợ SVM đa lớp.
2. **Hiện tượng suy giảm độ chính xác khi tăng quy mô**:
   - Khi tăng số lượng lớp từ 10 lên 100 rồi 1,000 lớp, độ chính xác của phân loại phẳng giảm mạnh.
   - Tuy nhiên, khi đánh giá bằng **Tree-Hop Semantic Distance**, mô hình vẫn giữ được khả năng gom nhóm các lớp cùng nhánh phả hệ cực tốt.
3. **Chuyển giao tri thức phân cấp (Hierarchical Knowledge Transfer)**: Huấn luyện mô hình nhận diện ở các nút tổ tiên mức cao (như nhận diện loài chim nói chung) giúp cải thiện đáng kể khả năng nhận diện các loài chim cụ thể ở mức lá khi số lượng ảnh huấn luyện ở mức lá bị giới hạn.

---

## 8. Di Sản, Thách Thức & Sự Tiến Hóa Đến Ngày Nay

### 8.1. Khởi xướng kỷ nguyên Pre-training & Fine-tuning
ImageNet đã tạo nên thói quen nghiên cứu kinh điển suốt một thập kỷ:  
*"Tiền huấn luyện mô hình thị giác trên tập dữ liệu ImageNet-1K để học các bộ lọc cạnh, kết cấu và hình khối cơ bản; sau đó tinh chỉnh (fine-tune) trên tập dữ liệu chuyên ngành (ảnh y tế, xe tự hành, vệ tinh) với lượng dữ liệu nhỏ hơn rất nhiều."*

### 8.2. Các tranh luận về bản quyền, quyền riêng tư & thiên kiến
- **Quyền riêng tư khuôn mặt:** Nhiều bức ảnh trong ImageNet thu thập từ mạng internet chứa khuôn mặt người qua đường chưa được sự đồng ý. Năm 2021, nhóm ImageNet đã tiến hành dự án làm mờ khuôn mặt (Face-blurring) trên hàng triệu bức ảnh để bảo vệ quyền riêng tư.
- **Thiên kiến địa lý và văn hóa:** Phần lớn ảnh thu thập trong giai đoạn 2007–2008 đến từ Bắc Mỹ và Tây Âu, dẫn đến mô hình nhận diện trang phục hoặc đám cưới theo phong cách phương Tây tốt hơn nhiều so với phong cách châu Á hoặc châu Phi.

### 8.3. Sự dịch chuyển sang Kỷ nguyên Tự Giám Sát (Self-Supervised) & Mô Hình Nền Tảng
- Khi quy mô mô hình vượt mốc hàng tỷ tham số, chi phí gán nhãn thủ công qua AMT không còn theo kịp.
- Các tập dữ liệu hiện đại chuyển dịch sang **hình ảnh tự nhiên kèm văn bản mô tả (Image-Text Pairs)** được cào tự động từ web mà không cần phân loại thủ công:
  - **CLIP (OpenAI 2021)**: Huấn luyện trên 400 triệu cặp ảnh - văn bản qua học đối sánh (Contrastive Learning).
  - **LAION-5B (2022)**: 5.85 tỷ cặp ảnh - văn bản công khai dùng huấn luyện Stable Diffusion.
  - **DINOv2 (Meta 2023)**: Tự học biểu diễn thị giác không giám sát trên hàng trăm triệu ảnh không nhãn.

---

## 9. Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành

Thư mục này cung cấp hệ thống 4 tệp cài đặt mã nguồn chuẩn mực (From Scratch & PyTorch) kèm tài liệu phân tích và PDF gốc:

```text
Day 04 - ImageNet/
├── README.md                      # Tài liệu phân tích chuyên sâu toàn bộ bài báo & đối chiếu học thuật
├── imagenet_scratch.py            # Cài đặt Cây Phả Hệ WordNet, khoảng cách Tree-Hop, AMT Bayes & Academic Benchmark
├── imagenet_scratch.ipynb         # Notebook thực hành tương tác From Scratch
├── imagenet_pytorch.py            # Triển khai Hierarchical Tree Loss, Vision CNN & Academic Benchmark 3 mô hình
├── imagenet_pytorch.ipynb         # Notebook thực hành PyTorch tương tác & trực quan hóa 4 biểu đồ
├── outputs/                       # Thư mục lưu biểu đồ đánh giá & đối chiếu học thuật
│   └── imagenet_academic_comparison.png # Biểu đồ 4 bảng phân tích học thuật chuyên sâu
└── resources/
    └── Day 4 ImageNet.pdf         # Toàn văn bài báo gốc CVPR 2009
```

### 9.1. Tệp `imagenet_scratch.py` (Cài đặt từ con số 0)
- **Công nghệ**: Thuần Python + NumPy (không phụ thuộc framework ngoài).
- **Tính năng nổi bật**:
  - `SynsetNode` & `WordNetHierarchy`: Xây dựng cấu trúc cây phả hệ WordNet với giải thuật tìm Tổ tiên chung gần nhất (LCA), độ sâu phân cấp, và hiển thị sơ đồ cây phả hệ dưới dạng **ASCII Tree**.
  - `AMTQualityVerifier`: Mô phỏng thuật toán kiểm định chất lượng đám đông Amazon Mechanical Turk bằng **Cập nhật xác suất hậu nghiệm Bayes (Bayesian Updating)** và giải thuật **Dừng thích ứng (Adaptive Stopping)**, chứng minh khả năng đạt độ sạch nhãn >98% với chi phí tiết kiệm ~70%.
  - `compute_top_k_accuracy` & `compute_hierarchical_cost`: Đo lường Top-1 / Top-5 Accuracy và Chi phí nhảy cây (Mean Tree-Hop Distance).
  - `run_academic_value_comparison()`: **Module đối chiếu giá trị học thuật** so sánh trực diện Kỷ nguyên Tiền-ImageNet (Flat 0/1 Error) với Đột phá ImageNet (Phân cấp ngữ nghĩa).

### 9.2. Tệp `imagenet_pytorch.py` (Triển khai bằng PyTorch)
- **Công nghệ**: PyTorch, torchvision, matplotlib.
- **Tính năng nổi bật**:
  - `HierarchicalCrossEntropyLoss`: Hàm mất mát phân cấp phạt sai số tỷ lệ với ma trận khoảng cách cây phả hệ WordNet.
  - `HierarchicalVisionCNN`: Mạng nơ-ron tích chập thị giác nhiều tầng (Conv-BatchNorm-ReLU-MaxPool) kèm Dropout.
  - `HandcraftedFeatureClassifier`: Mô phỏng trường phái Tiền-ImageNet trích xuất lược đồ màu (Color Histogram) kết hợp phân loại tuyến tính.
  - `run_academic_value_benchmark()`: Thực thi thử nghiệm đối chiếu 3 mô hình:
    1. *Mô Hình 1 (Tiền-ImageNet):* Hand-crafted features + Linear classifier.
    2. *Mô Hình 2 (CNN Dữ Liệu Nhỏ - Caltech-101 scale):* Quá khớp (Overfitting) trầm trọng do thiếu dữ liệu lớn đa dạng.
    3. *Mô Hình 3 (Tư duy ImageNet Scale + Tree Loss):* Học biểu diễn sâu phong phú, xóa bỏ khoảng cách quá khớp, giảm thiểu lỗi nhảy cây.
  - `plot_comprehensive_academic_comparison()`: Tự động lưu biểu đồ 4 bảng phân tích vào `outputs/imagenet_academic_comparison.png`.

### 9.3. Bảng Tổng Kết Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark)

| Tiêu Chí So Sánh | Mô Hình 1: Hand-crafted (Tiền-2009) | Mô Hình 2: CNN Dữ Liệu Nhỏ (Caltech Scale) | Mô Hình 3: CNN ImageNet Scale (Hierarchical) |
|---|:---:|:---:|:---:|
| **Trọng tâm tiếp cận** | Feature Engineering thủ công | Mô hình sâu nhưng dữ liệu nhỏ | **AI Lấy Dữ Liệu Làm Trung Tâm (Data-Centric)** |
| **Biểu diễn thị giác** | Nông, phụ thuộc góc chụp/màu nền | Sâu nhưng ghi nhớ dữ liệu tập train | **Sâu, đa tầng, bất biến ngữ nghĩa** |
| **Train Accuracy** | ~45% | ~99% (Bị học vẹt) | **~90%** |
| **Top-1 Val Accuracy** | ~38% | ~52% (Tổng quát hóa kém) | **~85% (Vượt trội)** |
| **Top-5 Val Accuracy** | ~72% | ~84% | **~98% (Gần như tuyệt đối)** |
| **Khoảng cách Quá Khớp** | Thấp (Do Underfitting) | **Cực cao (>45% - Overfitting)** | **Rất thấp (<5% - Tổng quát hóa cao)** |
| **Mean Tree-Hop Error** | Cao (~5.2 bước nhảy) | Trung bình (~4.1 bước nhảy) | **Rất thấp (~1.4 bước nhảy - Lỗi rất văn minh)** |
| **Lỗi Xuyên Miền (Cross-Domain)** | Phổ biến (Nhầm lẫn bừa bãi) | Khá nhiều | **Cực kỳ hiếm (Chỉ nhầm lẫn trong cùng họ)** |

### 9.4. Hướng Dẫn Chạy & Thực Hành

```bash
# 1. Chạy mã nguồn From Scratch (Thuần Python + NumPy)
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 04 - ImageNet/imagenet_scratch.py"

# 2. Chạy mã nguồn PyTorch & Xuất biểu đồ đối chiếu học thuật
python "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 04 - ImageNet/imagenet_pytorch.py"

# 3. Khởi chạy Jupyter Notebook để khám phá tương tác trực quan
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 04 - ImageNet/imagenet_scratch.ipynb"
jupyter notebook "papers-reading/20 AI Research Paper quan trọng nhất của TK 21/Day 04 - ImageNet/imagenet_pytorch.ipynb"
```

---

## 10. Câu Hỏi Ôn Tập Chuyên Sâu

1. **Tại sao việc tổ chức nhãn theo cấu trúc Cây Phả Hệ WordNet (Synsets) lại vượt trội hoàn toàn so với danh sách nhãn phẳng (Flat List) trong các tập dữ liệu trước đó?**
2. **Phân biệt bản chất giữa độ đo Top-1 Error và Top-5 Error. Trong hoàn cảnh nào thì việc đánh giá bằng Top-5 Error là công bằng và phản ánh đúng năng lực mô hình hơn?**
3. **Khoảng cách nhảy cây (Tree-Hop Distance) giữa hai nút $c_i, c_j$ thông qua Tổ tiên chung gần nhất (LCA) được tính như thế nào? Ý nghĩa của nó trong việc đánh giá lỗi phân loại là gì?**
4. **Nhóm nghiên cứu ImageNet đã giải quyết bài toán kiểm soát chất lượng dữ liệu gán nhãn từ hàng chục nghìn lao động trên Amazon Mechanical Turk (AMT) bằng cơ chế nào?**
5. **So sánh sự khác biệt căn bản giữa phương pháp tiếp cận của ImageNet (Supervised Learning với nhãn người đóng khung) và các mô hình nền tảng hiện đại như CLIP / DINO (Self-Supervised Learning / Multimodal Web-scale Data)?**

---
*Tài liệu được biên soạn công phu cho lộ trình Jetiev_learningAI.*
