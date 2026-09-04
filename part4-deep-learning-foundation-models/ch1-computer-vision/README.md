# Chương 10: Thị giác máy tính (Computer Vision)

Thị giác máy tính giúp AI "nhìn" và hiểu được thông tin từ hình ảnh và video. Nền tảng của lĩnh vực này trong học sâu là Mạng nơ-ron tích chập (CNN).

## 1. Giới thiệu về thị giác máy tính

Là lĩnh vực nghiên cứu các phương pháp cho phép máy tính thu nhận và suy luận thông tin từ hình ảnh hoặc video.

Lịch sử của thị giác máy tính là quá trình thay đổi câu hỏi nghiên cứu. Ban đầu, câu hỏi là "Làm sao mô tả cảnh bằng hình học và quy tắc?". Sau đó, câu hỏi trở thành "Làm sao phát biểu bài toán thành một bài toán tối ưu ổn định?". Tiếp thao, câu hỏi là "Làm sao biểu diễn ảnh bằng đặc trưng đủ tốt để học?". Và ngày nay, câu hỏi trung tâm là "Làm sao để mô hình tự học ra biểu diễn phù hợp trực tiếp từ dữ liệu quy mô lớn?".



## 2. Thị giác máy tính truyền thống

### Cách máy tính nhìn một bức ảnh.

- Trường hợp tổng quát, mỗi bức ảnh được biểu diễn bằng một tensor H x W x C, với H, W lần lượt là chiều dài và chiều rộng của ảnh, và C là số kênh màu.
- Cách biểu diễn thứ 2 là xem mối bức ảnh như một hàm cường độ. Tọa độ (i, j) của điểm ảnh trên mặt phẳng thì giá trị trục z của hàm I(i, j) là cường độ tại điểm đó. Với vùng mà I(i, j) gần như không đổi tương ứng với vùng ảnh đồng nhất. Nơi hàm I(i, j) có biến thiên nhanh (đạo hàm lớn) tương ứng với các biên ảnh; các điểm cực đại, cực tiểu cục bộ có thể tương ứng với các điểm sáng hoặc tối nổi bật trong ảnh.
- Một cách biểu diễn quan trọng khác của ảnh là trong miền tần số (frequency domain). Các thành phần tần số thấp tướng ứng với các vùng ảnh mượt và đồng nhất, các thành phần tần số cao biểu diễn những biến thiên nhanh, gắn với biên, chi tiết nhỏ và kết cấu của ảnh. 

### Bộ lọc

Bộ lọc là công cụ dùng để biến đổi ảnh đầu vào thành một biểu diễn mới, giúp làm nổi bật hoặc loại bỏ những đặc điểm phục vụ cho mục tiêu phân tích.

Có thể hiểu đơn giản:

> **Bộ lọc giúp ta “nhìn ảnh theo một cách có chủ đích”.**

Có thể chia thành hai nhóm lớn:

#### Bộ lọc điểm (Point Operators)

Với bộ lọc điểm, giá trị đầu ra tại một pixel chỉ phụ thuộc vào giá trị của chính pixel đó.

Một số loại phổ biến:

* **Biến đổi tuyến tính (Linear Transform):** điều chỉnh độ sáng hoặc độ tương phản.
* **Gamma Correction:** điều chỉnh cường độ ảnh theo một hàm lũy thừa.
* **Log Transform:** làm nổi bật các vùng tối.
* **Histogram Equalization:** phân bố lại cường độ pixel để cải thiện độ tương phản.

#### Bộ lọc lân cận (Neighborhood Operators)

Với bộ lọc lân cận, giá trị của một pixel đầu ra được tính dựa trên một vùng pixel xung quanh nó.

Nhóm các bộ lọc:

* **Bộ lọc lân cận tuyến tính (Linear Filters):** Các pixel trong neighborhood được kết hợp bằng một tổng có trọng số cố định.
* **Bộ lọc lân cận phi tuyến (Nonlinear Filters):** Các pixel trong neighborhood được áp dụng một quy tắc phi tuyến.
* **Toán tử hình thái học (Morphological Operators):** Mục tiêu thường là: loại bỏ object nhỏ, lấp lỗ;, nối các vùng, tách các object, phân tích hình dạng.

## 3. Mạng Nơ ron tích chập (Convolutional Neural Networks - CNNs)

Khác với MLP, mạng CNN được thiết kế đặc biệt để xử lý dữ liệu dạng lưới 2D như hình ảnh. MLP làm mất đi cấu trúc không gian của ảnh khi kéo phẳng (flatten) ảnh thành vector 1D, trong khi CNN giữ lại được cấu trúc này.

### Có thể chia CNN thành 3 nhóm chức năng chính:

- **Trích xuất đặc trưng (Feature Extraction)**

Gồm: Convolution + Activation

Nhiệm vụ: Sử dụng các bộ lọc (Filters/Kernels) trượt qua toàn bộ hình ảnh để thực hiện phép toán tích chập. Quá trình này giúp trích xuất các đặc trưng như cạnh, góc, vân, và các mẫu phức tạp.

- **Giảm hoặc biến đổi độ phân giải**

Gồm: Pooling, Strided Convolution, Các phương pháp Downsampling.

Nhiệm vụ: Giảm kích thước không gian của ma trận đặc trưng (giảm độ phân giải), giúp giảm số lượng tham số, tiết kiệm bộ nhớ tính toán và tạo ra tính bất biến dịch chuyển (translation invariance). Phổ biến nhất là Max Pooling.

- **Đưa ra dự đoán**

Fully Connected layer;
Global Average Pooling;
Classification Head.

Nhiệm vụ: Nằm ở cuối mạng, tương tự như một MLP thông thường, dùng để tổng hợp các đặc trưng bậc cao và đưa ra phân loại cuối cùng.

### Nguyên lý thiết kế

Mạng nơ-ron tích chập được xây dựng dựa trên một phép toán nền tảng là phép tích chập, vốn có nguồn gốc từ xử lý tín hiệu và xử lý ảnh. Trực quan , phép toán có thể được hiểu như sau. Ta sử dụng một kernel có kích thước nhỏ, dịch chuyển tuần tự trên toàn bộ ảnh đầu vào; tại mỗi vị trí, kernel chỉ "nhìn thấy" một vùng cục bộ của ảnh, rồi thực hiện phép nhân từng phần tử giữa các giá trị điểm ảnh trong vùng cục bộ đó và các trọng số trong kernel. Kết quả của phép toán này là một giá trị duy nhất.

> Cấu trúc của CNN không phải là lựa chọn tùy ý mà được xây dựng dựa trên 3 nguyên lý:
- **Tính kết nối thưa(Local Connectivity)**: Một neuron trong output không nhìn toàn bộ ảnh mà chỉ nhìn một vùng nhỏ trên input. Vùng đó là miền cảm thụ(receptive field). Bằng việc giả định các cấu trúc trong ảnh có tính cục bộ, không cần quan sát toàn bộ ảnh để nhận biết chúng. Tác dụng: giảm số tham số, khả năng học tốt hơn từ dữ liệu hữu hạn.
- **Chia sẻ tham số(Weight Sharing)**: Các trọng số trong một bộ lọc (kernel) sẽ được tái sử dụng lặp đi lặp lại trên toàn bộ ảnh. Điều này có nghĩa là nếu bộ lọc phát hiện ra một đặc trưng ở vị trí A, nó có khả năng phát hiện đặc trưng tương tự ở vị trí B. Cùng một loại đặc trưng có thể xuất hiện ở nhiều vị trí trong ảnh.
- **Tính tương đương theo dịch chuyển(Translation Invariance)**: Do việc chia sẻ tham số và cấu trúc lớp, mạng có khả năng nhận diện một đối tượng cho dù nó xuất hiện ở bất kỳ vị trí nào trong ảnh, không phụ thuộc vào vị trí tuyệt đối của nó mà chỉ phụ thuộc vào cấu trúc của nó.

### Các lớp cơ bản

#### Lớp tích chập (chi tiết)

#### Lớp kích hoạt

#### Lớp gộp

### Miền cảm thụ

## 4. Một số mạng CNN cho bài toán phân loại ảnh


### **LeNet (1998)**: Ứng dụng sớm nhất để nhận dạng chữ số viết tay.

### **AlexNet (2012)**: Mở ra kỷ nguyên bùng nổ của Deep Learning nhờ kích thước lớn, sử dụng ReLU và GPU.

### **VGGNet (2014)**: Sử dụng liên tục các bộ lọc nhỏ 3x3 để tạo ra mạng rất sâu.

### **ResNet (2015)**: Giới thiệu các "Kết nối tắt" (Residual/Skip Connections) cho phép huấn luyện mạng sâu hàng trăm, thậm chí hàng ngàn lớp mà không bị lỗi Triệt tiêu Gradient.

## 5. Một số mạng CNN cho bài toán nhận diện đối tượng

## 6. Mạng CNN cho bài toán phân đoạn ảnh
