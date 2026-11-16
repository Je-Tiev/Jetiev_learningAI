# Stanford CS229: Machine Learning (Autumn 2018)

**Giảng viên:** Andrew Ng (cùng các giảng viên khách mời như Tengyu Ma, Raphael Townshend, Anand Avati...)  
**Học kỳ:** Autumn 2018 (Mùa thu 2018)  
**Tài liệu gốc:** [Stanford CS229 Official Website](https://cs229.stanford.edu/) & [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

---

## Giới thiệu khóa học

**CS229** là một trong những khóa học nền tảng và nổi tiếng nhất về Machine Learning (Học máy) tại Đại học Stanford. Khóa học cung cấp cái nhìn sâu sắc về mặt toán học, lý thuyết và ứng dụng thực tiễn của các thuật toán học máy hiện đại. Khác với khóa học Machine Learning trên Coursera (vốn hướng tới trực giác và ứng dụng nhanh), CS229 đi sâu vào chứng minh toán học, tối ưu hóa lồi, xác suất thống kê và lý thuyết học thống kê.

### Yêu cầu kiến thức nền tảng (Prerequisites)
Để theo dõi tốt nội dung các file ghi chú này, bạn cần nắm vững:
1. **Đại số tuyến tính (Linear Algebra):** Ma trận, vector, giá trị riêng (eigenvalues), vector riêng (eigenvectors), phân rã giá trị suy biến (SVD), đạo hàm ma trận.
2. **Giải tích (Calculus):** Đạo hàm riêng, Gradient, ma trận Hessian, quy tắc chuỗi (Chain rule).
3. **Xác suất & Thống kê (Probability & Statistics):** Biến ngẫu nhiên, phân phối xác suất (Gaussian, Bernoulli, Multinomial), kỳ vọng, phương sai, định lý Bayes, ước lượng hợp lý cực đại (MLE).
4. **Lập trình cơ bản:** Python (NumPy, SciPy) và kiến thức cơ bản về thuật toán.

---

## Lộ trình & Danh sách bài giảng (Syllabus)

Khóa học bao gồm 20 bài giảng được chia thành 4 phần chính:

### Phần 1: Học có giám sát (Supervised Learning)
* **Lecture 01:** Giới thiệu về Machine Learning, phân loại bài toán (Supervised, Unsupervised, Reinforcement Learning) và định nghĩa chính thức của Arthur Samuel & Tom Mitchell. *(Đã có ghi chú chi tiết)*
* **Lecture 02:** Hồi quy tuyến tính (Linear Regression), Gradient Descent, Phương trình chuẩn (Normal Equations). *(Đã có ghi chú chi tiết)*
* **Lecture 03:** Hồi quy có trọng số cục bộ (Locally Weighted Regression), Diễn giải xác suất (Probabilistic Interpretation), Hồi quy Logistic (Logistic Regression).
* **Lecture 04:** Thuật toán Perceptron, Mô hình tuyến tính tổng quát (Generalized Linear Models - GLMs), Họ phân phối mũ (Exponential Family).
* **Lecture 05:** Thuật toán học sinh (Generative Learning Algorithms), Phân tích biệt thức Gaussian (GDA), Naive Bayes, Làm mịn Laplace (Laplace Smoothing).
* **Lecture 06:** Máy vector hỗ trợ (Support Vector Machines - SVMs), Lề tối ưu (Optimal Margin Classifier), Đối ngẫu Lagrange (Lagrange Duality).
* **Lecture 07:** Phương pháp Kernel (Kernels), Thuật toán SMO, Ứng dụng của SVM.

### Phần 2: Học sâu (Deep Learning)
* **Lecture 08:** Mạng nơ-ron nhân tạo (Neural Networks), Lan truyền tiến (Forward Propagation), Lan truyền ngược (Backpropagation).
* **Lecture 09:** Các kỹ thuật cải thiện mạng nơ-ron (Vectorization, Initialization, Normalization, Optimization: Momentum, RMSprop, Adam).

### Phần 3: Lý thuyết học & Thực hành tốt (Learning Theory & Advice)
* **Lecture 10:** Lý thuyết học (Learning Theory): Bias/Variance tradeoff, Bất đẳng thức Hoeffding, Hội tụ đồng đều (Uniform Convergence), Chiều VC (VC Dimension).
* **Lecture 11:** Lời khuyên khi áp dụng Machine Learning (Debugging ML algorithms, Error Analysis, Ablative Analysis).
* **Lecture 12:** Cây quyết định (Decision Trees) và Phương pháp kết hợp (Ensemble Methods: Bagging, Random Forests, Boosting).

### Phần 4: Học không giám sát & Học tăng cường (Unsupervised & Reinforcement Learning)
* **Lecture 13:** Phân cụm K-means (K-means Clustering), Mô hình hỗn hợp Gaussian (Gaussian Mixture Models - GMM) và Thuật toán EM (Expectation-Maximization).
* **Lecture 14:** Thuật toán EM tổng quát (General EM Algorithm), Phân tích nhân tố (Factor Analysis).
* **Lecture 15:** Phân tích thành phần chính (Principal Component Analysis - PCA), Phân tích thành phần độc lập (Independent Component Analysis - ICA).
* **Lecture 16:** Học tăng cường (Reinforcement Learning - RL) và Quá trình quyết định Markov (MDPs), Lặp giá trị (Value Iteration), Lặp chính sách (Policy Iteration).
* **Lecture 17:** Học tăng cường nâng cao: Mô hình hóa MDP liên tục, Q-Learning, Fitted Value Iteration.
* **Lecture 18:** Điều khiển tối ưu tuyến tính bậc hai (LQR), Lọc Kalman (Kalman Filtering), LQG.
* **Lecture 19:** Định hướng chính sách (Policy Gradient - REINFORCE), POMDPs.
* **Lecture 20:** Tổng kết khóa học và các hướng nghiên cứu mở rộng.
