# Lộ trình học tập (Roadmap)

Lộ trình này chia nhỏ mỗi phần của sách giáo khoa thành các mục con kèm theo danh sách kiểm tra (checklist). Hãy cập nhật các ô kiểm khi bạn hoàn thành từng chủ đề và ghi lại tiến độ hàng tuần trong thư mục [`logs/`](./logs).

Tốc độ gợi ý: điều chỉnh thoải mái dựa trên lịch trình và độ sâu của việc học của bạn.

---

## Phần 1 — Nền tảng của trí tuệ nhân tạo (`part1-ai-foundations`)

- [x] 01. Trí tuệ nhân tạo là gì (`ch1-what-is-ai`)
  - [x] Đọc chương / ghi chú
  - [x] Viết tóm tắt `README.md`
  - [x] `demo.ipynb` (Hệ thống luật suy diễn & máy trạng thái phản xạ)
- [x] 02. Tác tử và Giải quyết bài toán (`ch2-agents-and-problem-solving`)
  - [x] Đọc chương / ghi chú
  - [x] Viết tóm tắt `README.md`
  - [x] `demo.ipynb` (Thuật toán tìm kiếm BFS, DFS, UCS, A* với Manhattan heuristic)
- [x] 03. Biểu diễn tri thức và Lập luận (`ch3-knowledge-representation`)
  - [x] Đọc chương / ghi chú
  - [x] Viết tóm tắt `README.md`
  - [x] `demo.ipynb` (Logic mệnh đề, thuật toán suy diễn Forward Chaining & Resolution)

## Phần 2 — Cơ sở toán học cho học máy (`part2-math-foundations`)

- [x] 01. Đại số tuyến tính (`ch1-linear-algebra`)
  - [x] Vector, ma trận, trị riêng/vector riêng, SVD
  - [x] `demo.ipynb` (Cài đặt các phép toán ma trận, SVD nén ảnh, Power Iteration từ đầu)
- [x] 02. Xác suất & Thống kê (`ch2-probability-and-statistics`)
  - [x] Các phân phối, định lý Bayes, ước lượng MLE & MAP
  - [x] `demo.ipynb` (Mô phỏng phân phối, ước lượng MLE tham số Gaussian & Bernoulli)
- [x] 03. Tối ưu hóa và Giải tích (`ch3-optimization-and-calculus`)
  - [x] Giảm gradient, Momentum, Adam, đạo hàm ma trận Jacobian & Hessian
  - [x] `demo.ipynb` (Tự cài đặt Gradient Descent, Adam, và Backpropagation từ đầu)

## Phần 3 — Học máy (`part3-machine-learning`)

- [x] 01. Nguyên lý cơ bản của học máy (`ch1-machine-learning-basics`)
  - [x] Sự đánh đổi bias-variance, phân chia tập train/val/test, overfitting, chuẩn hóa L1/L2
  - [x] `demo.ipynb` (Mô phỏng Bias-Variance Tradeoff, Ridge/Lasso regularization)
- [x] 02. Học máy cổ điển (`ch2-classical-machine-learning`)
  - [x] Hồi quy tuyến tính/logistic, cây quyết định, SVM, k-NN, phân cụm K-Means
  - [x] `demo.ipynb` (Cài đặt Logistic Regression và K-Means clustering từ đầu)
- [x] 03. Mạng nơ-ron truyền thẳng (`ch3-feedforward-neural-networks`)
  - [x] Perceptron, MLP, hàm kích hoạt (ReLU, GELU), lan truyền ngược backprop
  - [x] `demo.ipynb` (Xây dựng ModularMLP giải quyết bài toán XOR và Two Spirals)

## Phần 4 — Học sâu & Các mô hình nền tảng (`part4-deep-learning-foundation-models`)

- [x] 01. Thị giác máy tính (`ch1-computer-vision`)
  - [x] CNN, pooling, receptive field, ResNet residual connection
  - [x] `demo.ipynb` (Cài đặt lớp Conv2D, MaxPooling2D và khối ResNet Block)
- [x] 02. Xử lý ngôn ngữ tự nhiên (`ch2-natural-language-processing`)
  - [x] Tokenization BPE, Word2Vec Skip-Gram, kiến trúc LSTM & GRU
  - [x] `demo.ipynb` (Thuật toán BPE Tokenizer và tế bào LSTM Cell từ đầu)
- [x] 03. Transformers (`ch3-transformer-models`)
  - [x] Cơ chế tự chú ý (Scaled Dot-Product Self-Attention), Multi-Head Attention, ViT
  - [x] `demo.ipynb` (Cài đặt Multi-Head Attention và Transformer Block hoàn chỉnh)
- [x] 04. Học biểu diễn quy mô lớn & Mô hình nền tảng (`ch4-large-scale-representation-learning`)
  - [x] Tiền huấn luyện (MLM, CLM, InfoNCE), tinh chỉnh LoRA, In-context learning
  - [x] `demo.ipynb` (Cài đặt tầng LoRA hạng thấp giảm 98% tham số, hàm mất mát InfoNCE)
- [x] 05. AI tạo sinh (`ch5-generative-ai`)
  - [x] GMM, VAE (ELBO & Reparameterization trick), GAN (Min-Max game), Diffusion DDPM
  - [x] `demo.ipynb` (Mô hình Toy VAE, Toy 1D GAN, và lịch trình Forward Diffusion)

## Phần 5 — AI đáng tin cậy & Trách nhiệm (`part5-trustworthy-responsible-ai`)

- [x] 01. Tính minh bạch & Khả năng giải thích (`ch1-transparency`)
  - [x] XAI, Hộp kính vs Hộp đen, LIME, SHAP, Saliency Maps, Integrated Gradients
  - [x] `demo.ipynb` (Cài đặt LIME từ đầu, tính toán giải tích Shapley Values, Integrated Gradients)
- [x] 02. Tính cứng cáp và Quyền riêng tư (`ch2-robustness-and-privacy`)
  - [x] Tấn công đối kháng FGSM/PGD, Đầu độc dữ liệu, Membership Inference, DP-SGD
  - [x] `demo.ipynb` (Tấn công FGSM & PGD, mô phỏng tấn công MIA, huấn luyện DP-SGD)
- [x] 03. Thiên kiến và Công bằng (`ch3-bias-and-fairness`)
  - [x] Demographic Parity, Equal Opportunity, Disparate Impact, Re-weighing, Threshold-moving
  - [x] `demo.ipynb` (Bộ đo lường công bằng toán học, tiền xử lý Re-weighing, hậu xử lý Threshold-moving)
- [x] 04. Đạo đức và Quản trị AI (`ch4-ethics-and-governance`)
  - [x] Value Alignment, Luật Goodhart, RLHF (Bradley-Terry), DPO Loss, Khung pháp lý EU AI Act, NIST AI RMF
  - [x] `demo.ipynb` (Mô hình phần thưởng Bradley-Terry và Direct Preference Optimization Loss)
