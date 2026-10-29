# Tổng Hợp & Đọc Các Bài Báo Nghiên Cứu AI (AI Papers Reading)

Kho lưu trữ và tài liệu hướng dẫn nghiên cứu chuyên sâu các bài báo khoa học (Research Papers) mang tính bước ngoặt định hình nên kỷ nguyên Trí tuệ Nhân tạo hiện đại.

Mọi bài báo trong hệ thống đều được định hướng xây dựng theo **Quy chuẩn 4 thành phần** khép kín, lấy công trình chuẩn mẫu tại [`Day 01 - Random Forest`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2001%20-%20Random%20Forest) và [`Day 02 - Autoencoder`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2002%20-%20Autoencoder) làm thước đo tiêu chuẩn.

---

## 📌 Mục Lục
1. [Danh Mục Các Chuyên Đề Nghiên Cứu](#1-danh-mục-các-chuyên-đề-nghiên-cứu)
2. [Cấu Trúc Thư Mục Tiêu Chuẩn 4 Thành Phần](#2-cấu-trúc-thư-mục-tiêu-chuẩn-4-thành-phần)
3. [Quy Chuẩn Biên Soạn Báo Cáo Lý Thuyết (`README.md`)](#3-quy-chuẩn-biên-soạn-báo-cáo-lý-thuyết-readmemd)
4. [Tiêu Chuẩn Kỹ Thuật Mã Nguồn From-Scratch (`*_scratch.py` & `*.ipynb`)](#4-tiêu-chuẩn-kỹ-thuật-mã-nguồn-from-scratch-_scratchpy--ipynb)
5. [Tiêu Chuẩn Kỹ Thuật Mã Nguồn PyTorch (`*_pytorch.py` & `*.ipynb`)](#5-tiêu-chuẩn-kỹ-thuật-mã-nguồn-pytorch-_pytorchpy--ipynb)
6. [Khung Chuẩn Hóa Khối Đối Chuẩn Giá Trị Học Thuật & Bình Giải Cho Người Mới Bắt Đầu](#6-khung-chuẩn-hóa-khối-đối-chuẩn-giá-trị-học-thuật--bình-giải-cho-người-mới-bắt-đầu)
7. [Quản Lý Tài Nguyên Gốc (`resources/`)](#7-quản-lý-tài-nguyên-gốc-resources)
8. [Bảng Tiến Độ & Trạng Thái Tổng Thể](#8-bảng-tiến-độ--trạng-thái-tổng-thể)

---

## 1. Danh Mục Các Chuyên Đề Nghiên Cứu

Kho lưu trữ được phân chia thành 3 chuyên đề độc lập với tổng cộng 39 công trình nghiên cứu:

| Chuyên Đề | Trọng Tâm Nghiên Cứu | Quy Mô | Đường Dẫn Chi Tiết |
| **20 AI Research Papers Quan Trọng Nhất Thế Kỷ 21** | Các cột mốc định hình Học sâu: Random Forest, Autoencoder, CNN, GAN, Transformer, BERT, GPT, ViT,... | 20 bài báo | [`20 AI Research Paper quan trọng nhất của TK 21/`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021) |
| **10 Research Papers Quan Trọng Nhất Về NLP** | Các đột phá xử lý ngôn ngữ tự nhiên: T5, Scaling Laws, LoRA, CoT, Instruction Tuning, DeepSeek-R1,... | 10 bài báo | [`10 research paper quan trọng nhất về NLP/`](./10%20research%20paper%20quan%20trọng%20nhất%20về%20NLP) |
| **10 Research Papers Quan Trọng Nhất Về RAG** | Các kiến trúc Truy xuất Tăng cường Tạo sinh & Agentic RAG: Self-RAG, Corrective RAG, GraphRAG, ReAct,... | 10 bài báo | [`10 research paper quan trọng nhất về RAG/`](./10%20research%20paper%20quan%20trọng%20nhất%20về%20RAG) |

---

## 2. Cấu Trúc Thư Mục Tiêu Chuẩn 4 Thành Phần

Mỗi bài báo khi được triển khai hoàn thiện phải tuân thủ nghiêm ngặt mô hình cấu trúc độc lập như sau:

```text
Day XX - [Tên Bài Báo]/
├── README.md                  # Báo cáo lý thuyết chuyên sâu (10 mục chuẩn)
├── [tên_mô_hình]_scratch.py   # Cài đặt thuật toán cốt lõi from-scratch thuần NumPy kèm khối so sánh học thuật
├── [tên_mô_hình]_scratch.ipynb# Notebook thực hành tương tác từng ô kèm bình giải chi tiết cho người mới
├── [tên_mô_hình]_pytorch.py   # Triển khai mô hình hiện đại bằng PyTorch, kèm Fallback an toàn & đồ thị outputs/
├── [tên_mô_hình]_pytorch.ipynb# Notebook thực hành PyTorch kèm trực quan hóa biểu đồ inline và đối chuẩn
└── resources/                 # Thư mục lưu trữ bài báo gốc định dạng PDF
    └── Day XX [Tên].pdf
```

> 🎯 **Tham chiếu mẫu (Gold Standard Benchmark):**  
> Hãy đối chiếu trực tiếp với [`Day 01 - Random Forest`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2001%20-%20Random%20Forest) và [`Day 02 - Autoencoder`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2002%20-%20Autoencoder) về độ sâu lý thuyết, chất lượng bình giải cho người mới bắt đầu và thiết kế khối thực nghiệm so sánh giá trị học thuật trước khi triển khai các bài báo tiếp theo.

---

## 3. Quy Chuẩn Biên Soạn Báo Cáo Lý Thuyết (`README.md`)

File `README.md` của mỗi bài báo là một tài liệu phân tích kỹ thuật độc lập, chi tiết bằng tiếng Việt, bao gồm đúng **10 mục tiêu chuẩn**:

### Cấu Trúc Khung 10 Mục:

```markdown
# Day XX: [Tên Khái Niệm] — [Tiêu Đề Đầy Đủ Của Bài Báo Gốc]

> **Bài báo gốc:** *[Tên bài báo in nghiêng]*  
> **Tác giả:** [Danh sách tác giả chính]  
> **Xuất bản:** [Hội nghị / Tạp chí, Năm, Thông tin ấn bản]  
> **Tài liệu PDF gốc:** [`resources/Day XX [Tên].pdf`](./resources/Day%20XX%20[Tên].pdf)

---

## 📌 Mục Lục
1. [Tổng Quan & Ý Nghĩa Lịch Sử](#1-tổng-quan--ý-nghĩa-lịch-sử)
2. [Bối Cảnh Lịch Sử & Bài Toán](#2-bối-cảnh-lịch-sử--bài-toán)
3. [Kiến Trúc Mô Hình Chi Tiết](#3-kiến-trúc-mô-hình-chi-tiết)
4. [Đột Phá Kỹ Thuật / Thuật Toán Cốt Lõi](#4-đột-phá-kỹ-thuật--thuật-toán-cốt-lõi)
5. [Cơ Chế Hoạt Động & Phân Tích Giải Thuật](#5-cơ-chế-hoạt-động--phân-tích-giải-thuật)
6. [Quy Trình Huấn Luyện & Thực Thi](#6-quy-trình-huấn-luyện--thực-thi)
7. [Kết Quả Thực Nghiệm Trong Bài Báo](#7-kết-quả-thực-nghiệm-trong-bài-báo)
8. [Di Sản, Hạn Chế & Sự Tiến Hóa Đến Ngày Nay](#8-di-sản-hạn-chế--sự-tiến-hóa-đến-ngày-nay)
9. [Cấu Trúc Mã Nguồn & Hướng Dẫn Thực Hành](#9-cấu-trúc-mã-nguồn--hướng-dẫn-thực-hành)
10. [Câu Hỏi Ôn Tập Chuyên Sâu](#10-câu-hỏi-ôn-tập-chuyên-sâu)
```

---

## 4. Tiêu Chuẩn Kỹ Thuật Mã Nguồn From-Scratch (`*_scratch.py` & `*.ipynb`)

Tệp mã nguồn cài đặt từ đầu đóng vai trò cốt lõi trong việc thấu hiểu bản chất đại số tuyến tính và vi tích phân của thuật toán:

1. **Không sử dụng framework học sâu:** Chỉ dùng thuần **Python chuẩn** và thư viện số học **NumPy**. Tuyệt đối không dùng PyTorch, TensorFlow hay Scikit-learn trong các lớp mô hình cốt lõi của tệp này.
2. **Tự cài đặt lan truyền ngược (Manual Backprop):** Mọi hàm kích hoạt, lớp tuyến tính/phi tuyến, hàm chi phí đều phải có hàm tính đạo hàm giải tích (derivative/backward pass) rõ ràng.
3. **Đảm bảo tính ổn định số học (Numerical Stability):** 
   - Kẹp giá trị (clipping) đối với hàm mũ/logarithm (`np.clip(..., -500, 500)`) để chống tràn số học (overflow/underflow).
   - Thêm $\epsilon = 10^{-15}$ khi tính log trong hàm Cross-Entropy để tránh $\log(0)$.
4. **Mô phỏng khép kín có thể chạy độc lập (Self-contained Demo):** 
   - Tự sinh dữ liệu mẫu (synthetic data) có kiểm soát nếu không nạp dữ liệu ngoài, đảm bảo bất kỳ ai chạy lệnh `python [model]_scratch.py` đều thực thi trơn tru và in kết quả kiểm chứng trực quan ra terminal.
5. **Bình giải chi tiết cho người mới bắt đầu (Beginner-Friendly Annotations):**
   - Docstring và chú thích dòng lệnh phải giải thích: Đầu vào (Input shape), Đầu ra (Output shape), Ý nghĩa trực giác, và ví dụ số học thực tế.
   - Tránh viết code rút gọn khó hiểu; ưu tiên sự tường minh của thuật toán.
6. **Bắt buộc có Khối Đối Chuẩn Giá Trị Học Thuật (Academic Value Benchmark):**
   - Phải lập trình một hàm so sánh trực tiếp phương pháp trước bài báo (Baseline) với thuật toán của bài báo đề xuất, in bảng định lượng Train/Test Acc, Overfitting Gap, và các chỉ số lý thuyết (như hệ số tương quan cây $\rho$, sai số OOB,...).

---

## 5. Tiêu Chuẩn Kỹ Thuật Mã Nguồn PyTorch (`*_pytorch.py` & `*.ipynb`)

Tệp mã nguồn PyTorch đại diện cho cách tiếp cận chuẩn hóa trong môi trường nghiên cứu và công nghiệp hiện đại:

1. **Tuân thủ chuẩn thiết kế PyTorch:** Kế thừa từ `torch.nn.Module`, phân tách rành mạch khối hàm khởi tạo `__init__` và luồng truyền thẳng `forward`.
2. **Cơ chế Fallback thông minh cho dữ liệu:** 
   - Nếu bài báo dùng dataset thực tế (như MNIST, CIFAR, Breast Cancer, WikiText), code cần hỗ trợ tải tự động qua `torchvision.datasets` hoặc thư viện chuẩn.
   - **Bắt buộc:** Phải có cơ chế `try...except` tự động chuyển sang sinh dữ liệu ngẫu nhiên (synthetic tensor fallback) khi không có kết nối internet, đảm bảo môi trường offline không bao giờ bị lỗi văng chương trình.
3. **Trực quan hóa và đối chuẩn chuyên sâu (Academic Benchmark & Visualization):**
   - So sánh định lượng với ít nhất 2 phương pháp tiền bối và phương pháp hiện đại (ví dụ: Linear $\to$ CART $\to$ Classical Random Forest $\to$ Neural Decision Forest).
   - Tự động xuất ảnh đồ thị 4 bảng con đánh giá (Loss curve, Accuracy benchmark, Overfitting gap, Feature importance / Latent space) lưu vào thư mục `outputs/`.
4. **Linh hoạt thiết bị (Device Agnostic):** Tự động phát hiện và chuyển đổi giữa GPU/CUDA và CPU: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
5. **Đồng bộ hóa sổ tay Notebook (`.ipynb`):**
   - Các file `.ipynb` tương ứng phải phân tách thành các ô Markdown giải thích lý thuyết chi tiết kèm hình vẽ ASCII, tiếp nối bởi ô code thực thi có thể chạy độc lập.

---

## 6. Khung Chuẩn Hóa Khối Đối Chuẩn Giá Trị Học Thuật & Bình Giải Cho Người Mới Bắt Đầu

Để đảm bảo tính nhất quán tuyệt đối khi triển khai **toàn bộ 39 bài báo tiếp theo** (từ Day 02 đến Day 20, chuyên đề NLP, chuyên đề RAG), tác giả/người đóng góp **bắt buộc phải tuân thủ khung đối chuẩn 3 thành phần** dưới đây:

### 6.1. Khung Cấu Trúc Khối So Sánh Học Thuật Trong Code:

Mỗi tệp code (`*_scratch.py` và `*_pytorch.py`) cần chứa hàm thực nghiệm học thuật (ví dụ `run_academic_value_comparison()`):

```python
# ==============================================================================
# KHUNG ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT (MẪU TIÊU CHUẨN CHO MỌI BÀI BÁO)
# ==============================================================================

def run_academic_value_comparison():
    """
    1. Giai đoạn tiền đề (Pre-Paper Baselines):
       Mô hình hóa giải pháp cổ điển trước khi bài báo ra đời.
       (Ví dụ: Cây CART đơn lẻ, PCA tuyến tính, RNN tuần tự, BoW/TF-IDF,...)
    
    2. Đột phá của bài báo (Proposed Innovation):
       Mô hình hóa giải pháp được đề xuất trong bài báo khoa học.
       (Ví dụ: Random Subspace RF, Deep Autoencoder Pre-training, Transformer Self-Attention,...)
    
    3. Biến thể / Cải tiến hiện đại (Modern Evolution / PyTorch):
       Mô hình hóa bước phát triển kế thừa ngày nay.
       (Ví dụ: Neural Decision Forest, VAE/Diffusion, RoBERTa/FlashAttention, GraphRAG,...)
    
    4. Đo lường định lượng các chỉ số then chốt:
       - Train Loss / Test Loss & Train Acc / Test Acc
       - Khoảng cách Quá khớp (Overfitting Gap = Train - Test)
       - Khả năng kháng nhiễu (Noise / Robustness Test)
       - Tốc độ huấn luyện & Bộ nhớ tiêu hao
    
    5. In Bảng Tổng Hợp Học Thuật & Đưa ra phân tích kết luận cho người mới bắt đầu.
    """
```

### 6.2. Bảng Tra Cứu Đối Chuẩn Cho Các Bài Báo Khoa Học Tiếp Theo:

Áp dụng bảng tra cứu sau đây khi xây dựng mã nguồn đối chuẩn cho từng bài báo:

| Mã Bài Báo | Chủ Đề / Bài Báo | Cột Mốc 1: Tiền Bối (Pre-Paper) | Cột Mốc 2: Đột Phá Của Bài Báo | Cột Mốc 3: Kế Thừa Hiện Đại | Chỉ Số Cần Đo Lường |
|:---:|---|---|---|---|---|
| **Day 01** | **Random Forest** (Breiman 2001) | Single CART (1984), Bagging (1996) | Random Subspace RF ($m = \sqrt{M}$) | Differentiable Neural Forest | Tương quan $\rho$, Overfitting Gap, OOB vs Test Error |
| **Day 02** | **Autoencoder** (Hinton 2006) | Tuyến tính PCA (1901), Shallow MLP | Greedy Layer-Wise RBM Pre-training | Variational Autoencoder (VAE) | Sai số tái tạo (MSE), Latent Manifold 2D |
| **Day 03** | **t-SNE** (van der Maaten 2008) | PCA, Multidimensional Scaling (MDS) | t-Distribution Student-t (Crowding problem) | UMAP (2018) | Mức độ tách cụm phi tuyến, KL Divergence |
| **Day 04** | **ImageNet** (Deng et al. 2009) | Tập dữ liệu nhỏ (Caltech-101, PASCAL) | Phân tầng WordNet 14M ảnh gán nhãn | Self-Supervised Large Datasets | Top-1 & Top-5 Error Rate qua quy mô dữ liệu |
| **Day 05** | **AlexNet** (Krizhevsky 2012) | Handcrafted (SIFT/HOG) + SVM, LeNet-5 | Deep CNN + ReLU + Dropout + 2 GPU | Modern ConvNeXt / EfficientNet | Top-1 Acc, Tốc độ hội tụ ReLU vs Sigmoid |
| **Day 06** | **Word2Vec** (Mikolov 2013) | One-hot Encoding, SVD Co-occurrence | Skip-gram & CBOW với Negative Sampling | Contextual Embeddings (BERT/RoBERTa) | Tương đồng Cosine, Phép toán ngữ nghĩa ($King - Man + Woman$) |
| **Day 07** | **VAE** (Kingma & Welling 2013) | Deterministic Autoencoder | Reparameterization Trick, ELBO Loss | Diffusion Models (DDPM) | Khả năng sinh mẫu ngẫu nhiên $z \sim \mathcal{N}(0, I)$ |
| **Day 08** | **GAN** (Goodfellow 2014) | Maximum Likelihood, VAE mờ ảnh | Trò chơi Minimax Đối nghịch ($G$ vs $D$) | WGAN-GP, StyleGAN | Độ sắc nét ảnh sinh, Jensen-Shannon / Wasserstein Loss |
| **Day 09** | **Dropout** (Srivastava 2014) | Regularization truyền thống ($L_2$ Decay) | Triệt tiêu ngẫu nhiên nơ-ron (Bernoulli $p$) | DropPath, Stochastic Depth | Overfitting Gap giữa mạng có và không có Dropout |
| **Day 10** | **Adam** (Kingma & Ba 2014) | Vanilla SGD, SGD có Quán tính (Momentum) | Thích ứng mô-men bậc 1 ($m_t$) và bậc 2 ($v_t$) | AdamW, Lion Optimizer | Tốc độ suy giảm hàm mất mát qua từng Epoch |
| **Day 11** | **BatchNorm** (Ioffe & Szegedy 2015) | Khởi tạo trọng số thủ công (Xavier) | Chuẩn hóa Mini-batch theo Mean & Var | LayerNorm (cho Transformer), RMSNorm | Ổn định Gradient, Khả năng dùng Learning Rate lớn |
| **Day 12** | **DQN** (Mnih et al. 2015) | Q-Learning dạng bảng (Tabular Q) | Deep Q-Network + Experience Replay + Target Net | Double DQN, PPO, DDPG | Phần thưởng tích lũy (Cumulative Reward), độ ổn định |
| **Day 13** | **Faster R-CNN** (Ren et al. 2015) | R-CNN, Fast R-CNN (Selective Search chậm) | Region Proposal Network (RPN) nội suy chia sẻ conv | Mask R-CNN, DETR (Transformer) | Khung hình/giây (FPS) & Mean Average Precision (mAP) |
| **Day 14** | **U-Net** (Ronneberger 2015) | FCN (Fully Convolutional Net) mất chi tiết | Kiến trúc U đối xứng + Skip Connections nối dài | TransUNet, Diffusion U-Net | Điểm số Dice / IoU trên phân vùng tế bào/y tế |
| **Day 15** | **ResNet** (He et al. 2015) | Plain Deep CNN (bị thoái hóa hiệu năng) | Residual Mapping ($F(x) + x$) vượt qua 100+ tầng | ConvNeXt, DenseNet, Pre-activation ResNet | Độ dốc suy giảm Gradient qua 50-100 tầng, Test Error |
| **Day 16** | **YOLO** (Redmon et al. 2016) | Two-stage Detectors (Faster R-CNN) | Hồi quy tọa độ trực tiếp trên lưới $S \times S$ (One-stage) | YOLOv8, YOLOv10, RT-DETR | Real-time Inference Latency (ms) vs Accuracy Tradeoff |
| **Day 17** | **Transformer** (Vaswani 2017) | Tuần tự RNN / LSTM ($O(N)$ thời gian) | Multi-Head Self-Attention tính toán song song | FlashAttention, Mamba (State Space Models) | Khả năng ghi nhớ ngữ cảnh dài, Thông lượng xử lý tokens/s |
| **Day 18** | **BERT** (Devlin et al. 2018) | Một chiều (GPT-1, Word2Vec, ELMo ghép nối) | Học biểu diễn hai chiều sâu qua Masked LM + NSP | RoBERTa, DeBERTa, Modern Encoders | GLUE Benchmark Score, Khả năng phân loại ngữ cảnh hai chiều |
| **Day 19** | **GPT-3** (Brown et al. 2020) | Fine-tuning theo tác vụ riêng lẻ | Học trong ngữ cảnh Few-Shot In-Context Learning | Instruction-Tuned LLMs (Llama 3, Mistral) | Tỷ lệ Zero-shot / Few-shot Accuracy theo số tham số |
| **Day 20** | **ViT** (Dosovitskiy 2020) | CNN truyền thống (Inductive Bias cục bộ) | Chia ảnh thành các Patches và nạp vào Transformer | Swin Transformer, EVA-02 | Accuracy trên dữ liệu tiền huấn luyện quy mô lớn |
| **NLP Papers** | **T5, LoRA, CoT, DeepSeek** | Full Fine-Tuning / Direct Output | Parameter-Efficient Tuning, Chain-of-Thought Reasoning | DPO, GRPO, Multi-Head Latent Attention | Bộ nhớ VRAM tiết kiệm, Reasoning Accuracy |
| **RAG Papers** | **Self-RAG, CRAG, GraphRAG** | Naive RAG (Chunking + Vector Search) | Phản tư (Reflection Tokens), Đồ thị tri thức Knowledge Graph | Agentic RAG, ReAct Workflows | Điểm số Faithfulness, Answer Relevance, Hallucination Rate |

---

## 7. Quản Lý Tài Nguyên Gốc (`resources/`)

- Thư mục con `resources/` bên trong mỗi thư mục bài báo được dùng để lưu trữ file PDF gốc của bài báo nghiên cứu được tải từ arXiv hoặc tạp chí khoa học chính thức.
- Tên tệp PDF cần đặt đồng nhất theo quy ước: `Day XX [Tên Thu Gọn].pdf`.
- Đảm bảo liên kết tương đối trong file `README.md` trỏ chính xác đến tệp PDF này.

---

## 8. Bảng Tiến Độ & Trạng Thái Tổng Thể

### 20 AI Research Papers Thế Kỷ 21 (20 bài)
- [`Day 01 - Random Forest`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2001%20-%20Random%20Forest): ✅ **Hoàn thành xuất sắc: Đủ 4 file code + Bình giải chi tiết cho người mới + Khối đối chuẩn giá trị học thuật**
- [`Day 02 - Autoencoder`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2002%20-%20Autoencoder): ✅ **Hoàn thành xuất sắc: Đủ 4 file code + Bình giải chi tiết cho người mới + Khối đối chuẩn giá trị học thuật**
- [`Day 03 - Visualizing Data using t-SNE`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2003%20-%20Visualizing%20Data%20using%20t-SNE): ✅ **Hoàn thành xuất sắc: Đủ 4 file code + Bình giải chi tiết cho người mới + Khối đối chuẩn giá trị học thuật**
- [`Day 04 - ImageNet`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2004%20-%20ImageNet): ✅ **Đã hoàn thành chuẩn mẫu 4 thành phần**
- [`Day 05 - AlexNet`](./20%20AI%20Research%20Paper%20quan%20trọng%20nhất%20của%20TK%2021/Day%2005%20-%20AlexNet): ✅ **Đã hoàn thành chuẩn mẫu 4 thành phần**
- `Day 06` đến `Day 20`: 📚 *Đã có PDF gốc trong `resources/`, sẵn sàng triển khai theo khung chuẩn hóa mục 6.*

### 10 Research Papers Về NLP (10 bài)
- `Day 01` đến `Day 10`: 📚 *Đã có PDF gốc trong `resources/`, sẵn sàng triển khai theo khung chuẩn hóa mục 6.*

### 10 Research Papers Về RAG (10 bài)
- `Day 01` đến `Day 10`: 📚 *Đã có PDF gốc trong `resources/`, sẵn sàng triển khai theo khung chuẩn hóa mục 6.*

---
*Tài liệu hướng dẫn chuẩn hóa được ban hành nhằm bảo đảm tính nhất quán và chất lượng học thuật cao nhất trong suốt kho lưu trữ Jetiev_learningAI.*
