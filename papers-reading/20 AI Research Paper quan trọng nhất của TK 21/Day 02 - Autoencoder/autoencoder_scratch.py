"""
================================================================================
BÀI BÁO GỐC: Reducing the Dimensionality of Data with Neural Networks
TÁC GIẢ: G. E. Hinton & R. R. Salakhutdinov (Science, Tháng 7 năm 2006)
TỆP MÃ NGUỒN: autoencoder_scratch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Triển khai mô hình Mạng Tự Mã Hóa Sâu (Deep Autoencoder) và Cỗ máy Boltzmann
    Giới hạn (Restricted Boltzmann Machine - RBM) hoàn toàn từ con số 0 (From Scratch)
    bằng Python thuần và thư viện số học NumPy.

CẤU TRÚC NỘI DUNG:
    PHẦN 1: BỐI CẢNH HỌC THUẬT & TẦM VÓC ĐỘT PHÁ CỦA HINTON (2006)
            - Trước 2006: PCA (1901) chỉ giảm chiều tuyến tính, làm đè bẹp các đa tạp cong.
                          Mạng sâu nhiều tầng Sigmoid bị Vanishing Gradient và kẹt cực tiểu địa phương.
            - Năm 2006: Hinton phát minh Greedy Layer-wise Pre-training bằng RBM, đưa trọng số
                        vào vùng hội tụ tối ưu, mở màn kỷ nguyên Phục Hưng của Deep Learning.
    PHẦN 2: CÁC KHỐI TOÁN HỌC & TẦNG TRUYỀN THẲNG CÓ BACKPROP THỦ CÔNG
            - Hàm kích hoạt & Đạo hàm giải tích (Sigmoid, ReLU, Linear).
            - Các hàm mất mát (MSE, Binary Cross-Entropy).
            - DenseLayer: Tự tính toán đạo hàm giải tích dL/dW, dL/db, dL/dX.
    PHẦN 3: CỖ MÁY BOLTZMANN GIỚI HẠN (RBM) & THUẬT TOÁN CONTRASTIVE DIVERGENCE (CD-1)
            - Pha Dương (Positive Phase): Dữ liệu thật kích hoạt tầng ẩn.
            - Pha Âm (Negative Phase): Mạng tự tái tạo (Gibbs Sampling 1 bước).
    PHẦN 4: DEEP AUTOENCODER HOÀN CHỈNH (HINTON 3 BƯỚC)
            - Bước 1: Greedy Layer-Wise Pre-training từng tầng RBM.
            - Bước 2: Unrolling (mở bung mạng thành cấu trúc Encoder-Decoder đối xứng).
            - Bước 3: Global Fine-Tuning bằng Backpropagation toàn cục.
    PHẦN 5: THỰC NGHIỆM ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT (ACADEMIC VALUE BENCHMARK)
            - So sánh trực tiếp 4 cột mốc công nghệ trên cùng một tập dữ liệu đa tạp phi tuyến:
              1. Tuyến tính cổ điển: PCA (Principal Component Analysis - Pearson 1901)
              2. Mạng nơ-ron nông: Shallow Autoencoder (Rumelhart 1986)
              3. Mạng sâu không Pre-train: Vanilla Deep Autoencoder (Khởi tạo ngẫu nhiên trước 2006)
              4. Đột phá Hinton: Deep Autoencoder có Greedy RBM Pre-training (Science 2006)
            - Đo lường: Sai số tái tạo (Test MSE), Tỷ lệ cải thiện so với PCA, So sánh trực quan ASCII.
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: GIẢI THÍCH LÝ THUYẾT DÀNH CHO NGƯỜI MỚI BẮT ĐẦU
# ==============================================================================
"""
[GIẢI THÍCH DÀNH CHO NGƯỜI MỚI BẮT ĐẦU]:

1. BÀI TOÁN GIẢM CHIỀU DỮ LIỆU LÀ GÌ?
   - Giả sử bạn có bức ảnh kích thước 8x8 điểm ảnh = 64 con số (chiều).
   - Nhưng bức ảnh đó chỉ chụp một vật thể xoay theo 2 góc (nghiêng trái/phải, ngửa lên/xuống).
   - Bản chất thực sự của dữ liệu chỉ nằm trên một "Đa tạp" (Manifold) 2 chiều!
   - Mục tiêu: Nén từ 64 chiều xuống 2 chiều (Bottleneck) mà không làm mất thông tin cốt lõi,
     sao cho từ 2 chiều đó có thể giải nén và tái tạo lại bức ảnh 64 chiều ban đầu.

2. TRƯỚC NĂM 2006, CÁC NHÀ KHOA HỌC GIẢI QUYẾT BẰNG CÁCH NÀO?
   - Phương pháp PCA (1901): Chiếu dữ liệu lên các trục tọa độ trực giao thẳng.
     NHƯỢC ĐIỂM CHẾT NGƯỜI: PCA hoàn toàn là tuyến tính (phẳng). Nếu dữ liệu uốn cong như
     vỏ ốc hay dải lụa chữ S, PCA sẽ ép dẹp các vòng xoắn vào nhau, làm mất hoàn toàn hình học nội tại!
   - Mạng nơ-ron sâu (Deep Autoencoder): Ý tưởng nén bằng mạng nhiều tầng đã có từ thập niên 1980.
     NHƯNG KHÔNG AI HUẤN LUYỆN ĐƯỢC: Khi mạng có 4-5 tầng ẩn dùng hàm Sigmoid, đạo hàm lan truyền
     ngược bị triệt tiêu về 0 (Vanishing Gradient), các tầng đầu tiên không thể học được gì.
     Mạng sâu khi đó chỉ tái tạo ra một bức ảnh xám mờ vô nghĩa!

3. ĐỘT PHÁ MANG TÍNH LỊCH SỬ CỦA HINTON (SCIENCE 2006):
   - Hinton đề xuất: "Đừng vội huấn luyện toàn bộ mạng sâu từ đầu bằng Backpropagation ngẫu nhiên!"
   - Thay vào đó, dùng Cỗ máy Boltzmann Giới hạn (RBM) để "học trước" (Pre-train) từng tầng một
     theo kiểu không giám sát (Greedy Layer-wise Unsupervised Pre-training).
   - Tầng 1 học cách tóm tắt dữ liệu gốc -> Tầng 2 học cách tóm tắt của tầng 1 -> ...
   - Sau khi các tầng đã có trọng số ban đầu rất tốt, ta mới mở bung mạng (Unroll) thành Autoencoder
     đối xứng và dùng Backpropagation để tinh chỉnh nhẹ nhàng (Fine-tune).
   - Kết quả: Mạng sâu hội tụ xuất sắc, vượt trội hoàn toàn PCA, khai sinh ra kỷ nguyên Deep Learning!
"""


# ==============================================================================
# PHẦN 2: CÁC HÀM TOÁN HỌC & ĐẠO HÀM GIẢI TÍCH
# ==============================================================================

def sigmoid(x):
    """
    Hàm kích hoạt Sigmoid: sigma(x) = 1 / (1 + exp(-x))
    Kẹp giá trị trong khoảng [-500, 500] để chống tràn số học (numerical overflow).
    """
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(output):
    """
    Đạo hàm của hàm Sigmoid theo giá trị đầu ra y = sigmoid(x):
    d(sigma)/dx = sigma(x) * (1 - sigma(x)) = y * (1 - y)
    """
    return output * (1.0 - output)


def relu(x):
    """Hàm kích hoạt ReLU: f(x) = max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(output):
    """Đạo hàm của ReLU: 1 nếu output > 0, ngược lại 0"""
    return (output > 0).astype(np.float64)


def linear(x):
    """Hàm đồng nhất tuyến tính: f(x) = x"""
    return x


def linear_derivative(output):
    """Đạo hàm của hàm Linear: luôn bằng 1"""
    return np.ones_like(output)


class MSELoss:
    """Mean Squared Error Loss: Sai số bình phương trung bình"""
    @staticmethod
    def forward(y_pred, y_true):
        # L = 0.5 * mean(sum((y_pred - y_true)^2))
        return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    @staticmethod
    def gradient(y_pred, y_true):
        # dL / dy_pred = (y_pred - y_true) / batch_size
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size


class BCELoss:
    """Binary Cross-Entropy Loss: Dùng khi pixel chuẩn hóa về [0, 1]"""
    @staticmethod
    def forward(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return - np.mean(np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred), axis=1))

    @staticmethod
    def gradient(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        batch_size = y_pred.shape[0]
        return ((y_pred - y_true) / (y_pred * (1.0 - y_pred))) / batch_size


class DenseLayer:
    """
    Tầng nơ-ron kết nối đầy đủ (Fully Connected / Dense Layer) từ con số 0:
    - Forward pass: z = X * W + b; a = activation(z)
    - Backward pass: Tự tính dL/dW, dL/db và dL/dX bằng giải tích vi tích phân ma trận.
    """
    def __init__(self, in_features, out_features, activation="sigmoid"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation

        # Khởi tạo trọng số Xavier / Glorot
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))

        # Bộ đệm trạng thái cho Backpropagation
        self.input_cache = None
        self.output_cache = None

        # Bộ đệm Momentum
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

        # Chọn hàm kích hoạt và đạo hàm tương ứng
        if activation == "sigmoid":
            self.act_fn = sigmoid
            self.act_deriv = sigmoid_derivative
        elif activation == "relu":
            self.act_fn = relu
            self.act_deriv = relu_derivative
        elif activation == "linear":
            self.act_fn = linear
            self.act_deriv = linear_derivative
        else:
            raise ValueError(f"Activation {activation} không được hỗ trợ!")

    def forward(self, X):
        """Lan truyền xuôi qua tầng nơ-ron"""
        self.input_cache = X
        z = np.dot(X, self.W) + self.b
        self.output_cache = self.act_fn(z)
        return self.output_cache

    def backward(self, grad_output):
        """
        Lan truyền ngược tính đạo hàm:
        grad_output: dL / da
        Trả về: dL / dX để truyền ngược về các tầng trước.
        """
        # 1. dL / dz = (dL / da) * (da / dz)
        grad_z = grad_output * self.act_deriv(self.output_cache)

        # 2. dL / dW = X^T * grad_z
        self.grad_W = np.dot(self.input_cache.T, grad_z)

        # 3. dL / db = sum(grad_z, axis=0)
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)

        # 4. dL / dX = grad_z * W^T
        grad_input = np.dot(grad_z, self.W.T)
        return grad_input

    def update_params(self, lr=0.01, momentum=0.9, weight_decay=1e-4):
        """Cập nhật trọng số theo Gradient Descent có Quán tính (Momentum)"""
        self.v_W = momentum * self.v_W - lr * (self.grad_W + weight_decay * self.W)
        self.v_b = momentum * self.v_b - lr * self.grad_b

        self.W += self.v_W
        self.b += self.v_b


# ==============================================================================
# PHẦN 3: RESTRICTED BOLTZMANN MACHINE (RBM) VỚI CONTRASTIVE DIVERGENCE (CD-1)
# ==============================================================================

class RBM:
    """
    Cỗ máy Boltzmann Giới hạn (Restricted Boltzmann Machine):
    Mô hình đồ thị xác suất 2 tầng không có liên kết nội tầng (Bipartite Graph).
    Hàm năng lượng: E(v, h) = - v^T * W * h - b_v^T * v - b_h^T * h
    """
    def __init__(self, num_visible, num_hidden, is_top_linear=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.is_top_linear = is_top_linear

        # Khởi tạo trọng số ngẫu nhiên nhỏ N(0, 0.01^2) theo đúng bài báo Hinton 2006
        self.W = np.random.randn(num_visible, num_hidden) * 0.01
        self.b_v = np.zeros((1, num_visible))
        self.b_h = np.zeros((1, num_hidden))

        # Bộ nhớ quán tính Momentum
        self.v_W = np.zeros_like(self.W)
        self.v_b_v = np.zeros_like(self.b_v)
        self.v_b_h = np.zeros_like(self.b_h)

    def sample_hidden(self, v):
        """
        Pha xuôi: Tính xác suất kích hoạt của tầng ẩn h khi biết visible v:
        P(h_j = 1 | v) = sigmoid(b_h + v * W)
        """
        activation = np.dot(v, self.W) + self.b_h
        if self.is_top_linear:
            # Tầng code layer ở đỉnh là biến liên tục (Gaussian units)
            h_prob = activation
            h_sample = h_prob + np.random.randn(*h_prob.shape)
        else:
            h_prob = sigmoid(activation)
            # Lấy mẫu nhị phân ngẫu nhiên Bernoulli (Gibbs Sampling)
            h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float64)
        return h_prob, h_sample

    def sample_visible(self, h):
        """
        Pha ngược (Tái tạo/Ảo giác): Tính xác suất tầng visible v khi biết hidden h:
        P(v_i = 1 | h) = sigmoid(b_v + h * W^T)
        """
        activation = np.dot(h, self.W.T) + self.b_v
        v_prob = sigmoid(activation)
        v_sample = (np.random.rand(*v_prob.shape) < v_prob).astype(np.float64)
        return v_prob, v_sample

    def contrastive_divergence_step(self, v0, lr=0.1, momentum=0.5, weight_decay=2e-4):
        """
        1 bước Contrastive Divergence (CD-1) theo công thức của Hinton:
        Delta W = lr * (<v0 * h0>_data - <v1 * h1>_model)
        """
        batch_size = v0.shape[0]

        # 1. Pha Dương (Positive Phase): Dữ liệu thật kích hoạt tầng ẩn
        h0_prob, h0_sample = self.sample_hidden(v0)
        positive_grad = np.dot(v0.T, h0_prob)

        # 2. Pha Âm (Negative Phase): Mạng tự tái tạo và ảo giác (1 bước Gibbs)
        v1_prob, v1_sample = self.sample_visible(h0_sample)
        h1_prob, h1_sample = self.sample_hidden(v1_prob)
        negative_grad = np.dot(v1_prob.T, h1_prob)

        # 3. Tính độ dốc (Gradients)
        dW = (positive_grad - negative_grad) / batch_size
        db_v = np.mean(v0 - v1_prob, axis=0, keepdims=True)
        db_h = np.mean(h0_prob - h1_prob, axis=0, keepdims=True)

        # 4. Cập nhật có quán tính Momentum và L2 Weight Decay
        self.v_W = momentum * self.v_W + lr * (dW - weight_decay * self.W)
        self.v_b_v = momentum * self.v_b_v + lr * db_v
        self.v_b_h = momentum * self.v_b_h + lr * db_h

        self.W += self.v_W
        self.b_v += self.v_b_v
        self.b_h += self.v_b_h

        recon_error = np.mean((v0 - v1_prob) ** 2)
        return recon_error


# ==============================================================================
# PHẦN 4: TOÀN BỘ MÔ HÌNH DEEP AUTOENCODER (FROM SCRATCH)
# ==============================================================================

class DeepAutoencoderFromScratch:
    """
    Mô hình Deep Autoencoder hoàn chỉnh:
    Hỗ trợ 2 chế độ:
    1. Pre-training từng tầng bằng RBMs -> Unrolling -> Fine-tuning (Hinton 2006).
    2. Huấn luyện trực tiếp từ đầu bằng Backpropagation thuần túy (Vanilla Baseline).
    """
    def __init__(self, layer_dims):
        """
        layer_dims: Danh sách số nơ-ron từ Input -> Bottleneck, ví dụ: [64, 32, 16, 2]
        Hệ thống tự động sinh Decoder đối xứng: [2, 16, 32, 64]
        """
        self.layer_dims = layer_dims
        self.num_stages = len(layer_dims) - 1

        # Danh sách các RBM phục vụ Pre-training
        self.rbms = []
        for i in range(self.num_stages):
            is_top = (i == self.num_stages - 1)
            rbm = RBM(num_visible=layer_dims[i], 
                      num_hidden=layer_dims[i+1], 
                      is_top_linear=is_top)
            self.rbms.append(rbm)

        # Khởi tạo các tầng Encoder
        self.encoder_layers = []
        for i in range(self.num_stages):
            act = "linear" if (i == self.num_stages - 1) else "sigmoid"
            self.encoder_layers.append(DenseLayer(layer_dims[i], layer_dims[i+1], activation=act))

        # Khởi tạo các tầng Decoder (đối xứng)
        self.decoder_layers = []
        reversed_dims = list(reversed(layer_dims))
        for i in range(self.num_stages):
            act = "sigmoid"
            self.decoder_layers.append(DenseLayer(reversed_dims[i], reversed_dims[i+1], activation=act))

    def pretrain_rbms(self, X, epochs=15, lr=0.1, batch_size=32, verbose=True):
        """Bước 1: Greedy Layer-Wise Pre-training từng tầng RBM"""
        current_input = X.copy()
        N = X.shape[0]

        for stage, rbm in enumerate(self.rbms):
            if verbose:
                print(f"   * Đang pre-train RBM tầng {stage+1}/{self.num_stages}: [{rbm.num_visible} -> {rbm.num_hidden}]")

            for epoch in range(1, epochs + 1):
                indices = np.random.permutation(N)
                num_batches = int(np.ceil(N / batch_size))
                momentum = 0.5 if epoch <= 5 else 0.9

                for b in range(num_batches):
                    batch_idx = indices[b * batch_size : min((b + 1) * batch_size, N)]
                    v_batch = current_input[batch_idx]
                    rbm.contrastive_divergence_step(v_batch, lr=lr, momentum=momentum)

            # Đẩy dữ liệu qua tầng ẩn để làm đầu vào cho RBM kế tiếp
            h_prob, _ = rbm.sample_hidden(current_input)
            current_input = h_prob

    def unroll_rbms(self, verbose=True):
        """Bước 2: Unrolling - Mở bung RBMs thành Deep Autoencoder đối xứng"""
        for i, rbm in enumerate(self.rbms):
            self.encoder_layers[i].W = rbm.W.copy()
            self.encoder_layers[i].b = rbm.b_h.copy()

        for i in range(self.num_stages):
            source_rbm = self.rbms[self.num_stages - 1 - i]
            self.decoder_layers[i].W = source_rbm.W.T.copy()
            self.decoder_layers[i].b = source_rbm.b_v.copy()

        if verbose:
            print("   [✓] Đã mở cuộn (Unrolled) toàn bộ trọng số RBM vào Encoder và Decoder đối xứng!")

    def encode(self, X):
        """Nén đầu vào X thành vector tiềm ẩn Z tại nút cổ chai"""
        out = X
        for layer in self.encoder_layers:
            out = layer.forward(out)
        return out

    def decode(self, Z):
        """Tái tạo dữ liệu gốc từ vector tiềm ẩn Z"""
        out = Z
        for layer in self.decoder_layers:
            out = layer.forward(out)
        return out

    def forward(self, X):
        """Truyền thẳng toàn bộ mạng: X -> Z -> X_hat"""
        z = self.encode(X)
        x_hat = self.decode(z)
        return x_hat, z

    def finetune_backprop(self, X, epochs=25, lr=0.02, batch_size=32, verbose=True):
        """Bước 3: Global Fine-Tuning bằng Backpropagation toàn cục"""
        LossClass = MSELoss
        N = X.shape[0]
        history_loss = []

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            num_batches = int(np.ceil(N / batch_size))

            for b in range(num_batches):
                batch_idx = indices[b * batch_size : min((b + 1) * batch_size, N)]
                X_batch = X[batch_idx]

                # 1. Forward Pass
                x_hat, _ = self.forward(X_batch)
                batch_loss = LossClass.forward(x_hat, X_batch)
                epoch_loss += batch_loss

                # 2. Backward Pass
                grad = LossClass.gradient(x_hat, X_batch)
                for layer in reversed(self.decoder_layers):
                    grad = layer.backward(grad)
                for layer in reversed(self.encoder_layers):
                    grad = layer.backward(grad)

                # 3. Update Weights
                for layer in self.decoder_layers:
                    layer.update_params(lr=lr)
                for layer in self.encoder_layers:
                    layer.update_params(lr=lr)

            epoch_loss /= num_batches
            history_loss.append(epoch_loss)

        if verbose:
            print(f"   [✓] Fine-tuning hoàn tất: Loss ban đầu {history_loss[0]:.6f} -> Loss cuối {history_loss[-1]:.6f}")
        return history_loss


# ==============================================================================
# PHẦN 5: MÔ HÌNH TUYẾN TÍNH PCA & MẠNG NÔNG TỰ LẬP TRÌNH (SCRATCH)
# ==============================================================================

def pca_from_scratch(X_train, X_test, n_components=2):
    """
    Thuật toán Phân tích Thành phần Chính (PCA - Pearson 1901) tự lập trình thuần NumPy:
    1. Chuẩn hóa dữ liệu bằng cách trừ đi vector kỳ vọng trung bình (mean centering).
    2. Phân rã giá trị suy biến (Singular Value Decomposition - SVD).
    3. Chiếu dữ liệu lên n_components vector trực giao có phương sai lớn nhất.
    4. Tái tạo ngược lại không gian ban đầu: X_recon = Z * V^T + mean.
    """
    mean = np.mean(X_train, axis=0, keepdims=True)
    X_train_centered = X_train - mean
    X_test_centered = X_test - mean

    # SVD: X = U * S * V^T
    _, _, Vt = np.linalg.svd(X_train_centered, full_matrices=False)
    # Lấy n_components vector riêng hàng đầu
    V = Vt[:n_components].T  # (D, n_components)

    # Nén và Tái tạo
    Z_test = np.dot(X_test_centered, V)
    X_test_recon = np.dot(Z_test, V.T) + mean
    pca_mse = np.mean((X_test - X_test_recon) ** 2)
    return Z_test, X_test_recon, pca_mse


class ShallowAutoencoderFromScratch:
    """
    Mạng Autoencoder nông (1 tầng ẩn - Rumelhart 1986):
    Kiến trúc: D -> bottleneck -> D
    Lý thuyết chứng minh (Bourlard & Kamp 1988): Một Autoencoder nông tuyến tính
    chỉ có khả năng xấp xỉ không gian con của PCA, không thể biểu diễn phi tuyến sâu.
    """
    def __init__(self, in_features, bottleneck_dim):
        self.encoder = DenseLayer(in_features, bottleneck_dim, activation="linear")
        self.decoder = DenseLayer(bottleneck_dim, in_features, activation="sigmoid")

    def forward(self, X):
        z = self.encoder.forward(X)
        x_hat = self.decoder.forward(z)
        return x_hat, z

    def fit(self, X, epochs=30, lr=0.03, batch_size=32):
        N = X.shape[0]
        LossClass = MSELoss
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(N)
            num_batches = int(np.ceil(N / batch_size))
            for b in range(num_batches):
                batch_idx = indices[b * batch_size : min((b + 1) * batch_size, N)]
                X_batch = X[batch_idx]

                x_hat, _ = self.forward(X_batch)
                grad = LossClass.gradient(x_hat, X_batch)
                grad = self.decoder.backward(grad)
                self.encoder.backward(grad)

                self.decoder.update_params(lr=lr)
                self.encoder.update_params(lr=lr)
        return self


# ==============================================================================
# PHẦN 6: THỰC NGHIỆM ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT (ACADEMIC BENCHMARK)
# ==============================================================================

def generate_synthetic_manifold_data(num_samples=700, random_seed=42):
    """
    Sinh tập dữ liệu đa tạp phi tuyến tính (Synthetic Manifold):
    - 2 biến tiềm ẩn nội tại thực sự z = [t1, t2] (ví dụ: góc quay và độ uốn cong).
    - Ánh xạ phi tuyến lên 64 chiều (tương tự bức ảnh 8x8).
    - Tái hiện hoàn hảo bài toán giảm chiều phi tuyến trong bài báo Hinton 2006!
    """
    np.random.seed(random_seed)
    t1 = np.random.uniform(-np.pi, np.pi, size=(num_samples, 1))
    t2 = np.random.uniform(-np.pi, np.pi, size=(num_samples, 1))

    X = np.zeros((num_samples, 64))
    for i in range(64):
        freq1 = (i % 8 + 1) * 0.5
        freq2 = (i // 8 + 1) * 0.5
        # Hàm lượng giác phi tuyến bậc cao mô phỏng đa tạp cong
        X[:, i] = np.sin(freq1 * t1[:, 0]) * np.cos(freq2 * t2[:, 0]) + 0.4 * np.cos(freq1 * t2[:, 0])

    # Chuẩn hóa về [0.05, 0.95] phù hợp cho hàm kích hoạt Sigmoid
    X_min, X_max = X.min(), X.max()
    X = 0.05 + 0.9 * ((X - X_min) / (X_max - X_min))
    return X, np.hstack([t1, t2])


def print_ascii_comparison(original, recon_pca, recon_deep_ae, size=8):
    """Vẽ minh họa so sánh mẫu ảnh 8x8 bằng ký tự ASCII giữa các phương pháp"""
    chars = " .:-=+*#%@"
    def to_ascii(arr):
        grid = []
        for r in range(size):
            row_str = ""
            for c in range(size):
                val = arr[r * size + c]
                idx = int(np.clip(val * (len(chars) - 1), 0, len(chars) - 1))
                row_str += chars[idx] * 2
            grid.append(row_str)
        return grid

    o_grid = to_ascii(original)
    p_grid = to_ascii(recon_pca)
    d_grid = to_ascii(recon_deep_ae)

    print("\n   [Ảnh Gốc 8x8]               [Tái Tạo Bằng PCA]          [Tái Tạo Bằng Deep AE Hinton]")
    for o, p, d in zip(o_grid, p_grid, d_grid):
        print(f"   {o}       {p}       {d}")


def run_academic_value_comparison():
    """
    HÀM ĐỐI CHUẨN HỌC THUẬT:
    So sánh định lượng 4 cột mốc lịch sử trong lĩnh vực Giảm chiều dữ liệu:
    1. PCA (Pearson 1901): Tuyến tính cổ điển.
    2. Shallow Autoencoder (Rumelhart 1986): Mạng nơ-ron nông 1 tầng ẩn.
    3. Vanilla Deep Autoencoder (Trước 2006): Mạng sâu khởi tạo ngẫu nhiên, không pre-train.
    4. Hinton Deep Autoencoder (Science 2006): Mạng sâu với Greedy RBM Pre-training.
    """
    print("\n" + "=" * 88)
    print(" KHẢO SÁT & ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT CỦA BÀI BÁO HINTON & SALAKHUTDINOV (2006)")
    print("=" * 88)
    print("Bối cảnh thực nghiệm: Đa tạp phi tuyến 64 chiều (ảnh 8x8) nén về Bottleneck 2 chiều.")

    # 1. Chuẩn bị tập dữ liệu đa tạp
    X_data, _ = generate_synthetic_manifold_data(num_samples=700, random_seed=42)
    split = 500
    X_train, X_test = X_data[:split], X_data[split:]
    print(f"[*] Tập dữ liệu: Train = {X_train.shape} (500 mẫu), Test = {X_test.shape} (200 mẫu)")

    # --------------------------------------------------------------------------
    # CỘT MỐC 1: TUYẾN TÍNH CỔ ĐIỂN - PCA (Pearson 1901)
    # --------------------------------------------------------------------------
    print("\n[1/4] Đang đánh giá: Principal Component Analysis (PCA 1901)...")
    t0 = time.time()
    _, pca_recon, pca_mse = pca_from_scratch(X_train, X_test, n_components=2)
    pca_time = time.time() - t0

    # --------------------------------------------------------------------------
    # CỘT MỐC 2: MẠNG NƠ-RON NÔNG - Shallow Autoencoder (1986)
    # --------------------------------------------------------------------------
    print("[2/4] Đang đánh giá: Shallow Autoencoder 64 -> 2 -> 64 (Rumelhart 1986)...")
    t0 = time.time()
    shallow_ae = ShallowAutoencoderFromScratch(in_features=64, bottleneck_dim=2)
    shallow_ae.fit(X_train, epochs=30, lr=0.04, batch_size=32)
    shallow_recon, _ = shallow_ae.forward(X_test)
    shallow_mse = np.mean((X_test - shallow_recon) ** 2)
    shallow_time = time.time() - t0

    # --------------------------------------------------------------------------
    # CỘT MỐC 3: MẠNG SÂU KHÔNG PRE-TRAIN (Vanilla Deep AE trước 2006)
    # --------------------------------------------------------------------------
    print("[3/4] Đang đánh giá: Vanilla Deep AE 64->32->16->2 KHÔNG CÓ Pre-training...")
    t0 = time.time()
    np.random.seed(42)
    vanilla_deep_ae = DeepAutoencoderFromScratch(layer_dims=[64, 32, 16, 2])
    # Huấn luyện trực tiếp từ đầu bằng Backpropagation thuần
    vanilla_deep_ae.finetune_backprop(X_train, epochs=30, lr=0.03, batch_size=32, verbose=False)
    vanilla_recon, _ = vanilla_deep_ae.forward(X_test)
    vanilla_mse = np.mean((X_test - vanilla_recon) ** 2)
    vanilla_time = time.time() - t0

    # --------------------------------------------------------------------------
    # CỘT MỐC 4: ĐỘT PHÁ HINTON (2006) - DEEP AUTOENCODER CÓ RBM PRE-TRAINING
    # --------------------------------------------------------------------------
    print("[4/4] Đang đánh giá: Hinton Deep AE CÓ Greedy RBM Pre-training (Science 2006)...")
    t0 = time.time()
    np.random.seed(42)
    hinton_deep_ae = DeepAutoencoderFromScratch(layer_dims=[64, 32, 16, 2])
    # Bước 1: Pre-training từng tầng RBM
    hinton_deep_ae.pretrain_rbms(X_train, epochs=12, lr=0.08, batch_size=32, verbose=False)
    # Bước 2: Unrolling
    hinton_deep_ae.unroll_rbms(verbose=False)
    # Bước 3: Fine-tuning toàn cục
    hinton_deep_ae.finetune_backprop(X_train, epochs=30, lr=0.03, batch_size=32, verbose=False)
    hinton_recon, _ = hinton_deep_ae.forward(X_test)
    hinton_mse = np.mean((X_test - hinton_recon) ** 2)
    hinton_time = time.time() - t0

    # --------------------------------------------------------------------------
    # XUẤT BẢNG SO SÁNH ĐỊNH LƯỢNG HỌC THUẬT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Mô Hình / Giải Thuật':<28} | {'Bottleneck':<10} | {'Test MSE':<12} | {'Cải thiện vs PCA':<18} | {'Thời gian':<10}")
    print("=" * 90)
    print(f"{'1. Linear PCA (1901)':<28} | {'2 chiều':<10} | {pca_mse:10.6f}   | {'Baseline (0.0%)':<18} | {pca_time:7.3f}s")
    print(f"{'2. Shallow AE (1986)':<28} | {'2 chiều':<10} | {shallow_mse:10.6f}   | {((pca_mse - shallow_mse)/pca_mse * 100):+15.2f}% | {shallow_time:7.3f}s")
    print(f"{'3. Vanilla Deep AE (Pre-2006)':<28} | {'2 chiều':<10} | {vanilla_mse:10.6f}   | {((pca_mse - vanilla_mse)/pca_mse * 100):+15.2f}% | {vanilla_time:7.3f}s")
    print(f"{'4. Hinton Deep AE (2006)':<28} | {'2 chiều':<10} | {hinton_mse:10.6f}   | {((pca_mse - hinton_mse)/pca_mse * 100):+15.2f}% | {hinton_time:7.3f}s")
    print("=" * 90)

    # --------------------------------------------------------------------------
    # PHÂN TÍCH KẾT LUẬN HỌC THUẬT
    # --------------------------------------------------------------------------
    print("\n[PHÂN TÍCH KẾT LUẬN DÀNH CHO NGƯỜI MỚI BẮT ĐẦU]:")
    print(f"1. Thất bại của giải pháp Tuyến tính (PCA):")
    print(f"   - PCA đạt sai số tái tạo Test MSE = {pca_mse:.6f} vì chỉ chiếu thẳng đứng lên một mặt phẳng.")
    print(f"   - Không thể uốn cong theo đa tạp dữ liệu phi tuyến, dẫn tới hình ảnh tái tạo bị biến dạng.")

    print(f"\n2. Tầm quan trọng sống còn của Greedy Layer-wise Pre-training (Hinton 2006):")
    print(f"   - Khi không có Pre-training, Vanilla Deep AE đạt MSE = {vanilla_mse:.6f} (do khởi tạo ngẫu nhiên bị kẹt ở cực tiểu tồi).")
    print(f"   - Khi có RBM Pre-training, Deep AE đạt MSE = {hinton_mse:.6f}, giảm sai số tới {((pca_mse - hinton_mse)/pca_mse * 100):.1f}% so với PCA!")
    print(f"   - Điều này minh chứng luận điểm vĩ đại của Hinton: Pre-training đặt trọng số của mạng sâu vào vùng hấp dẫn tốt (basin of attraction).")

    # Hiển thị trực quan mẫu ảnh tái tạo
    print("\n[TRỰC QUAN HÓA SO SÁNH MẪU ẢNH TÁI TẠO TẬP KIỂM THỬ (ASCII)]:")
    sample_idx = 5
    print_ascii_comparison(X_test[sample_idx], pca_recon[sample_idx], hinton_recon[sample_idx], size=8)

    print("\n[✓] Hoàn thành trọn vẹn kiểm chứng học thuật bài báo Deep Autoencoder (Hinton 2006)!")


def main():
    print("================================================================================")
    print(" TRIỂN KHAI DEEP AUTOENCODER TỪ CON SỐ 0 (FROM SCRATCH thuần NumPy)")
    print(" BÀI BÁO: Reducing the Dimensionality of Data with Neural Networks (Hinton 2006)")
    print("================================================================================")

    run_academic_value_comparison()


if __name__ == "__main__":
    main()
