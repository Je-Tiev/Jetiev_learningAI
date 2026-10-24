# 20 AI Research Papers Quan Trọng Nhất Của Thế Kỷ 21

Tập hợp các bài báo khoa học kinh điển đã khai sinh và thúc đẩy cuộc cách mạng Trí tuệ Nhân tạo & Học sâu (Deep Learning) từ những năm 2000 đến nay.

> 📖 **Quy chuẩn triển khai:**  
> Toàn bộ các bài báo trong chuyên đề này được định hướng triển khai đồng bộ theo **Quy chuẩn 4 thành phần & Khung lý thuyết 10 mục** đã hoàn thành mẫu tại [`Day 02 - Autoencoder`](./Day%2002%20-%20Autoencoder).  
> Chi tiết tiêu chuẩn kỹ thuật xem tại: [Tài liệu hướng dẫn tổng quan `papers-reading/README.md`](../README.md).

---

## Cấu Trúc Chuẩn Cho Từng Bài Báo

Mỗi thư mục bài báo độc lập bao gồm 4 phần:
1. 📄 **`README.md`**: Báo cáo phân tích chuyên sâu 10 mục bằng Tiếng Việt (Ý nghĩa lịch sử, bối cảnh toán học, kiến trúc, đột phá, quy trình huấn luyện, bảng đối chuẩn thực nghiệm, di sản/tiến hóa, hướng dẫn code, câu hỏi ôn tập).
2. ⚙️ **`[model]_scratch.py`**: Tự cài đặt thuật toán từ đầu bằng Python thuần + NumPy (tự tính đạo hàm manual backprop, kiểm thử toán học).
3. 🚀 **`[model]_pytorch.py`**: Triển khai tối ưu hóa bằng PyTorch (`torch.nn.Module`, fallback dataset an toàn khi offline, trực quan hóa biểu đồ ra thư mục `outputs/`).
4. 📚 **`resources/`**: Chứa tệp bài báo PDF gốc.

---

## Danh Sách Các Bài Báo & Tiến Độ Thực Hiện

| STT | Bài Báo / Chủ Đề | Tác Giả & Xuất Bản | Thư Mục | Tiến Độ & Trạng Thái |
|:---:|---|---|---|:---:|
| 02 | **Autoencoder** (*Reducing the Dimensionality of Data with Neural Networks*) | G. E. Hinton, R. R. Salakhutdinov (Science 2006) | [`Day 02 - Autoencoder/`](./Day%2002%20-%20Autoencoder) | ✅ **Mẫu chuẩn hoàn thành (Đủ 4 phần)** |
| 03 | **t-SNE** (*Visualizing Data using t-SNE*) | L. van der Maaten, G. Hinton (JMLR 2008) | [`Day 03 - Visualizing Data using t-SNE/`](./Day%2003%20-%20Visualizing%20Data%20using%20t-SNE) | ✅ **Đã hoàn thành (Đủ 4 phần)** |
| 04 | **ImageNet** (*ImageNet: A Large-Scale Hierarchical Image Database*) | J. Deng et al. (CVPR 2009) | [`Day 04 - ImageNet/`](./Day%2004%20-%20ImageNet) | ✅ **Đã hoàn thành (Đủ 4 phần)** |
| 05 | **AlexNet** (*ImageNet Classification with Deep CNNs*) | A. Krizhevsky, I. Sutskever, G. Hinton (NeurIPS 2012) | [`Day 05 - AlexNet/`](./Day%2005%20-%20AlexNet) | ✅ **Đã hoàn thành (Đủ 4 phần)** |
| 06 | **Word2Vec** (*Distributed Representations of Words and Phrases*) | T. Mikolov et al. (NeurIPS 2013) | [`Day 06 - Word2Vec/`](./Day%2006%20-%20Word2Vec) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 07 | **VAE** (*Auto-Encoding Variational Bayes*) | D. P. Kingma, M. Welling (ICLR 2014) | [`Day 07 - VAE - Auto-Encoding Variational Bayes/`](./Day%2007%20-%20VAE%20-%20Auto-Encoding%20Variational%20Bayes) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 08 | **GAN** (*Generative Adversarial Nets*) | I. Goodfellow et al. (NeurIPS 2014) | [`Day 08 - GAN/`](./Day%2008%20-%20GAN) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 09 | **Dropout** (*A Simple Way to Prevent Neural Networks from Overfitting*) | N. Srivastava et al. (JMLR 2014) | [`Day 09 - Dropout/`](./Day%2009%20-%20Dropout) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 10 | **Adam** (*A Method for Stochastic Optimization*) | D. P. Kingma, J. Ba (ICLR 2015) | [`Day 10 - Adam/`](./Day%2010%20-%20Adam) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 11 | **BatchNorm** (*Accelerating Deep Network Training*) | S. Ioffe, C. Szegedy (ICML 2015) | [`Day 11 - BatchNorm/`](./Day%2011%20-%20BatchNorm) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 12 | **Deep Q-Network** (*Human-level control through deep RL*) | V. Mnih et al. (Nature 2015) | [`Day 12 - Deep Q Network/`](./Day%2012%20-%20Deep%20Q%20Network) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 13 | **Faster R-CNN** (*Towards Real-Time Object Detection*) | S. Ren et al. (NeurIPS 2015) | [`Day 13 - Faster R-CNN/`](./Day%2013%20-%20Faster%20R-CNN) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 14 | **U-Net** (*Convolutional Networks for Biomedical Segmentation*) | O. Ronneberger et al. (MICCAI 2015) | [`Day 14 - U-Net/`](./Day%2014%20-%20U-Net) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 15 | **ResNet** (*Deep Residual Learning for Image Recognition*) | K. He, X. Zhang, S. Ren, J. Sun (CVPR 2016) | [`Day 15 - ResNet/`](./Day%2015%20-%20ResNet) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 16 | **YOLO** (*You Only Look Once: Unified, Real-Time Object Detection*) | J. Redmon et al. (CVPR 2016) | [`Day 16 - YOLO/`](./Day%2016%20-%20YOLO) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 17 | **Transformer** (*Attention Is All You Need*) | A. Vaswani et al. (NeurIPS 2017) | [`Day 17 - Transformer/`](./Day%2017%20-%20Transformer) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 18 | **BERT** (*Pre-training of Deep Bidirectional Transformers*) | J. Devlin et al. (NAACL 2019) | [`Day 18 - BERT/`](./Day%2018%20-%20BERT) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 19 | **GPT-3** (*Language Models are Few-Shot Learners*) | T. Brown et al. (NeurIPS 2020) | [`Day 19 - GPT-3/`](./Day%2019%20-%20GPT-3) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
| 20 | **ViT** (*An Image is Worth 16x16 Words: Transformers for Image Recognition*) | A. Dosovitskiy et al. (ICLR 2021) | [`Day 20 - ViT/`](./Day%2020%20-%20ViT) | 📚 Đã có PDF gốc (Chờ cấp phép triển khai) |
