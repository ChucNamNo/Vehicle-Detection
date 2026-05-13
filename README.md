# Vehicle Detection System in Vietnam (YOLOv8 & HOG+SVM)

## 1. Project Overview
[cite_start]Dự án tập trung vào việc xây dựng một hệ thống tự động nhận diện và phân loại các phương tiện giao thông trong môi trường thực tế tại Việt Nam[cite: 44].

* [cite_start]Bài toán cốt lõi: Xác định vị trí (Bounding Box) và gán nhãn loại phương tiện từ dữ liệu video/hình ảnh trích xuất từ camera giám sát[cite: 44, 45].
* [cite_start]Đối tượng nhận diện: Gồm 8 nhóm chính: Xe máy, xe hơi, xe tải, xe buýt, xe cứu hỏa, xe van, xe đạp và xe container[cite: 61].
* [cite_start]Công nghệ sử dụng: Ngôn ngữ Python, thư viện OpenCV, PyTorch và Ultralytics (YOLOv8)[cite: 65].
* [cite_start]Mô hình triển khai: Ứng dụng Web sử dụng Flask Framework chạy trên thiết bị Macbook Air M1[cite: 66, 67].

## 2. Project Objectives
* [cite_start]Nghiên cứu và ứng dụng YOLOv8n: Sử dụng phiên bản Nano nhẹ nhất, tối ưu cho thiết bị tài nguyên hạn chế nhưng đảm bảo xử lý thời gian thực[cite: 27, 463, 469].
* [cite_start]So sánh hiệu năng: Thực nghiệm đối chứng giữa Deep Learning (YOLOv8n) và học máy truyền thống (HOG + SVM) để làm rõ sự tiến hóa của công nghệ[cite: 64, 282].
* [cite_start]Hướng tới giao thông thông minh: Cung cấp cơ sở cho việc đếm lưu lượng xe, hỗ trợ điều khiển tín hiệu đèn và giám sát vi phạm[cite: 15, 1018, 1019].

## 3. Đánh giá kết quả thực nghiệm

### 3.1. Phân tích Learning Curves (50 Epochs)

#### a. Box Loss Learning Curve (Tổn thất định vị)
[cite_start]Biểu đồ thể hiện sự biến thiên của hàm mất mát về vị trí khung bao trên tập huấn luyện và tập kiểm thử[cite: 779].
* [cite_start]Nhận xét: Cả Train Loss và Validation Loss đều giảm nhanh trong 10 epochs đầu tiên[cite: 782].
* [cite_start]Giai đoạn 1-40: Đường Validation Loss luôn duy trì ở mức thấp hơn Train Loss do kỹ thuật điều chuẩn (Regularization) áp dụng trong huấn luyện[cite: 783, 785].
* [cite_start]Giai đoạn 41-50: Hai đường bắt đầu giao nhau, cho thấy dấu hiệu chớm bắt đầu của hiện tượng quá khớp (Overfitting)[cite: 788, 792].

#### b. Classification Loss Learning Curve (Tổn thất phân loại)
[cite_start]Biểu đồ đo lường độ chính xác trong việc gán nhãn đúng cho đối tượng[cite: 816].
* [cite_start]Nhận xét: Train Loss giảm mạnh từ mức 3.4 xuống 1.2 trong 5 epochs đầu, cho thấy mô hình học đặc trưng rất nhanh[cite: 819].
* [cite_start]Trạng thái: Cả hai chỉ số hội tụ về mức thấp quanh ngưỡng 0.6-0.7 tại epoch 50[cite: 821].
* [cite_start]Kết luận: Validation Loss thấp hơn Train Loss và không có dấu hiệu tăng ngược trở lại chứng tỏ mô hình không bị Overfitting ở mảng phân loại[cite: 823].

### 3.2. Confusion Matrix (Ma trận nhầm lẫn)
* [cite_start]Độ chính xác cao: Xe container (100%), xe cứu hỏa (98%), xe hơi (94%) và xe máy (93%)[cite: 888, 889].
* [cite_start]Hạn chế: Xe đạp đạt độ chính xác thấp nhất (59%) do tiết diện nhỏ dễ bị lẫn vào nền[cite: 890, 891].
* [cite_start]Sai số: 17% mẫu xe Van bị nhận nhầm thành xe hơi do tương đồng về đặc điểm hình học[cite: 895, 896].
* [cite_start]Dương tính giả: Tại các vùng nền trống, mô hình có xu hướng nhạy cảm quá mức dẫn đến nhận nhầm xe máy (67%) và xe hơi (25%)[cite: 903, 904].

## 4. Performance Metrics

| Metric | Giá trị | Mô tả |
| :--- | :---: | :--- |
| mAP@0.5 | 0.87 | [cite_start]Khả năng phát hiện và khoanh vùng đối tượng tốt[cite: 929]. |
| Recall | 0.86 | [cite_start]Tìm được 86% số lượng đối tượng thực tế, hạn chế bỏ sót[cite: 929]. |
| Precision | 0.78 | [cite_start]Tỷ lệ dự đoán đúng trên tổng dự đoán (vẫn còn dương tính giả)[cite: 931]. |
| F1-Score | 0.82 | [cite_start]Sự cân bằng giữa Precision và Recall[cite: 931]. |

## 5. Tổng Kết & Quan Sát
* [cite_start]Sự hội tụ: Các đường cong Loss giảm đều và tiệm cận mức thấp nhất, quá trình huấn luyện ổn định[cite: 907, 908].
* [cite_start]Hiệu suất vượt trội: YOLOv8n đạt tốc độ ~35 FPS, cao gấp 6 lần so với HOG+SVM (~5.78 FPS) trên cùng phần cứng M1[cite: 967, 968].
* [cite_start]Kết luận: HOG+SVM phù hợp cho mục đích học tập hoặc nền tĩnh, trong khi YOLOv8n là lựa chọn tối ưu cho bài toán thực tế yêu cầu thời gian thực[cite: 971, 972].

---
*Created by Pham Long Chuc - ICTU*
