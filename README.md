# Vehicle Detection System in Vietnam (YOLOv8 & HOG+SVM)

## 1. Project Overview
Dự án tập trung vào việc xây dựng một hệ thống tự động nhận diện và phân loại các phương tiện giao thông trong môi trường thực tế tại Việt Nam.

* Bài toán cốt lõi: Xác định vị trí (Bounding Box) và gán nhãn loại phương tiện từ dữ liệu video/hình ảnh trích xuất từ camera giám sát.
* Đối tượng nhận diện: Gồm 8 nhóm chính: Xe máy, xe hơi, xe tải, xe buýt, xe cứu hỏa, xe van, xe đạp và xe container.
* Công nghệ sử dụng: Ngôn ngữ Python, thư viện OpenCV, PyTorch và Ultralytics (YOLOv8).
* Mô hình triển khai: Ứng dụng Web sử dụng Flask Framework chạy trên thiết bị Macbook Air M1.

![Vehicle Detection](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253A9f00182c-e81e-43d9-8414-02cc976167cf%253Aimage.png%3Ftable%3Dblock%26id%3D35fda7f3-5722-805f-9810-d5d7c9c07b66%26cache%3Dv2&w=3840&q=75)

## 2. Project Objectives
* Nghiên cứu và ứng dụng YOLOv8n: Sử dụng phiên bản Nano nhẹ nhất, tối ưu cho thiết bị tài nguyên hạn chế nhưng đảm bảo xử lý thời gian thực.
* So sánh hiệu năng: Thực nghiệm đối chứng giữa Deep Learning (YOLOv8n) và học máy truyền thống (HOG + SVM) để làm rõ sự tiến hóa của công nghệ.
* Hướng tới giao thông thông minh: Cung cấp cơ sở cho việc đếm lưu lượng xe, hỗ trợ điều khiển tín hiệu đèn và giám sát vi phạm.

## 3. Đánh giá kết quả thực nghiệm

### 3.1. Phân tích Learning Curves (50 Epochs)

#### a. Box Loss Learning Curve (Tổn thất định vị)
Biểu đồ thể hiện sự biến thiên của hàm mất mát về vị trí khung bao trên tập huấn luyện và tập kiểm thử.

![Box Loss Chart](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253A3b52cee4-c170-4eb7-b261-cb18043af50f%253Aimage.png%3Ftable%3Dblock%26id%3D35fda7f3-5722-8096-a1ce-f8dfd3983ba5%26cache%3Dv2&w=3840&q=75)

* Nhận xét: Cả Train Loss và Validation Loss đều giảm nhanh trong 10 epochs đầu tiên.
* Giai đoạn 1-40: Đường Validation Loss luôn duy trì ở mức thấp hơn Train Loss do kỹ thuật điều chuẩn (Regularization) áp dụng trong huấn luyện.
* Giai đoạn 41-50: Hai đường bắt đầu giao nhau, cho thấy dấu hiệu chớm bắt đầu của hiện tượng quá khớp (Overfitting).

#### b. Classification Loss Learning Curve (Tổn thất phân loại)
Biểu đồ đo lường độ chính xác trong việc gán nhãn đúng cho đối tượng.

![Classification Loss Chart](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253A4dd499fa-7581-4b63-a20e-b6a5e1cd6998%253Aimage.png%3Ftable%3Dblock%26id%3D35fda7f3-5722-8047-adc3-fc0e788cdc68%26cache%3Dv2&w=3840&q=75)

* Nhận xét: Train Loss giảm mạnh từ mức 3.4 xuống 1.2 trong 5 epochs đầu, cho thấy mô hình học đặc trưng rất nhanh.
* Trạng thái: Cả hai chỉ số hội tụ về mức thấp quanh ngưỡng 0.6-0.7 tại epoch 50.
* Kết luận: Validation Loss thấp hơn Train Loss và không có dấu hiệu tăng ngược trở lại chứng tỏ mô hình không bị Overfitting ở mảng phân loại.

### 3.2. Confusion Matrix (Ma trận nhầm lẫn)

![Confusion Matrix](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253Ad9847c48-edcb-4baa-9b54-f79b7efe5253%253Aimage.png%3Ftable%3Dblock%26id%3D35fda7f3-5722-8015-9082-f9adb70f511e%26cache%3Dv2&w=3840&q=75)

* Độ chính xác cao: Xe container (100%), xe cứu hỏa (98%), xe hơi (94%) và xe máy (93%).
* Hạn chế: Xe đạp đạt độ chính xác thấp nhất (59%) do tiết diện nhỏ dễ bị lẫn vào nền.
* Sai số: 17% mẫu xe Van bị nhận nhầm thành xe hơi do tương đồng về đặc điểm hình học.
* Dương tính giả: Tại các vùng nền trống, mô hình có xu hướng nhạy cảm quá mức dẫn đến nhận nhầm xe máy (67%) và xe hơi (25%).

## 4. Performance Metrics

| Metric | Giá trị | Mô tả |
| :--- | :---: | :--- |
| mAP@0.5 | 0.87 | Khả năng phát hiện và khoanh vùng đối tượng tốt. |
| Recall | 0.86 | Tìm được 86% số lượng đối tượng thực tế, hạn chế bỏ sót. |
| Precision | 0.78 | Tỷ lệ dự đoán đúng trên tổng dự đoán (vẫn còn dương tính giả). |
| F1-Score | 0.82 | Sự cân bằng giữa Precision và Recall. |

## 5. Tổng Kết & Quan Sát
* Sự hội tụ: Các đường cong Loss giảm đều và tiệm cận mức thấp nhất, quá trình huấn luyện ổn định.
* Hiệu suất vượt trội: YOLOv8n đạt tốc độ ~35 FPS, cao gấp 6 lần so với HOG+SVM (~5.78 FPS) trên cùng phần cứng M1.
* Kết luận: HOG+SVM phù hợp cho mục đích học tập hoặc nền tĩnh, trong khi YOLOv8n là lựa chọn tối ưu cho bài toán thực tế yêu cầu thời gian thực.

---
*Created by Pham Long Chuc*
