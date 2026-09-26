CẤU TRÚC THƯ MỤC DATASET (TẬP DỮ LIỆU)
============================================================

Thư mục này chứa toàn bộ dữ liệu máy học phục vụ huấn luyện 2 mô hình cốt lõi trong đồ án Nhận diện biển số xe:

1. DỮ LIỆU NHẬN DẠNG KÝ TỰ (OCR / CRNN)
---------------------------------------
- Thư mục: plates/
- Tệp nhãn: labels.csv
- Mô tả: Chứa bộ dữ liệu "Vietnamese License Plate OCR" với hơn 6600+ bức ảnh CỰC KỲ DÃ CHIẾN: Các mảng cắt ngang/vuông trực tiếp từ Camera Giao Thông thật (Chứa xe dính bùn, rỉ sét, móp méo, bóng ma, rọi đèn lóa).
- Mục đích: Ép mạng nơ-ron học triệt để sự thay đổi của font dập khuôn siêu dày soái ca đời thực ngoài Việt Nam. Thay vì học các chữ mẫu giáo mỏng dính.

2. DỮ LIỆU PHÁT HIỆN BIỂN SỐ (OBJECT DETECTION / YOLOv8)
---------------------------------------------------------
- Thư mục: yolo_data/
- Cấu trúc con:
    + dataset.yaml : File chỉ mục cấu hình đường dẫn tuyệt đối cho Ultralytics YOLO.
    + images/      : Gồm tập ảnh chụp toàn cảnh xe trên đường thật (đủ biển ngang, vuông, oto, nhòe).
    + labels/      : Tập file txt mang tọa độ yolo object-box chú thích đúng vị trí tấm biển số đo bằng tọa độ ma trận.
- Mô tả: Tập dữ liệu ảnh đường phố thật (Real-world) tải từ nguồn bên ngoài.
- Mục đích: Ép mô hình YOLOv8 Nano học cách phân biệt ra thẻ biển số phương tiện giữa hàng trăm nhiễu như người, đồ vật, bánh xe, bảng hiệu. Giúp hệ thống không còn bị đoán nhầm vào viền ô tô trắng như OpenCV cũ.
- Lệnh huấn luyện: Chạy lệnh `python train_yolo.py` (tại thư mục gốc).
