# 🚗 Hệ Thống Nhận Diện Biển Số Xe Việt Nam (ALPR / ANPR)

> **Đồ án môn học:** Nghiên cứu & Xây dựng Hệ thống Tự Động Nhận Diện Biển Số Xe (Automatic License Plate Recognition)  
> **Công nghệ lõi:** `YOLOv8` ⚡ `CRNN (CNN + BiLSTM)` ⚡ `CTC Loss` ⚡ `PyTorch` ⚡ `OpenCV` ⚡ `Streamlit`

---

>  **Link Google Drive tải sản phẩm đóng gói:** (https://drive.google.com/file/d/1X1n2R2iVez5y6WRnL59zJwGNxEA87hab/view?usp=sharing)


## 📌 1. Giới Thiệu & Mục Tiêu Đề Tài

Hệ thống giải quyết bài toán cốt lõi trong **Giao thông thông minh (ITS)** và **Quản lý bãi đỗ xe tự động**: Tự động phát hiện vị trí và đọc chính xác chuỗi ký tự trên biển số xe phương tiện giao thông tại Việt Nam (ô tô, xe máy, biển 1 dòng và biển 2 dòng) trong điều kiện thời gian thực.

Hệ thống được thiết kế theo kiến trúc **End-to-End** gồm 2 mô-đun AI chuyên biệt:
1. **Mô-đun 1 — Phát Hiện Biển Số (Plate Detection):** Sử dụng mạng **YOLOv8** (hoặc giải thuật OpenCV Fallback) để khoanh vùng và cắt chính xác tọa độ biển số, loại bỏ hoàn toàn nhiễu từ thân xe và môi trường xung quanh.
2. **Mô-đun 2 — Nhận Dạng Ký Tự (OCR Recognition):** Ứng dụng mạng **CRNN (Convolutional Recurrent Neural Network)** kết hợp **CTC Loss**, đọc trực tiếp toàn bộ chuỗi ký tự mà không cần bước phân đoạn ký tự (character segmentation) phức tạp.

---

## 🏗️ 2. Kiến Trúc Hệ Thống (System Architecture)

### 🤖 Pipeline AI Nhận Diện Biển Số

```mermaid
graph TD
    A["Dau vao: Upload / Camera / URL"] --> B["Mo-dun 1: YOLOv8 Plate Detector"]
    B --> C["Cat vung bien so - Crop"]
    C --> D["Tien xu ly: Grayscale + CLAHE + Resize 192x64"]
    D --> E["Mo-dun 2: CRNN Feature Extractor CNN"]
    E --> F["Chui dac trung tuan tu: BiLSTM"]
    F --> G["Giai ma CTC Greedy Decoder"]
    G --> H["Kiem tra hop le: Format Bien So VN"]
    H --> I["Giao dien Web Streamlit Dashboard"]
    I --> J{"Khop bien so muc tieu?"}
    J -- "Co" --> K["Kich hoat canh bao IoT"]
    J -- "Khong" --> L["Hien thi thong tin xe binh thuong"]
    K --> M["iot_client.py: HTTP POST den ESP-01"]
    M --> N["ESP-01 Wi-Fi Bridge: UART den STM32"]
    N --> O["STM32F407VET6: LED D2 sang 10 giay"]
```

### 🔄 Luồng Tích Hợp IoT Đầy Đủ (End-to-End Flow)

```mermaid
sequenceDiagram
    autonumber
    actor User as Phuong tien vao tram
    participant Cam as Camera / Web UI
    participant AI as AI YOLOv8 + CRNN
    participant IoT as iot_client.py
    participant ESP as ESP-01 Wi-Fi Bridge
    participant STM as STM32F407VET6
    participant LED as LED D2 PA6

    User->>Cam: Phuong tien di chuyen vao tram
    Cam->>AI: Chup va truyen anh ve Web
    AI->>AI: YOLOv8 phat hien + cat vung bien so
    AI->>AI: CRNN nhan dang chuoi ky tu
    AI->>AI: Doi chieu voi danh sach bien so muc tieu
    alt Khop bien so muc tieu
        AI->>IoT: Goi send_alert(plate_text)
        IoT->>ESP: HTTP POST /alert data plate=29AB-12345
        ESP->>ESP: Parse HTTP request
        ESP->>STM: Gui UART "ALERT:29AB-12345" 115200 bps
        STM->>STM: Ngat USART2 bat chuoi ky tu
        STM->>LED: Kich hoat nhap nhay LED D2 PA6 x5 lan
        STM-->>ESP: Phan hoi "STM32_ACK:29AB-12345"
        ESP-->>IoT: HTTP 200 OK
        IoT-->>AI: Xac nhan da kich hoat canh bao
        AI->>Cam: Hien thi canh bao mau do tren Dashboard
    else Bien so binh thuong
        AI->>Cam: Hien thi thong tin xe, khong bat canh bao
    end
```

### 🔧 Chi Tiết Các Tầng Trong Mạng CRNN

| Tầng | Thành phần | Mô tả |
| :--- | :--- | :--- |
| **CNN** | 7 tầng Conv2D + BatchNorm + ReLU + MaxPool | Trích xuất đặc trưng hình ảnh biển số |
| **Map-to-Seq** | Reshape feature maps thành chuỗi vector | Chuyển đổi sang dạng chuỗi thời gian |
| **BiLSTM** | 2 tầng Bidirectional LSTM (hidden=256) | Học ngữ cảnh 2 chiều trái-phải |
| **CTC Decode** | Linear + CTC Greedy Decode | Giải mã ký tự, loại blank `_` và ký tự trùng |

## 🌟 3. Các Điểm Nổi Bật & Cải Tiến Kỹ Thuật

* **Xử lý mất cân bằng dữ liệu (Class Imbalance):** Tự động áp dụng kỹ thuật **Oversampling** cho các ký tự hiếm hoặc dễ nhầm lẫn (đặc biệt là ký tự `B` so với số `8`), triệt tiêu hiện tượng mô hình thiên vị đoán sang số `8`.
* **Phân chia tập dữ liệu chuẩn 3 phần (70/15/15):** Chia rõ ràng **Train (70%) / Validation (15%) / Test (15%)** độc lập với seed cố định. Tuyệt đối không đánh giá trên tập dữ liệu đã dùng để huấn luyện.
* **Cơ chế sao lưu và so sánh an toàn:** Tự động backup mô hình cũ thành `saved_models/plate_recognizer_backup.pth`. Sau khi huấn luyện, hệ thống tự động so sánh mô hình mới và mô hình cũ trên cùng **Test Set** và chỉ ghi đè mô hình production khi kết quả mới tốt hơn.
* **Xử lý thông minh biển 1 dòng & 2 dòng:** Tự động nhận diện biển vuông (xe máy, xe tải) và biển dài (ô tô), kết hợp song song chiến lược đọc nguyên khối và tách dòng dựa trên phân tích khoảng trống mật độ điểm ảnh.
* **Hậu xử lý chuẩn hóa format VN:** Xác thực định dạng biển số theo chuẩn vị trí mã tỉnh, seri chữ cái và số thứ tự mà không tự ý hoán đổi ký tự khi chưa đủ bằng chứng.

---

## 📊 4. Kết Quả Đánh Giá Thực Nghiệm (Trên Test Set Độc Lập)

Đánh giá toàn diện trên **945 ảnh tập Test độc lập (15% dataset)**:

| Chỉ số đánh giá | Kết quả đạt được | Ý nghĩa thực tế |
| :--- | :---: | :--- |
| **Character Accuracy** | **88.33%** | Tỷ lệ ký tự nhận diện đúng vị trí |
| **Exact Plate Accuracy** | **65.82%** (622/945) | Tỷ lệ biển số đúng hoàn toàn 100% |
| **Character Error Rate (CER)** | **9.69%** | Tỷ lệ khoảng cách chỉnh sửa Levenshtein |
| **Validation Exact Accuracy** | **66.14%** (625/945) | Độ chính xác trên tập kiểm định |
| **Tốc độ suy luận (Inference)** | **~15 - 30 ms / ảnh** | Đáp ứng yêu cầu thời gian thực trên CPU/GPU |

---

## 📁 5. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
DOAN/
├── app.py                       # Ứng dụng Web Dashboard hoàn chỉnh bằng Streamlit
├── config.py                    # Cấu hình trung tâm (Hyperparameters, Alphabet, Paths)
├── dataset.py                   # Pipeline nạp dữ liệu, chia 70/15/15, Augmentation, Oversample
├── detector.py                  # Thuật toán phát hiện biển số (YOLOv8 & OpenCV)
├── model.py                     # Kiến trúc CRNN PyTorch (CNN + BiLSTM + CTC Decode)
├── train.py                     # Huấn luyện OCR, Early Stopping, Backup & So sánh Test Set
├── train_yolo.py                # Kịch bản huấn luyện YOLOv8 phát hiện biển số
├── evaluate.py                  # Đánh giá độc lập, vẽ biểu đồ lỗi, xuất báo cáo chi tiết
├── iot_client.py                # Module giao tiếp IoT (HTTP → ESP-01 → UART → STM32)
├── launcher.py                  # Entry point đóng gói PyInstaller → BienSoAI.exe
├── BienSoAI.spec                # Cấu hình PyInstaller build (onedir mode)
├── plate_history.json           # Lịch sử nhận dạng biển số (lưu tự động khi chạy)
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── .gitignore                   # Loại trừ cache, build, dist, model backup
├── README.md                    # Tài liệu dự án (file này)
├── PROJECT_INFO_FOR_REPORT.txt  # Thông tin chi tiết dùng cho báo cáo
├── saved_models/                # Thư mục lưu trữ trọng số mô hình
│   ├── plate_recognizer.pth         # Model OCR CRNN chính thức (Production)
│   ├── plate_recognizer_backup.pth  # Model OCR sao lưu dự phòng (Backup)
│   └── yolo_plate_detector.pt       # Model YOLOv8 phát hiện biển số
├── esp01_firmware/              # Firmware Arduino/ESP8266 cho module Wi-Fi ESP-01
│   └── esp01_server/                # Sketch Arduino: HTTP server + UART bridge → STM32
├── results/                     # Kết quả đánh giá và biểu đồ xuất ra
│   ├── evaluation_report.txt        # Báo cáo đánh giá chi tiết
│   ├── char_error_distribution.png  # Biểu đồ phân phối ký tự bị nhầm
│   └── training_history.png         # Biểu đồ quá trình huấn luyện
├── dataset/                     # Thư mục dữ liệu (không gồm yolo_data - ~961MB)
│   ├── labels.csv                   # File nhãn (filename, plate_text)
│   └── plates/                      # Thư mục chứa ảnh biển số (~6300+ ảnh)
└── dist/                        # Thư mục đóng gói phân phối (tạo bởi PyInstaller)
    └── BienSoAI/                    # Ứng dụng standalone đã đóng gói
        ├── BienSoAI.exe                 # File thực thi chính (double-click để chạy)
        └── _internal/                   # Thư mục tài nguyên & môi trường Python nhúng
```

## ⚙️ 6. Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### Bước 1: Cài đặt môi trường
Yêu cầu Python 3.9 trở lên. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện mô hình OCR (Tùy chọn nếu muốn train lại)
Chương trình sẽ tự động chia 3 tập (Train 70% / Val 15% / Test 15%), oversample ký tự `B`, backup model cũ và so sánh trên Test set:
```bash
python train.py
```

### Bước 3: Đánh giá mô hình & Xem thống kê lỗi
Chạy đánh giá toàn diện trên tập Test độc lập:
```bash
# Đánh giá toàn bộ tập Val & Test
python evaluate.py

# Hoặc test riêng một ảnh biển số cụ thể
python evaluate.py --image "path/to/plate_image.jpg"
```

### Bước 4: 🚀 Khởi chạy Giao Diện Web Demo (Streamlit)
Khởi động giao diện Web tương tác trực quan:
```bash
streamlit run app.py
```
Mở trình duyệt truy cập: **`http://localhost:8501`** (hoặc port được hiển thị trên Terminal).

---

## 🖥️ 7. Tính Năng Giao Diện Web App
- 📷 **Nhận diện từ Ảnh (Image Recognition):**
  - 📁 **Nhiều phương thức nhập:** Tải ảnh từ máy tính, chụp trực tiếp bằng Camera, hoặc dán link URL ảnh.
  - 🎯 **Theo dõi biển số mục tiêu (Target Plate Tracking & Alert):** Cho phép nhập biển số xe mục tiêu cần theo dõi; hệ thống sẽ tự động đối chiếu, bật cảnh báo trực quan nổi bật và gắn nhãn highlight biển số trùng khớp.
  - 👁️ **Chế độ phát hiện kép:** Tự động phát hiện bằng **YOLOv8** hoặc thuật toán **OpenCV Canny Contour**.
  - 🔍 **Chi tiết nhận diện:** Hiển thị tọa độ Bounding Box, ảnh crop biển số, chuỗi ký tự đã nhận dạng, độ tin cậy (%) và thời gian xử lý (ms).
- 🎨 **Giao diện Modern Dark Tech Theme:** Thiết kế thân thiện, trực quan, bố cục chia 2 cột rõ ràng giữa đầu vào và kết quả phân tích.

---

## 🔌 8. Kiến Trúc Phần Cứng & Hệ Thống IoT Cảnh Báo (Hardware & IoT Integration)

### ❓ Trả Lời: 2 Thiết Bị Đã Đủ Chưa?

| Kịch Bản Ứng Dụng | Thiết Bị Cần Có | Trạng Thái | Mô Tả |
| :--- | :--- | :---: | :--- |
| **Kịch Bản 1: Test Trực Tiếp (Có dây Serial)** | 1. Laptop (Web AI)<br>2. Board STM32F407VET6<br>3. Mạch USB-UART (hoặc ST-Link VCP) | **✅ ĐÃ ĐỦ** | Cắm trực tiếp USB-UART từ máy tính vào chân PA2/PA3 của STM32. Khi AI phát hiện biển số mục tiêu, gửi lệnh `ALERT:<plate>` qua cổng COM. LED D2 (PA6) chớp ngay lập tức. |
| **Kịch Bản 2: Hệ Thống IoT Chuẩn (Không dây Wi-Fi)** | 1. Laptop (Web AI)<br>2. Board STM32F407VET6<br>3. Module Wi-Fi ESP-01 (ESP8266) | **⏳ CẦN THÊM ESP-01** | Vì chip STM32F407 không tích hợp Wi-Fi, cần thêm ESP-01 làm cầu nối (Web gửi HTTP POST qua Wi-Fi -> ESP-01 nhận và đẩy UART sang STM32). |

---

### 📊 Sơ Đồ Hoạt Động Tổng Thể (System Architecture Flow)

#### 🔹 Kịch Bản 1: Kết Nối Trực Tiếp Qua USB-UART (Hiện tại đang test)
```mermaid
flowchart LR
    subgraph Laptop["💻 Máy Tính / Laptop"]
        Cam[Camera / Ảnh Upload] --> AI[Web Streamlit: YOLOv8 + CRNN]
        AI --> Match{Trùng Biển<br/>Mục Tiêu?}
        Match -- Có --> Serial[Cổng COM USB-UART<br/>Baud: 115200]
    end

    subgraph Hardware["🎛️ Phần Cứng STM32"]
        Serial -->|Dây TX -> PA3| STM[STM32F407VET6<br/>Ngắt USART2 RX]
        STM -->|Kích hoạt GPIO| LED[🚨 LED D2 - PA6<br/>Nhấp nháy 5 lần / Còi Buzzer]
        STM -.->|Gửi phản hồi ACK: PA2 -> RX| Serial
    end
```

#### 🔹 Kịch Bản 2: Hệ Thống Không Dây Hoàn Chỉnh Qua Wi-Fi (ESP-01 + STM32)
```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 Người Dùng / Xe Đến
    participant Cam as 📷 Camera / Web UI
    participant AI as 🧠 AI (YOLOv8 + CRNN)
    participant ESP as 📶 ESP-01 (Wi-Fi Bridge)
    participant STM as ⚡ STM32F407VET6
    participant Actuator as 🚨 LED D2 / Còi Báo Động

    User->>Cam: Phương tiện di chuyển vào trạm
    Cam->>AI: Chụp và truyền ảnh về Web
    AI->>AI: Phát hiện biển số + Nhận dạng ký tự
    AI->>AI: Đối chiếu với danh sách Biển số mục tiêu (Blacklist/Whitelist)
    alt Khớp biển số mục tiêu
        AI->>ESP: HTTP POST /alert (data: plate=30A12345)
        ESP->>ESP: Parse dữ liệu HTTP
        ESP->>STM: Gửi chuỗi UART: "ALERT:30A12345\n" (115200 bps)
        STM->>STM: Ngắt USART2 bắt chuỗi ký tự
        STM->>Actuator: Kích hoạt chớp nháy LED D2 (PA6) 5 lần / Bật còi
        STM-->>ESP: Gửi phản hồi "STM32_ACK:30A12345\n"
        ESP-->>AI: Phản hồi HTTP 200 OK (Đã kích hoạt cảnh báo)
        AI->>Cam: Hiển thị cảnh báo đỏ trên Dashboard
    else Biển số bình thường
        AI->>Cam: Hiển thị thông tin xe, không bật cảnh báo
    end
```

---

### 📌 Sơ Đồ Đấu Nối Chân (Pinout & Wiring Diagram)

#### 1. Đấu Nối Test Trực Tiếp (USB-UART ↔ STM32F407VET6):
```text
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │      USB-UART (COMx)    │             │     STM32F407VET6       │
    │                         │             │                         │
    │                 TX (Dây)├────────────►│ PA3 (USART2_RX)         │
    │                 RX (Dây)│◄────────────┤ PA2 (USART2_TX)         │
    │                GND (Dây)├────────────►│ GND                     │
    │            5V / 3V3 (Dây)├────────────►│ 5V / 3V3 (Nếu cấp nguồn)│
    └─────────────────────────┘             └─────────────────────────┘
                                                         │
                                            ┌────────────┴────────────┐
                                            │ D2 (LED Xanh) -> PA6    │
                                            │ D3 (LED Đỏ)   -> PA7    │
                                            └─────────────────────────┘
```
> ⚠️ **Quy tắc bắt buộc:** **TX của mạch này phải nối vào RX của mạch kia** và **phải nối chung GND**!

#### 2. Đấu Nối Hệ Thống Wi-Fi IoT (ESP-01 ↔ STM32F407VET6):
```text
ESP-01 (ESP8266)                  STM32F407VET6
┌──────────────┐                 ┌──────────────┐
│ TXD          │────────────────►│ PA3 (USART2_RX)
│ RXD          │◄────────────────│ PA2 (USART2_TX)
│ GND          │─────────────────│ GND          │
│ VCC (3.3V)   │─────────────────│ 3.3V         │
│ CH_PD (EN)   │─────────────────│ 3.3V (Kéo lên mức cao để bật chip)
└──────────────┘                 └──────────────┘
```

---

### 💬 Giao Thức Lệnh Truyền Nhận (UART Protocol Specification)

* **Tốc độ truyền (Baudrate):** `115200 bps`, 8 data bits, no parity, 1 stop bit (`8N1`).
* **Ký tự kết thúc dòng (Line Terminator):** Hỗ trợ cả `\r\n` (CRLF), `\n` (LF) và `\r` (CR từ PuTTY).

| Lệnh gửi (Từ PC/ESP-01 sang STM32) | Phản hồi từ STM32 | Hành động phần cứng |
| :--- | :--- | :--- |
| `ALERT:TEST\r\n` | `STM32_ACK:TEST\r\n` | LED D2 (PA6) nhấp nháy 5 lần chu kỳ 150ms |
| `ALERT:30A12345\r\n` | `STM32_ACK:30A12345\r\n` | LED D2 (PA6) nhấp nháy 5 lần, báo còi |
| Khởi động nguồn STM32 | `STM32_READY\r\n` | Báo hiệu vi điều khiển đã sẵn sàng nhận lệnh |

---
*Đồ án phục vụ nghiên cứu và phát triển hệ thống Trí tuệ nhân tạo nhận diện biển số xe.*
