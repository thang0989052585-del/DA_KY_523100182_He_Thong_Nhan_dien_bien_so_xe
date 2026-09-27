# Firmware Cảnh Báo STM32F407VET6 (IoT Alert Node)

## 📌 Tổng Quan
Thư mục này chứa mã nguồn firmware nạp cho bo mạch vi điều khiển **STM32F407VET6**, đóng vai trò là cơ cấu chấp hành phần cứng cảnh báo nhận diện biển số xe trong hệ thống AI & IoT.

---

## ⚙️ Cấu Hình Phần Cứng & Giao Tiếp

* **Vi điều khiển:** STM32F407VET6 (Cortex-M4, xung nhịp 168 MHz).
* **Chuẩn giao tiếp nối tiếp:** USART2 (PA2 - TX, PA3 - RX) kết nối với module Wi-Fi ESP-01 hoặc cáp USB-UART máy tính.
* **Tốc độ truyền (Baudrate):** `115200 bps`, 8 data bits, no parity, 1 stop bit (`8N1`).
* **Đèn LED cảnh báo D2:** Chân **PA6** (Mạch kích mức Thấp - Active LOW):
  - `GPIO_PIN_RESET` (Mức 0V): **BẬT SÁNG ĐÈN LED**.
  - `GPIO_PIN_SET` (Mức 3.3V): **TẮT ĐÈN LED**.

---


> 💡 **Lưu ý phần cứng:** Hệ thống sử dụng duy nhất đèn LED D2 (PA6) làm cơ cấu chỉ thị cảnh báo trực quan sáng 10 giây (không kết nối còi Buzzer).
## 🎯 Cơ Chế Hoạt Động (LED Sáng 10 Giây)

1. **Khởi động nguồn:**
   - Hệ thống chớp nhanh LED D2 3 lần (mỗi lần 100ms) để thông báo vi điều khiển đã nạp code thành công và bắt đầu chạy.
   - Sau đó chốt chân PA6 ở mức SET để tắt đèn, mở ngắt nhận UART `HAL_UART_Receive_IT`.

2. **Khi nhận diện được xe mục tiêu:**
   - Web AI phát hiện biển số mục tiêu -> gửi bản tin HTTP POST tới ESP-01 -> ESP-01 đẩy chuỗi lệnh qua UART: `ALERT:<plate>\n` tới STM32.
   - Ngắt `HAL_UART_RxCpltCallback` của STM32 lập tức bắt tín hiệu.
   - Hàm `LED_On_Alert(LED_ON_DURATION)` kích hoạt:
     - Kéo chân PA6 xuống mức `GPIO_PIN_RESET` để **bật sáng đèn LED D2 liên tục trong 10 giây (10.000 ms)**.
     - Sau đúng 10 giây, kéo chân PA6 lên mức `GPIO_PIN_SET` để tắt đèn hoàn toàn, quay về trạng thái sẵn sàng cho lượt xe tiếp theo.

---

## 📁 Cấu Trúc File
* `main.c`: File mã nguồn chính chứa cấu hình Clock, USART2, GPIO PA6 và hàm điều khiển LED sáng 10s `LED_On_Alert()`.
* `main.h`: File header định nghĩa thư viện HAL và các prototype hàm.
* `canhbaonhandien.ioc`: File dự án cấu hình STM32CubeMX.
