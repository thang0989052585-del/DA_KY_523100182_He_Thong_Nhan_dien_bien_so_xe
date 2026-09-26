/*
  esp01_firmware.ino — Firmware cho ESP-01 (ESP8266) làm cầu nối IoT
  ===================================================================
  Đề tài: Hệ Thống Nhận Diện Biển Số Xe AI + Cảnh Báo IoT
  
  Chức năng:
    1. Kết nối vào mạng Wi-Fi (cùng mạng với Laptop chạy Web Streamlit).
    2. Khởi tạo Web Server ở cổng 80:
       - GET /status : Kiểm tra kết nối từ Web Streamlit
       - POST /alert : Nhận biển số từ Web, gửi lệnh UART sang STM32F4
    3. Gửi lệnh qua UART (Baudrate 115200) sang STM32:
       Format: "ALERT:<plate>\n"  (VD: "ALERT:30A12345\n")
    4. Nhận phản hồi ACK từ STM32: "STM32_ACK:<plate>\n"
  
  Sơ đồ chân nối ESP-01 sang STM32F407VET6:
    ESP-01 TXD  -----> STM32 PA3 (USART2_RX)
    ESP-01 RXD  <----- STM32 PA2 (USART2_TX)
    ESP-01 GND  -----> STM32 GND
    ESP-01 3V3  -----> Nguồn 3.3V (Cần cấp đủ dòng >= 300mA)
    ESP-01 EN   -----> 3.3V (Kéo lên mức cao để bật chip)
*/

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// ──────── CẤU HÌNH WI-FI ────────
// Khớp chính xác với Mobile Hotspot Windows của bạn
const char* WIFI_SSID     = "DUAN_IOT";          // Tên Mobile Hotspot vừa tạo
const char* WIFI_PASSWORD = "123457890";         // Mật khẩu Mobile Hotspot

// Khởi tạo Web Server cổng 80
ESP8266WebServer server(80);

// Xử lý endpoint GET / hoặc GET /status (Kiểm tra kết nối)
void handleStatus() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "text/plain", "ESP-01_CONNECTED_OK");
}

// Xử lý endpoint POST /alert (Nhận biển số và chuyển tiếp sang STM32)
void handleAlert() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  
  if (server.hasArg("plate")) {
    String plate = server.arg("plate");
    plate.trim();
    
    // Gửi lệnh UART sang STM32F407
    Serial.print("ALERT:");
    Serial.print(plate);
    Serial.print("\n");
    
    // Phản hồi về Web Streamlit
    String reply = "OK: " + plate;
    server.send(200, "text/plain", reply);
  } else {
    server.send(400, "text/plain", "Missing 'plate' parameter");
  }
}

// Xử lý khi người dùng truy cập trang không tồn tại
void handleNotFound() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(404, "text/plain", "Endpoint Not Found");
}

void setup() {
  // Cấu hình UART tốc độ 115200 bps (kết nối trực tiếp với STM32 USART2)
  Serial.begin(115200);
  delay(500);
  
  // Kết nối Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  // Chờ kết nối Wi-Fi thành công
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  
  // Thiết lập các route cho Web Server
  server.on("/", HTTP_GET, handleStatus);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/alert", HTTP_POST, handleAlert);
  server.onNotFound(handleNotFound);
  
  // Bật Server
  server.begin();
  
  // Báo hiệu ra UART cho STM32 hoặc Serial Monitor biết IP
  Serial.print("ESP_READY_IP:");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Xử lý các request HTTP đến từ Web Streamlit
  server.handleClient();
  
  // Nếu có dữ liệu phản hồi từ STM32 trả về ESP-01
  if (Serial.available()) {
    String stm_response = Serial.readStringUntil('\n');
    // Có thể dùng để debug hoặc forward trạng thái
  }
}
