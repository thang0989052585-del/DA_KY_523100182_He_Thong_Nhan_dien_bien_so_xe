"""
iot_client.py — Module Giao Tiếp IoT Với ESP-01 & STM32
=========================================================
Đề tài: Nghiên cứu và xây dựng chương trình nhận diện biển số xe.

Nhiệm vụ:
  1. Giao tiếp qua giao thức HTTP (REST API) với vi điều khiển ESP-01 (ESP8266).
  2. Kiểm tra trạng thái kết nối mạng của module ESP-01.
  3. Gửi tín hiệu cảnh báo (HTTP POST /alert) kèm chuỗi biển số xe mục tiêu.
  4. ESP-01 sẽ nhận bản tin và chuyển tiếp qua UART tới STM32F407VET6 để điều khiển đèn LED D2 sáng 10s (không còi).

Đặc tính an toàn:
  - Tất cả request đều có timeout ngắn (2.0s) để tránh treo giao diện Streamlit.
  - Xử lý toàn diện mọi ngoại lệ mạng (Timeout, ConnectionRefused, DNS, v.v.).
  - Tuyệt đối không làm crash Web UI khi ESP-01 chưa bật hoặc mất kết nối.
"""

import sys
import time
import re
import requests
from typing import Tuple, Optional, Dict, Any, List
import config


def _log(msg: str) -> None:
    """
    In log ra stdout một cách an toàn.
    Tránh ValueError: I/O operation on closed file
    xảy ra khi Streamlit redirect/đóng sys.stdout.
    """
    try:
        print(msg, flush=True)
    except (ValueError, OSError):
        # stdout bị đóng bởi Streamlit — bỏ qua, không crash
        try:
            sys.__stdout__.write(msg + "\n")
            sys.__stdout__.flush()
        except Exception:
            pass


def _normalize_plate_text(text: str) -> str:
    """Chuẩn hóa chuỗi biển số: Viết hoa, loại bỏ khoảng trắng, dấu chấm, gạch ngang."""
    if not text:
        return ""
    return re.sub(r"[\s\-\.]", "", text.upper().strip())


def build_esp_base_url(ip: Optional[str] = None, port: Optional[int] = None) -> Optional[str]:
    """
    Xây dựng base URL hợp lệ cho ESP-01.
    
    Args:
        ip: Địa chỉ IP hoặc hostname (ví dụ: "192.168.1.50" hoặc "esp01.local")
        port: Cổng kết nối (mặc định 80)
    
    Returns:
        Base URL dạng "http://192.168.1.50:80" hoặc None nếu IP không hợp lệ.
    """
    raw_ip = ip.strip() if (ip and isinstance(ip, str) and ip.strip()) else getattr(config, "ESP_DEFAULT_IP", "192.168.137.61")
    target_ip = (raw_ip or "192.168.137.61").strip()
    if not target_ip:
        return None

    # Loại bỏ tiền tố http:// hoặc https:// nếu người dùng vô tình nhập vào
    target_ip = re.sub(r"^https?://", "", target_ip).rstrip("/")

    # Xử lý nếu người dùng nhập kèm port trong chuỗi IP (ví dụ "192.168.1.50:8080")
    if ":" in target_ip:
        parts = target_ip.split(":", 1)
        target_ip = parts[0]
        try:
            port = int(parts[1])
        except ValueError:
            port = port or getattr(config, "ESP_DEFAULT_PORT", 80)
    else:
        port = port or getattr(config, "ESP_DEFAULT_PORT", 80)

    if port == 80:
        return f"http://{target_ip}"
    return f"http://{target_ip}:{port}"


def check_esp_connection(
    ip: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[float] = None
) -> Tuple[bool, str, float]:
    """
    Kiểm tra trạng thái kết nối tới module ESP-01 qua HTTP GET.

    Args:
        ip: Địa chỉ IP của ESP-01.
        port: Cổng WebServer của ESP-01.
        timeout: Thời gian timeout tính bằng giây (mặc định từ config.ESP_TIMEOUT).

    Returns:
        (is_connected, message, latency_ms)
        - is_connected (bool): True nếu ESP-01 phản hồi HTTP 200..299.
        - message (str): Thông điệp chi tiết mô tả trạng thái.
        - latency_ms (float): Thời gian phản hồi tính bằng mili-giây.
    """
    base_url = build_esp_base_url(ip, port)
    if not base_url:
        return False, "⚠️ ESP-01 chưa được cấu hình địa chỉ IP", 0.0

    req_timeout = timeout if timeout is not None else getattr(config, "ESP_TIMEOUT", 2.0)
    status_path = getattr(config, "ESP_STATUS_PATH", "/status")
    url = f"{base_url}{status_path}"

    t0 = time.time()
    try:
        # Thử GET endpoint /status trước, nếu 404 thì thử root /
        response = requests.get(url, timeout=req_timeout)
        latency_ms = (time.time() - t0) * 1000

        if response.status_code == 404:
            # Fallback về root endpoint '/'
            url_root = f"{base_url}/"
            response = requests.get(url_root, timeout=req_timeout)
            latency_ms = (time.time() - t0) * 1000

        if 200 <= response.status_code < 300:
            msg = f"🟢 ESP-01 đã kết nối thành công (HTTP {response.status_code}, {latency_ms:.1f}ms)"
            return True, msg, latency_ms
        else:
            msg = f"⚠️ ESP-01 phản hồi mã lỗi HTTP {response.status_code}"
            return False, msg, latency_ms

    except requests.exceptions.Timeout:
        latency_ms = (time.time() - t0) * 1000
        msg = f"🔴 Quá thời gian chờ phản hồi từ ESP-01 ({req_timeout}s)"
        return False, msg, latency_ms

    except requests.exceptions.ConnectionError as ce:
        latency_ms = (time.time() - t0) * 1000
        msg = f"🔴 Không thể kết nối tới ESP-01 tại {base_url} (Thiết bị chưa bật hoặc sai IP/mạng Wi-Fi)"
        return False, msg, latency_ms

    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        msg = f"🔴 Lỗi kết nối ESP-01: {str(e)}"
        return False, msg, latency_ms


def send_alert_to_esp(
    plate: str,
    ip: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[float] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Gửi bản tin cảnh báo phát hiện biển số mục tiêu tới ESP-01 qua HTTP POST /alert.

    Format gửi:
        POST http://<ESP_IP>:<PORT>/alert
        Content-Type: application/x-www-form-urlencoded
        Body: plate=<biển số>

    Args:
        plate: Chuỗi biển số xe phát hiện (ví dụ: "29AB-123.45" hoặc "29AB12345")
        ip: Địa chỉ IP của ESP-01 (tùy chọn)
        port: Cổng kết nối (tùy chọn)
        timeout: Thời gian timeout tính bằng giây

    Returns:
        (success, status_message, raw_response_text)
    """
    clean_plate = _normalize_plate_text(plate)
    if not clean_plate:
        return False, "⚠️ Biển số trống, không thể gửi cảnh báo", None

    base_url = build_esp_base_url(ip, port)
    if not base_url:
        _log(f"[IoT] ESP_IP chưa được cấu hình. Bỏ qua gửi cảnh báo cho biển số {clean_plate}.")
        return False, "⚠️ ESP-01 chưa được cấu hình địa chỉ IP", None

    alert_path = getattr(config, "ESP_ALERT_PATH", "/alert")
    url = f"{base_url}{alert_path}"
    req_timeout = timeout if timeout is not None else getattr(config, "ESP_TIMEOUT", 2.0)

    # Ghi log bắt đầu nhận diện mục tiêu & gửi
    _log(f"[IoT] Target plate detected: {clean_plate}")
    _log(f"[IoT] Sending alert to ESP at {url}...")

    payload = {"plate": clean_plate}

    try:
        # Gửi POST request với form data
        response = requests.post(
            url,
            data=payload,
            headers={"User-Agent": "BienSoAI-WebClient/2.4"},
            timeout=req_timeout
        )

        resp_body = response.text.strip()
        _log(f"[IoT] ESP response: HTTP {response.status_code} - {resp_body}")

        if 200 <= response.status_code < 300:
            return True, f"✅ Đã gửi cảnh báo thành công tới ESP-01 (HTTP {response.status_code})", resp_body
        else:
            return False, f"⚠️ ESP-01 trả về mã lỗi HTTP {response.status_code}: {resp_body}", resp_body

    except requests.exceptions.Timeout:
        err_msg = f"Quá thời gian chờ phản hồi ({req_timeout}s)"
        _log(f"[IoT] ESP connection failed: {err_msg}")
        return False, f"⚠️ Timeout: ESP-01 không phản hồi trong {req_timeout}s", None

    except requests.exceptions.ConnectionError as ce:
        err_msg = f"Không thể kết nối tới {base_url} (ESP-01 chưa bật hoặc khác mạng Wi-Fi)"
        _log(f"[IoT] ESP connection failed: {err_msg}")
        return False, f"⚠️ Không thể gửi cảnh báo: ESP-01 chưa kết nối", None

    except Exception as e:
        err_msg = str(e)
        _log(f"[IoT] ESP connection failed: {err_msg}")
        return False, f"⚠️ Lỗi giao tiếp ESP-01: {err_msg}", None


def process_target_alerts_for_detections(
    ocr_results: List[Dict[str, Any]],
    target_plate_query: str,
    esp_ip: Optional[str] = None,
    esp_port: Optional[int] = None
) -> Dict[str, Any]:
    """
    Xử lý kiểm tra và gửi cảnh báo IoT cho tất cả các biển số nhận diện được trong một ảnh.

    Quy trình:
      1. Kiểm tra nếu có target_plate_query được đặt.
      2. Lọc ra các biển số trùng với mục tiêu (đã normalize).
      3. Loại bỏ trùng lặp nếu cùng một biển số bị detect nhiều lần trong ảnh.
      4. Với mỗi biển số trùng: Gửi HTTP POST /alert tới ESP-01.
      5. Tổng hợp kết quả và trả về để hiển thị lên Web UI.

    Args:
        ocr_results: Danh sách kết quả nhận diện từ pipeline ảnh.
        target_plate_query: Chuỗi biển số mục tiêu nhập từ người dùng.
        esp_ip: IP của ESP-01.
        esp_port: Cổng của ESP-01.

    Returns:
        dict chứa:
          - has_target_hit: bool (Có biển số trùng mục tiêu hay không)
          - matched_plates: List[str] (Danh sách các biển số trùng)
          - send_results: List[dict] (Chi tiết kết quả gửi cho từng biển số)
          - summary_message: str (Thông điệp tóm tắt trạng thái IoT)
    """
    norm_target = _normalize_plate_text(target_plate_query)
    if not norm_target or not ocr_results:
        return {
            "has_target_hit": False,
            "matched_plates": [],
            "send_results": [],
            "summary_message": ""
        }

    # Tìm các biển số khớp mục tiêu
    matched_set = set()
    for res in ocr_results:
        if res.get("is_target_hit", False):
            p_text = res.get("plate_text", "")
            norm_p = _normalize_plate_text(p_text)
            if norm_p:
                matched_set.add(p_text)

    if not matched_set:
        return {
            "has_target_hit": False,
            "matched_plates": [],
            "send_results": [],
            "summary_message": ""
        }

    # Gửi cảnh báo cho từng biển số mục tiêu duy nhất
    send_results = []
    for plate in matched_set:
        success, msg, resp = send_alert_to_esp(plate, ip=esp_ip, port=esp_port)
        send_results.append({
            "plate": plate,
            "success": success,
            "message": msg,
            "response": resp
        })

    all_success = all(r["success"] for r in send_results)
    if all_success:
        summary_msg = f"🟢 Đã phát tín hiệu cảnh báo tới ESP-01 & STM32 cho {len(matched_set)} biển số mục tiêu."
    else:
        # Lấy thông báo lỗi của thiết bị
        first_err = next((r["message"] for r in send_results if not r["success"]), "Lỗi kết nối")
        summary_msg = f"{first_err}"

    return {
        "has_target_hit": True,
        "matched_plates": list(matched_set),
        "send_results": send_results,
        "summary_message": summary_msg
    }
