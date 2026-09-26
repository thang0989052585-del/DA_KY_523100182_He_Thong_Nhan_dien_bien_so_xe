"""
detector.py — Phát Hiện Biển Số Xe (Plate Detector)
======================================================
Module này xử lý Giai đoạn 1 của pipeline:
  ĐẦU VÀO: Ảnh xe nguyên (ví dụ: ảnh chụp xe trên đường)
  ĐẦU RA : Vùng ảnh đã được cắt chứa biển số xe (plate crop)

PHƯƠNG PHÁP:
  Sử dụng xử lý ảnh truyền thống bằng OpenCV theo quy trình:
    1. Chuyển ảnh sang xám (Grayscale)
    2. Làm mịn nhiễu bằng Bilateral Filter (giữ cạnh sắc)
    3. Phát hiện cạnh bằng thuật toán Canny
    4. Tìm đường viền (Contours) từ ảnh cạnh
    5. Lọc đường viền theo tỷ lệ chiều rộng/chiều cao (Aspect Ratio)
       → Giữ lại các vùng có hình chữ nhật nằm ngang như biển số
    6. Cắt và trả về vùng biển số

TẠI SAO KHÔNG DÙNG DL CHO PHÁT HIỆN?
  Với đề tài giảng dạy và nghiên cứu, phương pháp truyền thống OpenCV:
    ✓ Không cần GPU đắt tiền
    ✓ Không cần tập dữ liệu annotation lớn
    ✓ Dễ giải thích, phù hợp báo cáo học thuật
    ✓ Hoạt động ổn với ảnh chất lượng tốt, điều kiện ánh sáng đủ
"""

import cv2
import numpy as np
from PIL import Image

import config


# ──────────────────────────────────────────────────────────────
# LỌC VÙNG BIỂN SỐ BẰNG ASPECT RATIO
# ──────────────────────────────────────────────────────────────

def _is_valid_plate_contour(contour, img_area: int) -> bool:
    """
    Kiểm tra liệu một đường viền có thể là biển số xe không.

    Tiêu chí:
      1. Diện tích vừa phải (không quá nhỏ/quá lớn so với ảnh)
      2. Hình chữ nhật nằm ngang (tỷ lệ W:H từ 1.5 đến 8.0)
    """
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h

    # Kiểm tra diện tích
    area_ratio = area / img_area
    if not (config.MIN_PLATE_AREA_RATIO < area_ratio < config.MAX_PLATE_AREA_RATIO):
        return False

    # Kiểm tra tỷ lệ W:H (Aspect Ratio)
    aspect = w / max(h, 1)
    if not (config.MIN_ASPECT < aspect < config.MAX_ASPECT):
        return False

    return True


def detect_plate_opencv(image_bgr: np.ndarray) -> list:
    """
    Phát hiện và cắt vùng biển số xe từ ảnh BGR bằng OpenCV.

    Args:
        image_bgr: Ảnh đầu vào dạng numpy array BGR (đọc bằng cv2.imread)

    Returns:
        List[dict]: Danh sách các biển số tìm được, mỗi phần tử gồm:
          - "crop"   : numpy array ảnh biển số đã cắt (BGR)
          - "bbox"   : (x, y, w, h) — tọa độ vùng biển số trên ảnh gốc
          - "score"  : float — điểm tin cậy ước tính (0-1)
    """
    img_h, img_w = image_bgr.shape[:2]
    img_area = img_h * img_w

    # ── Bước 1: Tiền xử lý ────────────────────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Bilateral filter: Làm mịn nhưng giữ cạnh (không dùng GaussianBlur vì làm mờ cạnh)
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # ── Bước 2: Cân bằng histogram cục bộ (CLAHE) ─────────────
    # CLAHE (Contrast Limited Adaptive Histogram Equalization):
    # Tăng độ tương phản của từng vùng nhỏ → biển số dễ thấy hơn trong điều kiện sáng/tối không đều
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # ── Bước 3: Phát hiện cạnh Canny ──────────────────────────
    edges = cv2.Canny(enhanced, config.CANNY_LOW, config.CANNY_HIGH)

    # Làm dày cạnh bằng dilation để các đường viền liên thông
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # ── Bước 4: Tìm đường viền ───────────────────────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        if not _is_valid_plate_contour(cnt, img_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Thêm padding nhỏ để không cắt sát cạnh biển số
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        crop = image_bgr[y1:y2, x1:x2]

        # Ước tính điểm tin cậy đơn giản dựa trên aspect ratio
        aspect = w / max(h, 1)
        ideal_car = 4.0   # Biển ô tô dẹt
        ideal_moto = 1.3  # Biển xe máy/ô tô vuông
        score1 = 1.0 - abs(aspect - ideal_car) / ideal_car
        score2 = 1.0 - abs(aspect - ideal_moto) / ideal_moto
        score = max(score1, score2)
        score = max(0.0, min(1.0, score))

        results.append({
            "crop" : crop,
            "bbox" : (x1, y1, x2 - x1, y2 - y1),
            "score": score,
        })

    # Sắp xếp theo score giảm dần
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def detect_plate_yolo(image_bgr: np.ndarray, yolo_model) -> list:
    """
    Phát hiện biển số xe bằng mô hình YOLOv8.
    """
    results = []
    # yolo_model trả về 1 list các kết quả (batch)
    preds = yolo_model(image_bgr, verbose=False)[0]
    
    for box in preds.boxes:
        # Lấy tọa độ
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        # class_id = int(box.cls[0]) # 0: BSD, 1: BSV
        
        # Thêm padding
        pad = 4
        img_h, img_w = image_bgr.shape[:2]
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(img_w, x2 + pad)
        y2_p = min(img_h, y2 + pad)
        
        crop = image_bgr[y1_p:y2_p, x1_p:x2_p]
        
        results.append({
            "crop": crop,
            "bbox": (x1_p, y1_p, x2_p - x1_p, y2_p - y1_p),
            "score": score
        })
        
    return results


def draw_detections(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    Vẽ hộp bao (bounding box) lên ảnh gốc để hiển thị kết quả phát hiện.

    Args:
        image_bgr  : Ảnh gốc BGR
        detections : Danh sách kết quả từ detect_plate_opencv()

    Returns:
        Ảnh BGR đã được vẽ hộp bao
    """
    vis = image_bgr.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        score = det["score"]
        color = (0, 220, 50)  # Xanh lá

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        label = f"Bien so #{i+1}  ({score:.2f})"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x, y - lh - 8), (x + lw + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return vis


# ──────────────────────────────────────────────────────────────
# TIỀN XỬ LÝ BIỂN SỐ ĐỂ ĐƯA VÀO MÔ HÌNH OCR
# ──────────────────────────────────────────────────────────────

def preprocess_plate(plate_crop) -> np.ndarray:
    """
    Tiền xử lý vùng biển số đã cắt để đưa vào mạng OCR.

    Quy trình (v2 - cải tiến):
      1. Chuyển sang xám
      2. CLAHE — tăng tương phản cục bộ (giữ chi tiết nét chữ)
      3. Gaussian blur nhẹ — giảm nhiễu nhưng không mất chi tiết
      4. Resize về kích thước chuẩn (PLATE_HEIGHT × PLATE_WIDTH)
      5. Chuẩn hóa giá trị pixel về [0, 1]

    So với v1 (Adaptive Threshold):
      ✓ Giữ lại gradient nét chữ → phân biệt tốt hơn 5/3, V/K, B/G
      ✓ CLAHE xử lý tốt nền không đều (nắng/bóng)
      ✓ Không mất chi tiết do binarize cứng

    Args:
        plate_crop: numpy array hoặc PIL.Image — ảnh biển số màu hoặc xám

    Returns:
        numpy array (H, W) dtype float32, giá trị trong [0.0, 1.0]
    """
    # Chấp nhận cả PIL Image
    if isinstance(plate_crop, Image.Image):
        plate_crop = np.array(plate_crop.convert("RGB"))
        plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)

    # Chuyển xám
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # CLAHE — tăng tương phản cục bộ (xử lý tốt nền không đều)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Gaussian blur nhẹ — giảm nhiễu mà vẫn giữ nét chữ
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Resize về kích thước chuẩn
    resized = cv2.resize(enhanced, (config.PLATE_WIDTH, config.PLATE_HEIGHT),
                         interpolation=cv2.INTER_LINEAR)

    # Chuẩn hóa [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def preprocess_plate_keep_ratio(plate_crop) -> np.ndarray:
    """
    Tiền xử lý biển số GIỮ NGUYÊN tỷ lệ gốc, padding bên phải.

    Dùng cho trường hợp OCR từng dòng của biển số 2 dòng.
    Tránh kéo dãn ngang làm model "ảo tưởng" ra nhiều ký tự thừa.

    Quy trình (v2):
      1. Chuyển xám → CLAHE → Gaussian blur nhẹ
      2. Resize theo chiều cao chuẩn, giữ tỷ lệ W:H
      3. Pad nền (mean gray) bên phải cho đủ chiều rộng chuẩn
      4. Chuẩn hóa [0, 1]
    """
    # Chấp nhận cả PIL Image
    if isinstance(plate_crop, Image.Image):
        plate_crop = np.array(plate_crop.convert("RGB"))
        plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)

    # Chuyển xám
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # CLAHE + blur (đồng bộ với preprocess_plate)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    target_h = config.PLATE_HEIGHT   # 64
    target_w = config.PLATE_WIDTH    # 192
    h, w = enhanced.shape[:2]

    # Resize giữ tỷ lệ: scale theo chiều cao
    scale = target_h / max(h, 1)
    new_w = min(int(w * scale), target_w)  # Không vượt quá target_w
    new_h = target_h

    resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo canvas với giá trị mean (nền giống vùng biên ảnh gốc)
    pad_val = int(resized.mean())
    canvas = np.full((target_h, target_w), pad_val, dtype=np.uint8)
    canvas[:, :new_w] = resized

    normalized = canvas.astype(np.float32) / 255.0
    return normalized



# ──────────────────────────────────────────────────────────────
# CHẠY THỬ
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Cú pháp: python detector.py <duong_dan_anh>")
        print("Ví dụ  : python detector.py test_car.jpg")
    else:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không đọc được ảnh: {img_path}")
        else:
            dets = detect_plate_opencv(img)
            print(f"Tìm thấy {len(dets)} vùng biển số tiềm năng:")
            for i, d in enumerate(dets):
                x, y, w, h = d["bbox"]
                print(f"  #{i+1}: x={x}, y={y}, w={w}, h={h}, score={d['score']:.3f}")

            vis = draw_detections(img, dets)
            out_path = "detection_result.jpg"
            cv2.imwrite(out_path, vis)
            print(f"\nĐã lưu kết quả phát hiện tại: {out_path}")
