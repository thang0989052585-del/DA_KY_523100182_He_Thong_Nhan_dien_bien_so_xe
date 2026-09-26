"""
dataset.py — Tải & Xử Lý Dữ Liệu Biển Số Xe
===============================================
File này xây dựng Pipeline dữ liệu (Data Pipeline) cho bài toán OCR biển số xe.

CẤU TRÚC DỮ LIỆU YÊU CẦU:
  dataset/
  ├── labels.csv         ← File nhãn (2 cột: filename, plate_text)
  ├── plates/            ← Thư mục chứa ảnh biển số đã cắt sẵn
  │   ├── plate_0001.jpg
  │   ├── plate_0002.jpg
  │   └── ...
  ├── train/             ← Ảnh huấn luyện (tự tạo khi chạy --prepare)
  └── val/               ← Ảnh kiểm định

VÍ DỤ NỘI DUNG FILE labels.csv:
  filename,plate_text
  plate_0001.jpg,51A-12345
  plate_0002.jpg,29B-67890
  plate_0003.jpg,51G-00123

AUGMENTATION CHO BIỂN SỐ:
  Khác với phân loại ảnh thông thường, biển số xe cần xử lý cẩn thận hơn:
  ✓ Lật ngang   → KHÔNG làm (sẽ đổi chiều đọc ký tự)
  ✓ Xoay nhẹ   → CÓ (mô phỏng biển số hơi nghiêng, góc nhỏ ±5°)
  ✓ Độ sáng    → CÓ (biển số dưới nắng / bóng mờ)
  ✓ Nhiễu Gauss → CÓ (mô phỏng ảnh chất lượng thấp)
  ✓ Motion blur → CÓ (xe đang di chuyển, ảnh hơi nhòe)
"""

import os
import sys
import csv
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import config
from detector import preprocess_plate


# ──────────────────────────────────────────────────────────────
# AUGMENTATION CHUYÊN BIỆT CHO BIỂN SỐ XE
# ──────────────────────────────────────────────────────────────

def augment_plate(image_gray: np.ndarray) -> np.ndarray:
    """
    Áp dụng các phép biến đổi ngẫu nhiên (Data Augmentation) 
    lên ảnh xám biển số trong lúc huấn luyện.
    """
    img = (image_gray * 255).astype(np.uint8)

    # 1. Xoay nhẹ (±7°)
    if random.random() < 0.4:
        angle = random.uniform(-7, 7)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Thay đổi độ sáng VÀ contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-40, 40)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 3. Nhiễu Gaussian
    if random.random() < 0.3:
        noise = np.random.randn(*img.shape) * random.uniform(5, 25)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Motion blur
    if random.random() < 0.25:
        ksize = random.choice([3, 5, 7])
        kernel = np.zeros((ksize, ksize))
        if random.random() < 0.7:
            kernel[ksize // 2, :] = 1.0 / ksize
        else:
            kernel[:, ksize // 2] = 1.0 / ksize
        img = cv2.filter2D(img, -1, kernel)

    # 5. Perspective transform
    if random.random() < 0.3:
        h, w = img.shape[:2]
        pts1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
        d = 5
        dx = [random.randint(-d, d) for _ in range(4)]
        dy = [random.randint(-d, d) for _ in range(4)]
        pts2 = np.float32([[dx[0], dy[0]], [w-1+dx[1], dy[1]], [dx[2], h-1+dy[2]], [w-1+dx[3], h-1+dy[3]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 6. Erosion/Dilation
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # 7. Random shadow
    if random.random() < 0.2:
        h, w = img.shape[:2]
        shadow = np.ones((h, w), dtype=np.float32)
        direction = random.choice(['left', 'right', 'top', 'bottom'])
        shadow_intensity = random.uniform(0.3, 0.7)
        if direction == 'left':
            for i in range(w // 2): shadow[:, i] = shadow_intensity + (1 - shadow_intensity) * (i / (w // 2))
        elif direction == 'right':
            for i in range(w // 2, w): shadow[:, i] = shadow_intensity + (1 - shadow_intensity) * ((w - i) / (w // 2))
        elif direction == 'top':
            for i in range(h // 2): shadow[i, :] = shadow_intensity + (1 - shadow_intensity) * (i / (h // 2))
        elif direction == 'bottom':
            for i in range(h // 2, h): shadow[i, :] = shadow_intensity + (1 - shadow_intensity) * ((h - i) / (h // 2))
        img = np.clip(img.astype(np.float32) * shadow, 0, 255).astype(np.uint8)

    # 8. CLAHE
    if random.random() < 0.15:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)

    return img.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────
# MÃ HÓA / GIẢI MÃ NHÃN
# ──────────────────────────────────────────────────────────────

def encode_label(plate_text: str) -> list:
    plate_text = plate_text.upper().strip()
    indices = [config.CHAR2IDX[ch] for ch in plate_text if ch in config.CHAR2IDX]
    return indices


def decode_label(indices: list) -> str:
    return "".join(config.IDX2CHAR.get(i, "?") for i in indices)


# ──────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────

class LicensePlateDataset(Dataset):
    def __init__(self, data_dir: str, labels: list, augment: bool = False):
        self.data_dir = data_dir
        self.labels   = labels
        self.augment  = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        img_path = os.path.join(self.data_dir, item["filename"])
        try:
            pil_img = Image.open(img_path).convert("RGB")
            plate_np = preprocess_plate(pil_img)
        except Exception:
            plate_np = np.zeros((config.PLATE_HEIGHT, config.PLATE_WIDTH), dtype=np.float32)

        if self.augment:
            plate_np = augment_plate(plate_np)

        image_tensor = torch.from_numpy(plate_np).unsqueeze(0)
        label_indices = encode_label(item["plate_text"])
        label_tensor  = torch.tensor(label_indices, dtype=torch.long)

        return image_tensor, label_tensor, len(label_indices)


def collate_fn(batch):
    images, labels, label_lens = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.cat(labels, dim=0)
    target_lengths  = torch.tensor(label_lens, dtype=torch.long)
    T = images.shape[-1] // 4
    input_lengths = torch.full((len(images),), T, dtype=torch.long)
    return images, targets, input_lengths, target_lengths


# ──────────────────────────────────────────────────────────────
# ĐỌC & TÁCH DỮ LIỆU
# ──────────────────────────────────────────────────────────────

def load_labels_csv(csv_path: str) -> list:
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate_text = row["plate_text"].strip().upper()
            if all(ch in config.CHAR2IDX for ch in plate_text) and len(plate_text) > 0:
                items.append({"filename": row["filename"].strip(), "plate_text": plate_text})
    return items


def get_split_data():
    all_labels = load_labels_csv(config.LABELS_FILE)
    random.seed(config.RANDOM_SEED)
    random.shuffle(all_labels)

    n = len(all_labels)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_data = all_labels[:train_end]
    val_data = all_labels[train_end:val_end]
    test_data = all_labels[val_end:]

    # Oversampling ký tự 'B'
    b_data = [x for x in train_data if 'B' in x['plate_text']]
    train_data.extend(b_data)
    random.shuffle(train_data)
    
    return train_data, val_data, test_data


def create_dataloaders_with_test():
    """
    Tạo DataLoaders cho 3 tập: train / val / test.

    QUAN TRỌNG:
      - Train: augmentation + oversampling 'B'
      - Val/Test: không augmentation, KHÔNG DÙNG ĐỂ TRAIN

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_data, val_data, test_data = get_split_data()
    print(f"[Dataset] Tập train  : {len(train_data)} ảnh (70% + oversample)")
    print(f"[Dataset] Tập val    : {len(val_data)} ảnh (15%)")
    print(f"[Dataset] Tập test   : {len(test_data)} ảnh (15%) — KHÔNG DÙNG ĐỂ TRAIN")

    train_ds = LicensePlateDataset(config.PLATES_DIR, train_data, augment=True)
    val_ds   = LicensePlateDataset(config.PLATES_DIR, val_data,   augment=False)
    test_ds  = LicensePlateDataset(config.PLATES_DIR, test_data,  augment=False)

    g = torch.Generator()
    g.manual_seed(config.RANDOM_SEED)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def create_dataloaders():
    """
    Tạo DataLoaders cho tập huấn luyện và kiểm định.
    Giữ nguyên giao diện cũ để các file khác không bị lỗi.
    """
    train_loader, val_loader, _ = create_dataloaders_with_test()
    return train_loader, val_loader


def get_test_labels() -> list:
    """
    Trả về danh sách nhãn của tập test (dùng trong evaluate.py).
    Dùng cùng seed với lúc train để đảm bảo cùng phân chia.
    """
    if not os.path.isfile(config.LABELS_FILE):
        raise FileNotFoundError(f"Không tìm thấy: {config.LABELS_FILE}")
    _, _, test_data = get_split_data()
    return test_data


# ──────────────────────────────────────────────────────────────
# TẠO DỮ LIỆU MẪU
# ──────────────────────────────────────────────────────────────

def generate_synthetic_data(num_samples: int = 500):
    os.makedirs(config.PLATES_DIR, exist_ok=True)
    SERIES_LETTERS = "ABCDEFGHKLMNPSTUVXYZ"
    rows = []
    for i in range(num_samples):
        plate_text = f"{random.randint(10,99)}{random.choice(SERIES_LETTERS)}-{random.randint(10000, 99999)}"
        fname = f"plate_{i:05d}.jpg"
        cv2.imwrite(os.path.join(config.PLATES_DIR, fname), _draw_plate(plate_text))
        rows.append({"filename": fname, "plate_text": plate_text})

    with open(config.LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "plate_text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Dữ liệu mẫu] Đã tạo {num_samples} biển số tổng hợp")
    print(f"  Nhãn lưu tại  : {config.LABELS_FILE}")


def _draw_plate(plate_text: str) -> np.ndarray:
    """Vẽ biển số xe bằng PIL với Font siêu dày để giả lập đời thực."""
    from PIL import ImageDraw, ImageFont

    H, W = 100, 320
    # Tạo nền bằng PIL
    pil_img = Image.new("RGB", (W, H), color=(235, 235, 235))
    draw = ImageDraw.Draw(pil_img)

    # Thử load font siêu dày của Windows (Arial Bold hoặc Impact)
    font_paths = [
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\impact.ttf",
        "C:\\Windows\\Fonts\\consola.ttf"
    ]
    font = None
    for p in font_paths:
        if os.path.exists(p):
            font = ImageFont.truetype(p, 65)
            break
    if font is None:
        font = ImageFont.load_default()

    # Đo kích thước chữ
    bbox = draw.textbbox((0, 0), plate_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    tx = (W - tw) / 2
    ty = (H - th) / 2 - int(th * 0.2) # Chỉnh offset y cho chuẩn giữa

    # Vẽ viền xanh
    draw.rectangle([(2, 2), (W-3, H-3)], outline=(0, 100, 0), width=3)

    # Vẽ chữ
    draw.text((tx, ty), plate_text, fill=(20, 20, 20), font=font)

    # Chuyển về Numpy BGR
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Thêm nhiễu nhẹ
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Tool Chuẩn Bị Dữ Liệu Biển Số Xe")
    parser.add_argument("--prepare", action="store_true",
                        help="Tạo dữ liệu tổng hợp (synthetic) để thử nghiệm")
    parser.add_argument("--num",     type=int, default=500,
                        help="Số lượng biển số tổng hợp (mặc định: 500)")
    args = parser.parse_args()

    if args.prepare:
        generate_synthetic_data(args.num)
    else:
        try:
            train_dl, val_dl, test_dl = create_dataloaders_with_test()
            images, targets, input_lens, target_lens = next(iter(train_dl))
            print(f"\n[Test] images     : {tuple(images.shape)}")
            print(f"[Test] targets    : {tuple(targets.shape)}")
            print(f"[Test] input_lens : {input_lens[:4].tolist()} ...")
            print(f"[Test] target_lens: {target_lens[:4].tolist()} ...")
        except FileNotFoundError as e:
            print(e)
