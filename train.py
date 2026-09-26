"""
train.py — Kịch Bản Huấn Luyện Mô Hình OCR Biển Số Xe
=========================================================
Huấn luyện mạng CRNN (CNN + BiLSTM + CTC) với:
  - Phân chia rõ ràng: Train (70%) / Validation (15%) / Test (15%)
  - Oversampling ký tự hiếm / dễ nhầm ('B')
  - Backup model cũ vào saved_models/plate_recognizer_backup.pth
  - Đo lường đầy đủ metrics:
      + Character Accuracy (Độ chính xác ký tự)
      + Exact Plate Accuracy (Độ chính xác toàn bộ biển số)
      + Character Error Rate - CER (Tỷ lệ lỗi ký tự)
      + Validation Accuracy
      + Thống kê các cặp ký tự bị nhầm (đặc biệt B ↔ 8)
  - So sánh model cũ vs model mới trên tập TEST độc lập
  - Chỉ thay thế model production nếu model mới tốt hơn.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import random
import shutil
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import create_dataloaders_with_test
from model import build_model, get_device, count_parameters, ctc_greedy_decode


# ──────────────────────────────────────────────────────────────
# TÁI LẬP KẾT QUẢ (REPRODUCIBILITY)
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# TÍNH TOÁN METRICS (CER, EXACT MATCH, CONFUSION PAIRS)
# ──────────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Tính khoảng cách Levenshtein (số phép thêm/xóa/sửa tối thiểu)."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_batch_metrics(log_probs: torch.Tensor, targets: torch.Tensor,
                          target_lengths: torch.Tensor) -> dict:
    """
    Tính metrics chi tiết cho 1 batch:
      - char_acc: tỷ lệ ký tự khớp tại cùng vị trí
      - seq_acc (exact plate accuracy): tỷ lệ chuỗi khớp 100%
      - cer: tổng edit distance / tổng độ dài gt
      - confusion_pairs: danh sách các cặp (gt_char, pred_char)
    """
    preds = ctc_greedy_decode(log_probs)

    gt_strings = []
    offset = 0
    for length in target_lengths.tolist():
        indices = targets[offset: offset + length].tolist()
        gt_strings.append("".join(config.IDX2CHAR.get(i, "?") for i in indices))
        offset += length

    correct_chars = 0
    total_chars = 0
    exact_plates = 0
    total_edit_distance = 0
    total_gt_len = 0
    confusions = defaultdict(int)

    for pred, gt in zip(preds, gt_strings):
        if pred == gt:
            exact_plates += 1
            correct_chars += len(gt)
            total_chars += len(gt)
        else:
            # So sánh từng ký tự
            min_len = min(len(pred), len(gt))
            for p_ch, g_ch in zip(pred[:min_len], gt[:min_len]):
                if p_ch == g_ch:
                    correct_chars += 1
                else:
                    confusions[(g_ch, p_ch)] += 1
            # Phần dư độ dài
            total_chars += max(len(pred), len(gt))
            for g_ch in gt[min_len:]:
                confusions[(g_ch, "MISSING")] += 1
            for p_ch in pred[min_len:]:
                confusions[("EXTRA", p_ch)] += 1

        dist = levenshtein_distance(pred, gt)
        total_edit_distance += dist
        total_gt_len += max(len(gt), 1)

    return {
        "correct_chars": correct_chars,
        "total_chars": max(total_chars, 1),
        "exact_plates": exact_plates,
        "total_samples": len(preds),
        "total_edit_dist": total_edit_distance,
        "total_gt_len": total_gt_len,
        "confusions": dict(confusions),
    }


# ──────────────────────────────────────────────────────────────
# ĐÁNH GIÁ TRÊN DATALOADER
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct_chars = 0
    total_chars = 0
    total_exact_plates = 0
    total_samples = 0
    total_edit_dist = 0
    total_gt_len = 0
    all_confusions = defaultdict(int)

    for images, targets, input_lengths, target_lengths in loader:
        images, targets = images.to(device), targets.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

        log_probs = model(images)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()

        batch_m = compute_batch_metrics(log_probs.cpu(), targets.cpu(), target_lengths.cpu())
        total_correct_chars += batch_m["correct_chars"]
        total_chars += batch_m["total_chars"]
        total_exact_plates += batch_m["exact_plates"]
        total_samples += batch_m["total_samples"]
        total_edit_dist += batch_m["total_edit_dist"]
        total_gt_len += batch_m["total_gt_len"]

        for pair, cnt in batch_m["confusions"].items():
            all_confusions[pair] += cnt

    char_acc = total_correct_chars / max(total_chars, 1)
    exact_acc = total_exact_plates / max(total_samples, 1)
    cer = total_edit_dist / max(total_gt_len, 1)
    avg_loss = total_loss / max(len(loader), 1)

    return {
        "loss": avg_loss,
        "char_acc": char_acc,
        "exact_plate_acc": exact_acc,
        "cer": cer,
        "confusions": dict(all_confusions),
        "total_samples": total_samples,
    }


# ──────────────────────────────────────────────────────────────
# VẼ BIỂU ĐỒ QUÁ TRÌNH HUẤN LUYỆN
# ──────────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "training_history.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2563EB", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#DC2626", linewidth=2)
    axes[0].set_title("Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Character Accuracy
    axes[1].plot(epochs, [v * 100 for v in history["train_char_acc"]], label="Train Char Acc", color="#2563EB", linewidth=2)
    axes[1].plot(epochs, [v * 100 for v in history["val_char_acc"]], label="Val Char Acc", color="#16A34A", linewidth=2)
    axes[1].set_title("Character Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Exact Plate Accuracy & CER
    axes[2].plot(epochs, [v * 100 for v in history["val_exact_acc"]], label="Val Exact Plate Acc (%)", color="#9333EA", linewidth=2)
    axes[2].plot(epochs, [v * 100 for v in history["val_cer"]], label="Val CER (%)", color="#EA580C", linewidth=2, linestyle="--")
    axes[2].set_title("Exact Plate Acc & CER (%)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Biểu đồ] Lưu quá trình huấn luyện tại: {save_path}")


# ──────────────────────────────────────────────────────────────
# SO SÁNH MODEL CŨ VÀ MODEL MỚI TRÊN TEST SET
# ──────────────────────────────────────────────────────────────

def compare_models_on_test(old_model_path: str, new_model_path: str,
                           test_loader, criterion, device) -> dict:
    """
    So sánh khách quan 2 model trên cùng 1 Test Set (15% dataset).
    """
    print("\n" + "=" * 70)
    print("  SO SÁNH MODEL CŨ VS MODEL MỚI TRÊN TEST SET (ĐỘC LẬP)")
    print("=" * 70)

    results = {}

    # Đánh giá model cũ (nếu có)
    if os.path.exists(old_model_path):
        old_model = build_model().to(device)
        old_model.load_state_dict(torch.load(old_model_path, map_location=device, weights_only=True))
        old_metrics = evaluate_loader(old_model, test_loader, criterion, device)
        results["old"] = old_metrics
    else:
        results["old"] = None

    # Đánh giá model mới
    new_model = build_model().to(device)
    new_model.load_state_dict(torch.load(new_model_path, map_location=device, weights_only=True))
    new_metrics = evaluate_loader(new_model, test_loader, criterion, device)
    results["new"] = new_metrics

    print(f"\n{'Metric':<25} | {'Model Cũ (Backup)':<18} | {'Model Mới (Candidate)':<22} | {'Thay đổi':<10}")
    print("-" * 82)

    metrics_list = [
        ("Exact Plate Accuracy", "exact_plate_acc", True),
        ("Character Accuracy", "char_acc", True),
        ("Character Error Rate (CER)", "cer", False),
        ("Test Loss", "loss", False),
    ]

    for label, key, higher_is_better in metrics_list:
        new_val = new_metrics[key]
        if results["old"] is not None:
            old_val = results["old"][key]
            diff = (new_val - old_val) * (100 if "acc" in key or "cer" in key else 1)
            sign = "+" if diff > 0 else ""
            if "acc" in key or "cer" in key:
                print(f"{label:<25} | {old_val*100:>16.2f}% | {new_val*100:>20.2f}% | {sign}{diff:>8.2f}%")
            else:
                print(f"{label:<25} | {old_val:>18.4f} | {new_val:>22.4f} | {sign}{diff:>8.4f}")
        else:
            if "acc" in key or "cer" in key:
                print(f"{label:<25} | {'N/A':>18} | {new_val*100:>20.2f}% | {'N/A':>10}")
            else:
                print(f"{label:<25} | {'N/A':>18} | {new_val:>22.4f} | {'N/A':>10}")

    # Kiểm tra riêng cặp B ↔ 8
    print("-" * 82)
    print("  Chi tiết cặp nhầm B ↔ 8 trên tập Test:")
    for model_name, m_dict in [("Model Cũ", results["old"]), ("Model Mới", results["new"])]:
        if m_dict is not None:
            b_as_8 = m_dict["confusions"].get(("B", "8"), 0)
            eight_as_b = m_dict["confusions"].get(("8", "B"), 0)
            print(f"    - {model_name}: 'B' bị đoán thành '8': {b_as_8} lần | '8' bị đoán thành 'B': {eight_as_b} lần")

    return results


# ──────────────────────────────────────────────────────────────
# QUY TRÌNH HUẤN LUYỆN CHÍNH
# ──────────────────────────────────────────────────────────────

def train():
    print("=" * 70)
    print("  HUẤN LUYỆN OCR BIỂN SỐ XE — TỐI ƯU CÂN BẰNG LỚP & ĐÁNH GIÁ TEST SET")
    print("=" * 70)

    set_seed(config.RANDOM_SEED)
    device = get_device()
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. BACKUP MODEL CŨ
    backup_path = config.RECOGNIZER_PATH.replace(".pth", "_backup.pth")
    if os.path.exists(config.RECOGNIZER_PATH):
        shutil.copy2(config.RECOGNIZER_PATH, backup_path)
        print(f"[✓ BACKUP] Đã sao lưu model hiện tại sang: {backup_path}")
    else:
        print(f"[i INFO] Chưa có model cũ tại {config.RECOGNIZER_PATH} để backup.")

    # 2. TẠO DATALOADERS 3 TẬP (TRAIN 70% + OVERSAMPLE B / VAL 15% / TEST 15%)
    print("\n[1/4] Chuẩn bị DataLoaders (Train / Val / Test)...")
    train_dl, val_dl, test_dl = create_dataloaders_with_test()

    # 3. KHỞI TẠO MÔ HÌNH VÀ OPTIMIZER
    model = build_model().to(device)
    total_params = count_parameters(model)
    print(f"[2/4] Khởi tạo mô hình CRNN ({total_params:,} tham số) trên {device}")

    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config.SCHEDULER_FACTOR,
                                  patience=config.SCHEDULER_PATIENCE, min_lr=config.MIN_LR)

    candidate_model_path = os.path.join(config.SAVED_MODELS_DIR, "plate_recognizer_candidate.pth")

    print(f"\n[3/4] Bắt đầu huấn luyện tối đa {config.NUM_EPOCHS} vòng...")
    print(f"{'Vòng':>5} | {'Tr Loss':>8} | {'Tr Char%':>9} | {'Val Loss':>8} | {'Val Char%':>9} | "
          f"{'Val Exact%':>10} | {'Val CER%':>9} | {'LR':>8} | {'Thời gian':>9}")
    print("-" * 96)

    history = {
        "train_loss": [], "val_loss": [],
        "train_char_acc": [], "val_char_acc": [],
        "val_exact_acc": [], "val_cer": [],
        "best_epoch": 0,
    }

    best_val_exact_acc = -1.0
    patience_counter = 0
    patience = config.PATIENCE

    for epoch in range(config.NUM_EPOCHS):
        t0 = time.time()

        # Training epoch
        model.train()
        train_loss = 0.0
        train_correct_chars = 0
        train_total_chars = 0

        for images, targets, input_lengths, target_lengths in train_dl:
            images, targets = images.to(device), targets.to(device)
            input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            bm = compute_batch_metrics(log_probs.detach().cpu(), targets.cpu(), target_lengths.cpu())
            train_correct_chars += bm["correct_chars"]
            train_total_chars += bm["total_chars"]

        avg_train_loss = train_loss / max(len(train_dl), 1)
        avg_train_char_acc = train_correct_chars / max(train_total_chars, 1)

        # Validation epoch
        val_m = evaluate_loader(model, val_dl, criterion, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch+1:>5} | {avg_train_loss:>8.4f} | {avg_train_char_acc*100:>8.2f}% | "
              f"{val_m['loss']:>8.4f} | {val_m['char_acc']*100:>8.2f}% | "
              f"{val_m['exact_plate_acc']*100:>9.2f}% | {val_m['cer']*100:>8.2f}% | {lr:>8.1e} | {elapsed:>8.1f}s")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_m["loss"])
        history["train_char_acc"].append(avg_train_char_acc)
        history["val_char_acc"].append(val_m["char_acc"])
        history["val_exact_acc"].append(val_m["exact_plate_acc"])
        history["val_cer"].append(val_m["cer"])

        scheduler.step(val_m["exact_plate_acc"])

        # Early Stopping dựa trên Exact Plate Accuracy
        if val_m["exact_plate_acc"] > best_val_exact_acc + config.MIN_DELTA:
            best_val_exact_acc = val_m["exact_plate_acc"]
            history["best_epoch"] = epoch
            patience_counter = 0
            torch.save(model.state_dict(), candidate_model_path)
            print(f"  [✓ LƯU CANDIDATE] Vòng {epoch+1}: Exact Plate Acc = {best_val_exact_acc*100:.2f}%, CER = {val_m['cer']*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[STOP] Dừng sớm sau {patience} vòng không cải thiện (Tốt nhất: vòng {history['best_epoch']+1})")
                break

    # 4. SO SÁNH VỚI MODEL CŨ TRÊN TẬP TEST
    print("\n[4/4] Đánh giá so sánh trên tập TEST...")
    plot_training_history(history)

    comp_results = compare_models_on_test(
        old_model_path=backup_path,
        new_model_path=candidate_model_path,
        test_loader=test_dl,
        criterion=criterion,
        device=device
    )

    old_res = comp_results["old"]
    new_res = comp_results["new"]

    # Quyết định thay thế model production:
    # Điều kiện: Nếu chưa có model cũ HOẶC model mới có Exact Plate Acc >= model cũ và CER <= model cũ
    should_replace = True
    if old_res is not None:
        old_acc = old_res["exact_plate_acc"]
        new_acc = new_res["exact_plate_acc"]
        old_cer = old_res["cer"]
        new_cer = new_res["cer"]

        if (new_acc < old_acc - 0.005) or (new_cer > old_cer + 0.01):
            should_replace = False

    print("\n" + "=" * 70)
    if should_replace:
        shutil.copy2(candidate_model_path, config.RECOGNIZER_PATH)
        print("  [KẾT LUẬN] Model MỚI TỐT HƠN / ĐẠT CHUẨN!")
        print(f"  -> Đã cập nhật vào: {config.RECOGNIZER_PATH}")
        print(f"  -> Model cũ đã được backup an toàn tại: {backup_path}")
    else:
        print("  [KẾT LUẬN] Model mới chưa vượt qua model cũ trên test set.")
        print(f"  -> Giữ nguyên model production: {config.RECOGNIZER_PATH}")
        print(f"  -> Model candidate lưu tại: {candidate_model_path}")
    print("=" * 70)

    return history


if __name__ == "__main__":
    train()
