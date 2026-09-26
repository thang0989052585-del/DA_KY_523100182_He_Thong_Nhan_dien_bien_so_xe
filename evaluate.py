"""
evaluate.py — Đánh Giá Toàn Diện & So Sánh Mô Hình OCR Biển Số Xe
================================================================
File này thực hiện các phân tích:
  1. Tính Character Accuracy, Exact Plate Accuracy, Character Error Rate (CER)
  2. Thống kê phân phối lỗi theo từng ký tự & các cặp ký tự dễ nhầm (Top confused pairs)
  3. So sánh trực tiếp Model cũ vs Model mới trên cùng tập Test độc lập
  4. Test trên ảnh cụ thể (e.g. biển số 29 AB 90150)
  5. Vẽ Confusion Matrix và xuất báo cáo kết quả ra file .txt
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import build_model, get_device, load_model, ctc_greedy_decode
from dataset import create_dataloaders_with_test, LicensePlateDataset, collate_fn
from detector import preprocess_plate


# ──────────────────────────────────────────────────────────────
# TÍNH METRICS & LEVENSHTEIN DISTANCE
# ──────────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


@torch.no_grad()
def full_evaluation(model, loader, device) -> dict:
    """
    Đánh giá toàn diện trên DataLoader và thu thập thống kê chi tiết.
    """
    model.eval()

    results = []
    correct_chars = 0
    total_chars = 0
    exact_plates = 0
    total_samples = 0
    total_edit_distance = 0
    total_gt_len = 0

    char_errors = defaultdict(int)       # gt_char -> count
    confusion_pairs = defaultdict(int)   # (gt_char, pred_char) -> count

    for images, targets, input_lengths, target_lengths in loader:
        images = images.to(device)
        log_probs = model(images)
        preds = ctc_greedy_decode(log_probs.cpu())

        offset = 0
        for b, length in enumerate(target_lengths.tolist()):
            gt_indices = targets[offset: offset + length].tolist()
            gt_str = "".join(config.IDX2CHAR.get(i, "?") for i in gt_indices)
            pred_str = preds[b]

            is_correct = (pred_str == gt_str)
            if is_correct:
                exact_plates += 1
                correct_chars += len(gt_str)
                total_chars += len(gt_str)
            else:
                min_len = min(len(pred_str), len(gt_str))
                for p_ch, g_ch in zip(pred_str[:min_len], gt_str[:min_len]):
                    if p_ch == g_ch:
                        correct_chars += 1
                    else:
                        char_errors[g_ch] += 1
                        confusion_pairs[(g_ch, p_ch)] += 1
                total_chars += max(len(pred_str), len(gt_str))
                for g_ch in gt_str[min_len:]:
                    char_errors[g_ch] += 1
                    confusion_pairs[(g_ch, "MISSING")] += 1
                for p_ch in pred_str[min_len:]:
                    confusion_pairs[("EXTRA", p_ch)] += 1

            dist = levenshtein_distance(pred_str, gt_str)
            total_edit_distance += dist
            total_gt_len += max(len(gt_str), 1)
            total_samples += 1

            results.append({
                "gt": gt_str,
                "pred": pred_str,
                "correct": is_correct,
                "edit_dist": dist
            })
            offset += length

    char_acc = correct_chars / max(total_chars, 1)
    exact_plate_acc = exact_plates / max(total_samples, 1)
    cer = total_edit_distance / max(total_gt_len, 1)

    return {
        "char_acc": char_acc,
        "exact_plate_acc": exact_plate_acc,
        "cer": cer,
        "results": results,
        "char_errors": dict(char_errors),
        "confusion_pairs": dict(confusion_pairs),
        "total_samples": total_samples,
        "exact_plates": exact_plates,
    }


# ──────────────────────────────────────────────────────────────
# VẼ BIỂU ĐỒ & BÁO CÁO
# ──────────────────────────────────────────────────────────────

def plot_char_error_distribution(char_errors: dict, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "char_error_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not char_errors:
        print("[Đánh giá] Không có lỗi ký tự nào để vẽ.")
        return

    sorted_items = sorted(char_errors.items(), key=lambda x: -x[1])[:20]
    chars = [k for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(max(8, len(chars) * 0.5), 5))
    bars = ax.bar(chars, counts, color="#E74C3C", edgecolor="white", linewidth=0.5)
    ax.set_title("Phân Phối Lỗi Ký Tự (Top 20 Ground Truth bị đoán sai)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ký tự đúng")
    ax.set_ylabel("Số lần bị đoán sai")
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Biểu đồ] Đã lưu phân phối lỗi tại: {save_path}")


def save_evaluation_report(val_metrics: dict, test_metrics: dict = None, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "evaluation_report.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  BÁO CÁO ĐÁNH GIÁ MÔ HÌNH NHẬN DẠNG BIỂN SỐ XE (CRNN)\n")
        f.write("=" * 70 + "\n\n")

        # 1. Bảng tóm tắt
        f.write("1. TỔNG QUAN CHỈ SỐ METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Metric':<30} | {'Validation Set':<15} | {'Test Set (Độc lập)':<18}\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Character Accuracy':<30} | {val_metrics['char_acc']*100:>13.2f}% | "
                f"{(test_metrics['char_acc']*100 if test_metrics else 0.0):>16.2f}%\n")
        f.write(f"  {'Exact Plate Accuracy':<30} | {val_metrics['exact_plate_acc']*100:>13.2f}% | "
                f"{(test_metrics['exact_plate_acc']*100 if test_metrics else 0.0):>16.2f}%\n")
        f.write(f"  {'Character Error Rate (CER)':<30} | {val_metrics['cer']*100:>13.2f}% | "
                f"{(test_metrics['cer']*100 if test_metrics else 0.0):>16.2f}%\n")
        f.write(f"  {'Tổng số mẫu':<30} | {val_metrics['total_samples']:>14} | "
                f"{(test_metrics['total_samples'] if test_metrics else 0):>17}\n")
        f.write("-" * 70 + "\n\n")

        # 2. Top cặp ký tự bị nhầm
        active_m = test_metrics if test_metrics else val_metrics
        f.write("2. TOP CÁC CẶP KÝ TỰ BỊ NHẦM LẪN NHIỀU NHẤT (GT -> Pred):\n")
        f.write("-" * 70 + "\n")
        sorted_conf = sorted(active_m["confusion_pairs"].items(), key=lambda x: -x[1])
        for (g_ch, p_ch), cnt in sorted_conf[:20]:
            f.write(f"  '{g_ch}' nhầm thành '{p_ch}': {cnt} lần\n")

        # 3. Mẫu sai tiêu biểu
        f.write("\n3. MẪU DỰ ĐOÁN SAI TIÊU BIỂU (Tối đa 30 mẫu):\n")
        f.write("-" * 70 + "\n")
        wrong_cases = [r for r in active_m["results"] if not r["correct"]]
        for r in wrong_cases[:30]:
            f.write(f"  GT: {r['gt']:<15}  ->  Pred: {r['pred']:<15} (Edit Dist: {r['edit_dist']})\n")

    print(f"[Báo cáo] Đã lưu báo cáo hoàn chỉnh tại: {save_path}")


# ──────────────────────────────────────────────────────────────
# TEST TRÊN ẢNH CỤ THỂ
# ──────────────────────────────────────────────────────────────

def test_single_image(image_path: str, model_path: str = config.RECOGNIZER_PATH):
    """
    Test trực tiếp một file ảnh biển số (VD: ảnh 29 AB 90150).
    """
    print("\n" + "=" * 60)
    print(f"  TEST ẢNH BIỂN SỐ: {os.path.basename(image_path)}")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"[LỖI] Không tìm thấy ảnh: {image_path}")
        return

    device = get_device()
    model = load_model(model_path, device)
    model.eval()

    pil_img = Image.open(image_path).convert("RGB")
    plate_np = preprocess_plate(pil_img)
    tensor = torch.from_numpy(plate_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(tensor)
        pred_text = ctc_greedy_decode(log_probs.cpu())[0]

    print(f"  Đường dẫn ảnh   : {image_path}")
    print(f"  Model sử dụng   : {model_path}")
    print(f"  KẾT QUẢ DỰ ĐOÁN : '{pred_text}'")
    print("=" * 60)
    return pred_text


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Đánh Giá Mô Hình OCR Biển Số")
    parser.add_argument("--test", action="store_true", help="Đánh giá trên tập test")
    parser.add_argument("--compare", action="store_true", help="So sánh model cũ và model mới")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh để test nhận dạng")
    parser.add_argument("--model", type=str, default=config.RECOGNIZER_PATH, help="Path model")
    args = parser.parse_args()

    device = get_device()

    if args.image:
        test_single_image(args.image, args.model)
        return

    if not os.path.exists(args.model):
        print(f"[LỖI] Không tìm thấy file model: {args.model}")
        return

    print("=" * 70)
    print(f"  ĐÁNH GIÁ MÔ HÌNH: {args.model}")
    print("=" * 70)

    train_dl, val_dl, test_dl = create_dataloaders_with_test()
    model = load_model(args.model, device)

    print("\n[1/3] Đánh giá trên tập Validation...")
    val_m = full_evaluation(model, val_dl, device)

    print("\n[2/3] Đánh giá trên tập Test độc lập...")
    test_m = full_evaluation(model, test_dl, device)

    print("\n" + "=" * 70)
    print(f"  KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (15% dataset độc lập):")
    print(f"    - Exact Plate Accuracy : {test_m['exact_plate_acc']*100:.2f}% ({test_m['exact_plates']}/{test_m['total_samples']})")
    print(f"    - Character Accuracy   : {test_m['char_acc']*100:.2f}%")
    print(f"    - Character Error Rate : {test_m['cer']*100:.2f}%")
    print("=" * 70)

    # Top confused pairs on test set
    print("\n  Top 10 cặp ký tự bị nhầm lẫn trên Test set:")
    sorted_conf = sorted(test_m["confusion_pairs"].items(), key=lambda x: -x[1])
    for (g_ch, p_ch), cnt in sorted_conf[:10]:
        print(f"    '{g_ch}' -> '{p_ch}': {cnt} lần")

    print("\n[3/3] Xuất biểu đồ và báo cáo text...")
    plot_char_error_distribution(test_m["char_errors"])
    save_evaluation_report(val_m, test_m)


if __name__ == "__main__":
    main()
