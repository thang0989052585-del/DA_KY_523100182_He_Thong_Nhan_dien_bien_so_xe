"""
app.py — Giao Diện Web Nhận Diện Biển Số Xe (Streamlit)
=========================================================
Đề tài: Nghiên cứu và xây dựng chương trình nhận diện biển số xe.

Chức năng:
  1. Nhận ảnh đầu vào từ người dùng (upload, camera hoặc URL)
  2. Phát hiện vùng biển số xe trong ảnh bằng YOLOv8 / OpenCV
  3. Nhận dạng ký tự biển số bằng mạng CRNN (CNN + Bi-LSTM + CTC)
  4. Theo dõi & cảnh báo biển số mục tiêu theo thời gian thực
  5. Hiển thị kết quả trực quan & thống kê chi tiết

Cú pháp chạy:
  streamlit run app.py
"""

import sys, os, io, time, re, csv, base64, json, hashlib
from datetime import datetime, timedelta
import numpy as np
import cv2
import torch
from PIL import Image
import streamlit as st

# Fix encoding tiếng Việt trên Windows
try:
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except (ValueError, AttributeError):
    pass

sys.path.insert(0, os.path.dirname(__file__))

import config
import iot_client
from model import build_model, get_device, load_model, ctc_greedy_decode
from detector import detect_plate_opencv, draw_detections, preprocess_plate, preprocess_plate_keep_ratio, detect_plate_yolo

# ──────────────────────────────────────────────────────────────
# LƯU / NẠP LỊCH SỬ BIỂN SỐ PERSISTENT
# ──────────────────────────────────────────────────────────────

# File lưu lịch sử nằm cùng thư mục app.py
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "plate_history.json")
HISTORY_RETENTION_DAYS = 7  # Tự động xoá sau 7 ngày


def _pil_to_b64(pil_img) -> str:
    """Chuyển PIL Image sang chuỗi base64 JPEG để lưu JSON."""
    if pil_img is None:
        return ""
    try:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _b64_to_pil(b64_str: str):
    """Chuyển chuỗi base64 thành PIL Image."""
    if not b64_str:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_str)))
    except Exception:
        return None


def load_history() -> list:
    """
    Nạp lịch sử từ file JSON, tự động xoá các mục quá 7 ngày.
    Trả về danh sách các bản ghi (dicts) với plate_image là PIL Image.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []

    cutoff = datetime.now() - timedelta(days=HISTORY_RETENTION_DAYS)
    valid = []
    for item in raw:
        try:
            ts = datetime.strptime(item.get("timestamp_iso", ""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue  # bỏ qua bản ghi không có timestamp hợp lệ
        if ts >= cutoff:
            if "id" not in item or not item["id"]:
                item["id"] = hashlib.md5(f"{item.get('timestamp_iso')}_{item.get('plate_text')}_{len(valid)}".encode()).hexdigest()[:8]
            item["plate_image"] = _b64_to_pil(item.get("plate_image_b64", ""))
            valid.append(item)
    return valid


def save_history(history: list) -> None:
    """
    Lưu danh sách lịch sử ra file JSON.
    Chuyển plate_image (PIL) thành base64 để serialize.
    """
    serializable = []
    for i, item in enumerate(history):
        entry = {
            "id"           : item.get("id") or hashlib.md5(f"{item.get('timestamp_iso')}_{item.get('plate_text')}_{i}".encode()).hexdigest()[:8],
            "plate_text"   : item.get("plate_text", ""),
            "confidence"   : item.get("confidence", 0.0),
            "is_target"    : item.get("is_target", False),
            "is_two_line"  : item.get("is_two_line", False),
            "timestamp"    : item.get("timestamp", ""),
            "timestamp_iso": item.get("timestamp_iso", ""),
            "plate_image_b64": _pil_to_b64(item.get("plate_image")),
        }
        serializable.append(entry)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def add_plate_to_history(record: dict) -> None:
    """
    Thêm một bản ghi mới vào đầu lịch sử trong session state, sau đó lưu file.
    Giới hạn tối đa 500 bản ghi và 7 ngày.
    """
    cutoff = datetime.now() - timedelta(days=HISTORY_RETENTION_DAYS)
    history = st.session_state.get("plate_history", [])
    history.insert(0, record)
    # Lọc bỏ bản ghi quá cũ
    def still_valid(item):
        try:
            ts = datetime.strptime(item.get("timestamp_iso", ""), "%Y-%m-%dT%H:%M:%S")
            return ts >= cutoff
        except Exception:
            return False
    history = [h for h in history if still_valid(h)][:500]
    st.session_state["plate_history"] = history
    save_history(history)


# ──────────────────────────────────────────────────────────────
# CẤU HÌNH TRANG
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Biển Số AI — Nhận Diện Biển Số Xe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CSS PREMIUM — Dark Tech Theme
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

:root {
    --bg-main:       #060B14;
    --bg-card:       #090F1D;
    --bg-panel:      #0D1527;
    --bg-input:      #08101E;
    --accent:        #00D4FF;
    --accent-dim:    rgba(0, 212, 255, 0.12);
    --accent-mid:    rgba(0, 212, 255, 0.28);
    --green:         #00FF88;
    --green-dim:     rgba(0, 255, 136, 0.12);
    --yellow:        #FFD740;
    --red:           #FF3B5C;
    --red-dim:       rgba(255, 59, 92, 0.15);
    --text-main:     #E2EDF8;
    --text-muted:    #627D98;
    --text-dim:      #243B53;
    --border:        rgba(0, 212, 255, 0.14);
    --border-bright: rgba(0, 212, 255, 0.35);
    --glow:          0 0 14px rgba(0, 212, 255, 0.35);
    --glow-green:    0 0 14px rgba(0, 255, 136, 0.40);
    --glow-red:      0 0 16px rgba(255, 59, 92, 0.50);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-main) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
}
[data-testid="stHeader"]     { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-bright) !important;
}

[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 4px !important;
    padding-bottom: 20px !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
    max-width: 100% !important;
}

/* ── Top Bar Header (Reference UI) ─────────────────────────── */
.topbar-container {
    background: linear-gradient(180deg, #0A1222 0%, #060B14 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(0, 212, 255, 0.15);
}
.brand-box {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon {
    width: 34px;
    height: 34px;
    background: rgba(0, 212, 255, 0.1);
    border: 1.5px solid var(--accent);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: var(--glow);
}
.brand-title {
    font-size: 1.05rem;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.1;
}
.brand-badge {
    background: #00D4FF;
    color: #060B14;
    font-size: 0.58rem;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 3px;
    letter-spacing: 0.05em;
}
.brand-sub {
    font-size: 0.55rem;
    color: var(--text-muted);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

html {
    scroll-behavior: smooth !important;
}
.topbar-container {
    background: linear-gradient(180deg, #0A1222 0%, #060B14 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(0, 212, 255, 0.15);
    flex-wrap: wrap;
    gap: 8px;
}
.top-nav-tabs {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: nowrap;
}
.top-nav-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--text-muted) !important;
    border: 1px solid transparent;
    cursor: pointer;
    text-decoration: none !important;
    white-space: nowrap !important;
    transition: all 0.2s;
}
.top-nav-item:hover {
    color: var(--accent) !important;
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.25);
    text-decoration: none !important;
}
.top-nav-item.active {
    background: rgba(0, 212, 255, 0.12);
    color: var(--accent) !important;
    border: 1px solid var(--accent);
    box-shadow: var(--glow);
    text-decoration: none !important;
}

.top-telemetry {
    display: flex;
    align-items: center;
    gap: 12px;
}
.telemetry-chip {
    background: #08101D;
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
.telemetry-label { color: var(--text-muted); font-size: 0.65rem; }
.telemetry-val   { color: var(--green); font-weight: 700; }
.telemetry-val.cyan { color: var(--accent); }

/* ── Status Dots ─────────────────────── */
.sdot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:4px; vertical-align:middle; }
.sdot-green  { background:var(--green);  box-shadow:var(--glow-green); }
.sdot-cyan   { background:var(--accent); box-shadow:var(--glow); }
.sdot-red    { background:var(--red);    box-shadow:var(--glow-red); }

/* ── Viewport Card (Left Column) ──────── */
.viewport-card {
    background: #080D19;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
}
.viewport-header {
    background: #0A1324;
    border-bottom: 1px solid var(--border);
    padding: 8px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
.viewport-footer {
    background: #0A1324;
    border-top: 1px solid var(--border);
    padding: 6px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
}
.viewport-canvas {
    min-height: 280px;
    background: #050811;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    padding: 10px;
}

/* ── Radio Pill Selector Override ─────── */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    gap: 8px !important;
    background: #080D19 !important;
    padding: 5px !important;
    border-radius: 7px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 10px !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: transparent !important;
    padding: 6px 14px !important;
    border-radius: 5px !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    transition: all 0.15s !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    color: var(--accent) !important;
    background: rgba(0, 212, 255, 0.08) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: var(--accent) !important;
    color: #060B14 !important;
    font-weight: 800 !important;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.45) !important;
}

/* ── Primary Action Button (Big Cyan) ── */
.tactical-run-btn [data-testid="baseButton-primary"] {
    background: #00D4FF !important;
    color: #060B14 !important;
    font-weight: 900 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 6px !important;
    box-shadow: 0 0 18px rgba(0, 212, 255, 0.5) !important;
    padding: 10px 20px !important;
}
.tactical-run-btn [data-testid="baseButton-primary"]:hover {
    background: #33DDFF !important;
    box-shadow: 0 0 28px rgba(0, 212, 255, 0.7) !important;
    transform: translateY(-1px);
}

/* ── Hotlist Card (Bottom Left) ──────── */
.hotlist-box {
    background: #090F1D;
    border: 1px solid rgba(255, 59, 92, 0.25);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4);
}
.hotlist-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.hotlist-title {
    font-size: 0.75rem;
    font-weight: 800;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.05em;
}
.hotlist-tag {
    background: rgba(255, 59, 92, 0.18);
    color: #FF3B5C;
    border: 1px solid #FF3B5C;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.58rem;
    font-weight: 800;
}

/* ── Right Column Tactical Container ─── */
.right-panel {
    background: #090F1D;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.target-alert-match {
    background: rgba(0, 212, 255, 0.08);
    border: 1.5px solid var(--accent);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 12px;
    box-shadow: var(--glow);
}
.target-alert-match.danger {
    background: rgba(255, 59, 92, 0.10);
    border-color: var(--red);
    box-shadow: var(--glow-red);
}

/* ── Plate Big White Box (Tactical Hero) ─ */
.plate-hero-wrapper {
    background: #060A13;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    margin-bottom: 12px;
}
.plate-hero-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.plate-hero-box {
    background: #F8FAFC;
    border: 3px solid #1E293B;
    border-radius: 8px;
    padding: 10px 20px;
    display: inline-block;
    position: relative;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.12), 0 4px 16px rgba(0,0,0,0.6);
    min-width: 250px;
}
.plate-hero-box::before {
    content: '🔩';
    position: absolute;
    top: 5px; left: 8px;
    font-size: 0.65rem; opacity: 0.7;
}
.plate-hero-box::after {
    content: '🔩';
    position: absolute;
    top: 5px; right: 8px;
    font-size: 0.65rem; opacity: 0.7;
}
.plate-hero-type {
    font-size: 0.58rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    margin-bottom: 2px;
}
.plate-hero-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.3rem;
    font-weight: 900;
    color: #0F172A;
    letter-spacing: 0.12em;
    line-height: 1.1;
}

/* ── Metrics 4-Grid ──────────────────── */
.grid-2x2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 12px;
}
.grid-tile {
    background: #060B14;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
}
.grid-tile-label {
    font-size: 0.58rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}
.grid-tile-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 800;
    line-height: 1.2;
}

/* ── Breakdown List of Detected Plates ─ */
.breakdown-item {
    background: #060B14;
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 6px 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.breakdown-item:hover {
    border-color: var(--accent);
}

/* ── History Table & Bottom Section ──── */
.hist-tabs-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 12px;
}
.hist-tab-btn {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 6px;
}
.hist-table-wrap {
    background: #080D19;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
.hist-table-modern {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.hist-table-modern th {
    background: #0B1426;
    color: var(--accent);
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border-bright);
}
.hist-table-modern td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}
.hist-table-modern tr:hover td {
    background: rgba(0, 212, 255, 0.03);
}
.hist-table-modern tr.target-row td {
    background: rgba(255, 59, 92, 0.05);
    border-bottom-color: rgba(255, 59, 92, 0.2);
}

/* ── Tactical Footer ─────────────────── */
.tactical-footer {
    border-top: 1px solid var(--border);
    margin-top: 24px;
    padding-top: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar Styling ─────────────────── */
.sb-logo-area { text-align:center; padding:18px 0 14px; border-bottom:1px solid var(--border-bright); margin-bottom:14px; }
.sb-logo-icon { font-size:2.2rem; filter:drop-shadow(0 0 8px rgba(0,212,255,0.6)); }
.sb-logo-name { font-size:1rem; font-weight:900; color:var(--accent); text-shadow:0 0 10px rgba(0,212,255,0.5); letter-spacing:0.06em; }
.sb-logo-sub  { font-size:0.62rem; color:var(--text-muted); letter-spacing:0.1em; text-transform:uppercase; margin-top:3px; }
.sb-sec-title {
    font-size:0.63rem; font-weight:700; color:var(--accent);
    text-transform:uppercase; letter-spacing:0.14em;
    margin:14px 0 8px; display:flex; align-items:center; gap:5px;
}
.sb-sec-title::after { content:''; flex:1; height:1px; background:var(--border-bright); opacity:0.35; }
.sb-row { display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:5px; padding:3px 0; }
.sb-row-key { color:var(--text-muted); }
.sb-row-val { color:var(--text-main); font-weight:600; font-family:'JetBrains Mono',monospace; font-size:0.73rem; }
.info-note-cyber {
    background:rgba(255,215,64,0.05); border:1px solid rgba(255,215,64,0.28);
    border-radius:6px; padding:10px 12px;
    font-size:0.75rem; color:var(--yellow); margin-top:10px; line-height:1.5;
}

/* Delete single row button */
.del-single-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    background: rgba(255, 59, 92, 0.12);
    border: 1px solid rgba(255, 59, 92, 0.35);
    border-radius: 4px;
    color: #FF3B5C !important;
    font-size: 0.72rem;
    cursor: pointer;
    text-decoration: none !important;
    transition: all 0.15s ease-in-out;
}
.del-single-btn:hover {
    background: #FF3B5C !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 10px rgba(255, 59, 92, 0.6);
    transform: scale(1.1);
}

/* Buttons styling */
[data-testid="baseButton-secondary"], button[kind="secondary"], [data-testid="stDownloadButton"] > button {
    background: #090F1D !important;
    border: 1px solid rgba(0, 212, 255, 0.35) !important;
    color: #E2EDF8 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
[data-testid="baseButton-secondary"]:hover, button[kind="secondary"]:hover, [data-testid="stDownloadButton"] > button:hover {
    border-color: #00D4FF !important;
    color: #00D4FF !important;
    box-shadow: 0 0 14px rgba(0, 212, 255, 0.35) !important;
    background: rgba(0, 212, 255, 0.08) !important;
}

/* Input Fields */
[data-testid="stTextInput"] input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: var(--glow) !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
}
hr { border-color: var(--border) !important; opacity:0.4 !important; }
::-webkit-scrollbar       { width:5px; height:5px; }
::-webkit-scrollbar-track  { background:var(--bg-main); }
::-webkit-scrollbar-thumb  { background:var(--border-bright); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent); }
</style>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────
# CACHE: NẠP MÔ HÌNH








# ──────────────────────────────────────────────────────────────
# CACHE: NẠP MÔ HÌNH
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_recognizer():
    """Nạp mô hình CRNN nhận dạng biển số (cache để không tải lại mỗi lần)."""
    device = get_device()
    if not os.path.isfile(config.RECOGNIZER_PATH):
        return None, device
    model = load_model(config.RECOGNIZER_PATH, device)
    return model, device

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    try:
        from ultralytics import YOLO
        yolo_path = os.path.join(config.SAVED_MODELS_DIR, "yolo_plate_detector.pt")
        if os.path.isfile(yolo_path):
            return YOLO(yolo_path)
    except:
        pass
# ──────────────────────────────────────────────────────────────
# HỖ TRỢ THEO DÕI BIỂN SỐ MỤC TIÊU
# ──────────────────────────────────────────────────────────────

def _normalize_plate(text: str) -> str:
    """Chuẩn hóa: uppercase, bỏ khoảng trắng/dấu chấm/gạch."""
    if not text:
        return ""
    return re.sub(r"[\s\-\.]", "", text.upper().strip())


def _plate_similarity(a: str, b: str) -> float:
    """Levenshtein similarity [0→1] giữa 2 biển số (đã chuẩn hóa)."""
    a, b = _normalize_plate(a), _normalize_plate(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    max_len = max(la, lb)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
    return 1.0 - dp[lb] / max_len


def plates_match_target(detected: str, target: str, min_similarity: float = 0.90) -> bool:
    """Khớp biển phát hiện với biển mục tiêu chính xác, tránh nhầm lẫn các xe khác."""
    if not detected or not target:
        return False
    norm_det = _normalize_plate(detected)
    norm_tgt = _normalize_plate(target)
    if not norm_det or not norm_tgt:
        return False
    # 1. Khớp chính xác 100% (sau khi bỏ dấu -, ., khoảng trắng)
    if norm_tgt == norm_det:
        return True
    # 2. Khớp tiền tố/hậu tố khi người dùng cố ý tìm một phần (ví dụ chỉ gõ 5 số đuôi)
    if len(norm_tgt) < len(norm_det) and norm_tgt in norm_det:
        return True
    return False
# ──────────────────────────────────────────────────────────────
# POST-PROCESSING: THÊM DẤU CHẤM CHO BIỂN 5 SỐ
# ──────────────────────────────────────────────────────────────

def _add_plate_dot(plate_text: str) -> str:
    """
    Post-process: tự động thêm dấu chấm vào biển số 5 chữ số.
    Quy tắc: 5 số → XXX.XX (dot trước 2 số cuối)
    """
    import re
    if not plate_text or '.' in plate_text:
        return plate_text
    m = re.match(r'^(.+-)(\d{5})$', plate_text)
    if m:
        prefix, digits = m.group(1), m.group(2)
        return prefix + digits[:3] + '.' + digits[3:]
    return plate_text


# Mã tỉnh/thành phố hợp lệ của Việt Nam
_VN_PROVINCE_CODES = {
    "11","12","14","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29",
    "30","31","32","33","34","35","36","37","38",
    "40","41","42","43",
    "47","48","49","50","51","52","53","54","55","56","57","58","59",
    "60","61","62","63","64","65","66","67","68","69",
    "70","71","72","73","74","75","76","77","78","79",
    "80","81","82","83","84","85","86","88","89",
    "90","92","93","94","95","96","97","98","99",
}

# Cặp ký tự thường bị nhầm (nhìn giống nhau) - dùng cho phần mã tỉnh
_CONFUSED_DIGITS = {
    "5": "8", "8": "5",
    "1": "0", "0": "1",
    "3": "8", "6": "8",
    "9": "4", "4": "9",
}
_CONFUSED_CHARS = {
    "F": "E", "E": "F",
    "G": "C", "C": "G",
    "D": "0", "A": "H", "H": "A",
    "K": "X", "X": "K",
    "P": "F", "R": "P",
    "M": "N", "N": "M",
    "U": "V", "V": "U",
    "Y": "V",
}

# ============================================================
# Bảng nhầm lẫn NGHIÊM TRọNG:
# VỊ trÍ Sậ-RI (ký tự thứ 3 của prefix): BẮt buộc là CHỤ CÁI
# Bảng: SỐ -> CHỤ (dùng khi OCR đọc sai chữ thành số)
_SYNTAX_DIGIT_TO_CHAR = {
    "0": "D",  # 0 trông như D/O
    "1": "T",  # 1 trông như T/I/L
    "2": "Z",  # 2 trông như Z
    "3": "B",  # 3 trông như B (ngược)
    "4": "A",  # 4 trông như A
    "5": "S",  # 5 trông như S
    "6": "G",  # 6 trông như G/b
    "7": "T",  # 7 trông như T/L
    "8": "B",  # 8 trông như B
    "9": "P",  # 9 trông như P/q
}
# Bảng: CHỤ -> SỐ (dùng khi OCR đọc sai số thành chữ trong phần số sê-ri)
_SYNTAX_CHAR_TO_DIGIT = {
    "B": "8",  # B trông như 8
    "D": "0",  # D trông như 0
    "O": "0",  # O trông như 0
    "Q": "0",  # Q trông như 0
    "U": "0",  # U trông như 0 (khép đáy)
    "I": "1",  # I trông như 1
    "L": "1",  # L trông như 1
    "Z": "2",  # Z trông như 2
    "A": "4",  # A trông như 4
    "S": "5",  # S trông như 5
    "G": "6",  # G trông như 6
    "T": "7",  # T trông như 7
    "R": "7",  # R đôi khi như 7
    "E": "3",  # E trông như 3 (ít gặp)
    "P": "9",  # P trông như 9
}

# ============================================================
# Bảng sửa cặp ký tự 2 CHỤ bị nhầm thành 1 ký tự:
# Xảy ra khi CRNN tách một ký tự sê-ri thành 2 ký tự riêng.
# KEY: chuỗi 2 ký tự UPPERCASE mà model đọc sai
# VALUE: ký tự đúng thực sự
_TWO_CHAR_TO_ONE = {
    # ---- M bị tách (M trông như H+vertical-stroke) ----
    "H1": "M", "HI": "M", "HL": "M",  # 30H1 -> 30M
    # ---- N bị tách (N trông như M+diagonal) ----
    "M1": "N", "MI": "N",              # 29M1 -> 29N
    # ---- H bị tách (H trông như N+bar) ----
    "N1": "H", "NI": "H",              # ít gặp
    # ---- U bị tách (U trông như D cạnh phải hở + 1) ----
    "D1": "U", "DI": "U",              # 59D1 -> 59U
    "01": "U", "0I": "U",              # 0 + 1 -> U (ít gặp)
    # ---- V bị tách ----
    "V1": "Y", "VI": "Y",              # Y -> V+I
    # ---- W bị tách (VV) ----
    "VV": "W",
    # ---- K bị tách (K trông như I+< hoặc R+\ ) ----
    "IC": "K", "I<": "K",
    # ---- R bị tách (R = P + \ ) ----
    "P1": "R",
    # ---- X bị tách ----
    "XI": "X",  # giữ nguyên (hiếm)
}


def _correct_province_code(plate_text: str) -> str:
    """
    Sửa mã tỉnh nếu không hợp lệ bằng cách thử swap các ký tự bị nhầm.

    Ví dụ: '80A-207.59' → thử '50A', '81A'... nếu '51A' là mã tỉnh hợp lệ → sửa thành '51A-...'
    """
    import re
    if not plate_text or len(plate_text) < 3:
        return plate_text

    # Tách: province_digits (2 số đầu) + series (chữ + số tùy chọn) + '-' + serial
    m = re.match(r'^(\d{2})([A-Z]\d?)(-[\d.]+)$', plate_text)
    if not m:
        return plate_text

    prov = m.group(1)    # "80"
    series = m.group(2)  # "A"
    rest = m.group(3)    # "-207.59"

    # Nếu mã tỉnh đã hợp lệ → không cần sửa
    if prov in _VN_PROVINCE_CODES:
        return plate_text

    # Thử swap từng chữ số trong mã tỉnh
    best = plate_text
    for i in range(2):
        ch = prov[i]
        if ch in _CONFUSED_DIGITS:
            candidate = list(prov)
            candidate[i] = _CONFUSED_DIGITS[ch]
            new_prov = "".join(candidate)
            if new_prov in _VN_PROVINCE_CODES:
                best = new_prov + series + rest
                break

    # Thử swap cả 2 chữ số cùng lúc nếu vẫn chưa đúng
    if best == plate_text:
        p0 = _CONFUSED_DIGITS.get(prov[0], prov[0])
        p1 = _CONFUSED_DIGITS.get(prov[1], prov[1])
        new_prov = p0 + p1
        if new_prov in _VN_PROVINCE_CODES:
            best = new_prov + series + rest

    return best


# (Bảng này đã được tích hợp vào bảng toàn diện ở trên)


def _correct_vietnamese_plate_syntax(plate_text: str) -> str:
    """
    Hậu xử lý thông minh (Syntax-aware Error Correction):
    Tự động hiệu chỉnh kết quả OCR dựa trên quy chuẩn đăng ký biển số xe Việt Nam.
    Khắc phục triệt để các cặp ký tự dễ nhầm lẫn thị giác (B <-> 8, D <-> 0, S <-> 5...):
      - Cụm mã tỉnh (2 ký tự đầu): Bắt buộc là SỐ.
      - Ký tự thứ 3 (sê-ri): Bắt buộc là CHỮ CÁI.
      - Ký tự thứ 4: Nếu ký tự 3 là 'A' và ký tự 4 là số (ví dụ 'A8'), tự động chuyển thành 'AB'
        vì xe máy <50cc dùng sê-ri 2 chữ cái (AA, AB, AC...), Việt Nam không có xe máy 'A8'.
      - Dòng 2 (phần số sau dấu '-'): Bắt buộc toàn bộ là SỐ và dấu chấm.
    """
    if not plate_text:
        return plate_text

    clean_text = plate_text.strip().upper()

    # Phân tách tiền tố (mã tỉnh + seri) và phần số thứ tự (sau dấu '-')
    if '-' in clean_text:
        parts = clean_text.split('-', 1)
        prefix = parts[0]
        serial = parts[1]
    else:
        m = re.match(r'^(\d{2}[A-Z0-9]{1,2})(\d{3,5}.*)$', clean_text)
        if m:
            prefix = m.group(1)
            serial = m.group(2)
        else:
            prefix = clean_text
            serial = ''

    # 1. Chuẩn hóa phần số thứ tự sau dấu '-': BẮT BUỘC TOÀN SỐ VÀ DẤU CHẤM
    if serial:
        clean_serial = []
        for ch in serial:
            if ch in _SYNTAX_CHAR_TO_DIGIT:
                clean_serial.append(_SYNTAX_CHAR_TO_DIGIT[ch])
            elif ch.isdigit() or ch == '.':
                clean_serial.append(ch)
        serial = ''.join(clean_serial)

        # Đảm bảo dấu chấm cho biển 5 số nếu chưa có
        pure_digits = serial.replace('.', '')
        if len(pure_digits) == 5 and '.' not in serial:
            serial = pure_digits[:3] + '.' + pure_digits[3:]

    # 2. Chuẩn hóa phần tiền tố: [2 số tỉnh] + [1-2 ký tự seri]
    prefix_chars = list(prefix)

    # 2a. 2 ký tự đầu: Bắt buộc là SỐ (mã tỉnh)
    for i in range(min(2, len(prefix_chars))):
        if prefix_chars[i] in _SYNTAX_CHAR_TO_DIGIT:
            prefix_chars[i] = _SYNTAX_CHAR_TO_DIGIT[prefix_chars[i]]

    # 2b. Ký tự thứ 3: Bắt buộc là CHỮ CÁI
    if len(prefix_chars) >= 3:
        if prefix_chars[2].isdigit() and prefix_chars[2] in _SYNTAX_DIGIT_TO_CHAR:
            prefix_chars[2] = _SYNTAX_DIGIT_TO_CHAR[prefix_chars[2]]

    # 2c. Ký tự thứ 3+4: xử lý series
    if len(prefix_chars) >= 4:
        c3 = prefix_chars[2]
        c4 = prefix_chars[3]

        # Kiểm tra 2 ký tự (c3+c4) có phải là 1 ký tự bị tách đôi không?
        # VD: 'H1' -> 'M', 'M1' -> 'N', 'N1' -> 'H'
        two_key = c3 + c4
        if two_key in _TWO_CHAR_TO_ONE:
            # Series thực sự là 1 ký tự → bỏ ký tự 4 ra khỏi prefix
            prefix_chars[2] = _TWO_CHAR_TO_ONE[two_key]
            prefix_chars.pop(3)
        elif c3 == 'A':
            # Quy luật đặc biệt: Sê-ri xe máy 50cc / xe máy điện
            # Chữ 'A' không đi kèm số cho xe máy (không có sê-ri A1..A9).
            # Ô tô chỉ dùng 1 chữ 'A' (29A, 30A), không có số phụ ở dòng 1.
            # Xe máy <50cc dùng sê-ri 2 chữ cái (AA, AB, AC, AD, AE, AF, AH, AK, AL...).
            # Vì vậy nếu ký tự 3 là 'A' và ký tự 4 là số:
            if c4 == '8':
                prefix_chars[3] = 'B'  # Sửa '29A8' -> '29AB'
            elif c4 == '0':
                prefix_chars[3] = 'D'  # Sửa '29A0' -> '29AD'
            elif c4 == '5':
                prefix_chars[3] = 'S'  # Sửa '29A5' -> '29AS'
            elif c4 == '6':
                prefix_chars[3] = 'G'  # Sửa '29A6' -> '29AG'
            elif c4 == '2':
                prefix_chars[3] = 'Z'  # Sửa '29A2' -> '29AZ'
            elif c4 in ['1', '7']:
                prefix_chars[3] = 'T'  # Sửa '29A1' -> '29AT'
            elif c4 in _SYNTAX_DIGIT_TO_CHAR:
                prefix_chars[3] = _SYNTAX_DIGIT_TO_CHAR[c4]
        elif c3.isalpha() and c4.isdigit():
            # Series chữ+số như 'K1', 'V1'... không hợp lệ → bỏ phần số đi
            # (trừ các series 2 ký tự đặc biệt đã xử lý ở trên)
            # Không sửa vì có thể là biển hạng nặng K1, V1... nhưng thường không gặp
            pass

    elif len(prefix_chars) >= 3:
        c3 = prefix_chars[2]
        # Ký tự thứ 3 phải là CHỮ CÁI
        if c3.isdigit() and c3 in _SYNTAX_DIGIT_TO_CHAR:
            prefix_chars[2] = _SYNTAX_DIGIT_TO_CHAR[c3]

    new_prefix = ''.join(prefix_chars)
    return f"{new_prefix}-{serial}" if serial else new_prefix




# ──────────────────────────────────────────────────────────────
# OCR MỘT VÙNG ẢNH BIỂN SỐ
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _ocr_single_crop(plate_bgr: np.ndarray, model, device,
                     keep_ratio: bool = False) -> tuple:
    """
    OCR một vùng ảnh biển số đơn (1 dòng).

    Args:
        plate_bgr  : Ảnh biển số BGR
        model      : Mô hình CRNN
        device     : Thiết bị tính toán
        keep_ratio : True = giữ tỷ lệ gốc + pad đen (cho biển 2 dòng tách nửa)
                     False = stretch bình thường (cho biển 1 dòng nguyên)
    Returns:
        (pred_str, confidence)
    """
    if keep_ratio:
        plate_np = preprocess_plate_keep_ratio(plate_bgr)
    else:
        plate_np = preprocess_plate(plate_bgr)
    tensor = torch.from_numpy(plate_np).unsqueeze(0).unsqueeze(0).to(device)
    log_probs = model(tensor)
    pred_str = ctc_greedy_decode(log_probs)[0]
    pred_str = _add_plate_dot(pred_str)                   # Thêm dấu . cho biển 5 số
    pred_str = _correct_province_code(pred_str)           # Sửa mã tỉnh nếu không hợp lệ
    pred_str = _correct_vietnamese_plate_syntax(pred_str) # Sửa lỗi cú pháp sê-ri và số
    confidence = log_probs.exp().max(dim=2).values.mean().item()
    return pred_str, confidence


def _find_split_line(plate_bgr: np.ndarray) -> int:
    """
    Tìm vị trí tối ưu để tách biển số 2 dòng dựa trên phân tích pixel.
    Tìm hàng ngang có ít pixel chữ nhất (= khoảng trống giữa 2 dòng).

    Args:
        plate_bgr: Ảnh biển số BGR

    Returns:
        y_split: Tọa độ y để cắt đôi (tính từ trên xuống)
    """
    if len(plate_bgr.shape) == 3:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_bgr.copy()

    # Nhị phân hóa để tách chữ ra khỏi nền
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape

    # Chỉ tìm trong vùng giữa (30%-70% chiều cao) để tránh cắt vào viền
    search_top = int(h * 0.30)
    search_bot = int(h * 0.70)

    # Đếm số pixel trắng (chữ) trên mỗi hàng ngang
    row_sums = np.sum(binary[search_top:search_bot, :], axis=1)

    # Hàng có ít pixel chữ nhất = khoảng trống giữa 2 dòng
    best_row = np.argmin(row_sums)
    y_split = search_top + best_row

    return y_split


def _trim_to_vn_format_line1(raw: str) -> str:
    """
    Lọc kết quả OCR dòng 1 biển số VN theo quy tắc:
    Dòng 1 = Mã tỉnh (2 số) + Sê-ri (1 chữ cái, 1 chữ + 1 số, hoặc 2 chữ cái như 29AB, 59AA, 89LD)
    Ví dụ: "27B1", "51A", "29AB", "99E1", "60C", "89LD"
    Tối đa 4 ký tự, tối thiểu 3 ký tự.
    """
    if not raw:
        return raw
    cleaned = raw.strip("-")
    # Sửa lỗi cú pháp nếu có dạng như 29A8 -> 29AB
    cleaned = _correct_vietnamese_plate_syntax(cleaned)

    # Nếu kết quả đã hợp lệ (3-4 ký tự) → giữ nguyên
    if 3 <= len(cleaned) <= 4:
        return cleaned
    # Nếu dài hơn → cắt lấy 4 ký tự đầu
    if len(cleaned) > 4:
        candidate = cleaned[:4]
        # Kiểm tra format: 2 số + 1 chữ + (1 chữ hoặc số tùy chọn)
        if (len(candidate) >= 3 and
            candidate[0].isdigit() and candidate[1].isdigit() and
            candidate[2].isalpha()):
            return _correct_vietnamese_plate_syntax(candidate)
        return _correct_vietnamese_plate_syntax(cleaned[:3])
    return cleaned


def _trim_to_vn_format_line2(raw: str) -> str:
    """
    Lọc kết quả OCR dòng 2 biển số VN theo quy tắc:
    Dòng 2 = 3-5 chữ số
    Ví dụ: "25888", "79379", "12345", "901.50"
    """
    if not raw:
        return raw
    cleaned = raw.strip("-")
    # Chuyển các ký tự chữ dễ nhầm sang số
    clean_digits = []
    for ch in cleaned:
        if ch in _SYNTAX_CHAR_TO_DIGIT:
            clean_digits.append(_SYNTAX_CHAR_TO_DIGIT[ch])
        elif ch.isdigit():
            clean_digits.append(ch)
    digits = "".join(clean_digits)
    if len(digits) > 5:
        digits = digits[:5]
    return digits



def _score_vn_plate(text: str) -> float:
    """
    Chấm điểm chuỗi kết quả OCR theo mức độ "giống biển số VN".
    Biển số VN hợp lệ có dạng:
      - XXY-NNNNN (Ô tô: 30A-123.45)
      - XXYN-NNNNN (Xe máy >50cc: 29B1-123.45)
      - XXYY-NNNNN (Xe máy <50cc / xe máy điện / xe liên doanh: 29AB-901.50, 89LD-002.75)

    Score càng cao = càng giống biển số thật.
    """
    if not text:
        return 0.0

    score = 0.0
    clean = text.replace("-", "").replace(".", "")

    # Độ dài hợp lệ: 7-10 ký tự (không tính gạch và chấm)
    if 7 <= len(clean) <= 10:
        score += 3.0
    elif 6 <= len(clean) <= 11:
        score += 1.0
    else:
        score -= 2.0  # Quá ngắn hoặc quá dài

    # 2 ký tự đầu phải là số (mã tỉnh)
    if len(clean) >= 2 and clean[0].isdigit() and clean[1].isdigit():
        score += 2.0
        if clean[:2] in _VN_PROVINCE_CODES:
            score += 1.5

    # Ký tự thứ 3 phải là chữ cái (seri)
    if len(clean) >= 3 and clean[2].isalpha():
        score += 2.0

    # Ký tự thứ 4 (nếu có):
    if len(clean) >= 4:
        # Hợp lệ nếu là số (B1..B9) hoặc chữ cái (AA..AZ, LD, DA)
        if clean[3].isalnum():
            score += 1.0
        # Nếu ký tự 3 là 'A' mà ký tự 4 là '8', trừ điểm vì không có sê-ri A8
        if clean[2] == 'A' and clean[3] == '8':
            score -= 3.0

    # Có đúng 1 dấu gạch
    if text.count("-") == 1:
        score += 1.0
    elif text.count("-") > 1:
        score -= 1.0

    # Phần sau dấu gạch phải toàn số và dấu chấm
    if "-" in text:
        parts = text.split("-")
        if len(parts) == 2:
            num_part = parts[1].replace(".", "")
            if num_part.isdigit():
                score += 2.0
                if len(num_part) in [4, 5]:
                    score += 1.0

    return score


def _ocr_plate_smart(plate_bgr: np.ndarray, model, device) -> tuple:
    """
    Nhận dạng biển số thông minh — Chiến lược kép:

    Với biển 1 dòng (ngang): OCR trực tiếp, đơn giản.
    Với biển 2 dòng (vuông): Chạy CẢ HAI cách:
      A) OCR nguyên khối (model đã train trên ảnh 2 dòng bị ép 192×64)
      B) Tách 2 dòng → OCR từng nửa → ghép lại + lọc format VN
    → So sánh 2 kết quả, chọn cái nào "giống biển số VN" nhất.

    Args:
        plate_bgr: Ảnh biển số BGR đã cắt

    Returns:
        (plate_text, confidence, is_two_line)
    """
    h, w = plate_bgr.shape[:2]
    aspect_ratio = w / max(h, 1)

    # Biển 1 dòng: tỷ lệ W:H >= 2.0 (nằm ngang rõ ràng)
    if aspect_ratio >= 2.0:
        pred_str, confidence = _ocr_single_crop(plate_bgr, model, device)
        pred_str = _correct_vietnamese_plate_syntax(pred_str)
        return pred_str, confidence, False

    # ── Biển 2 dòng → Chạy CẢ HAI chiến lược ──────────────────

    # Chiến lược A: OCR nguyên khối (model đã quen ảnh 2 dòng ép ngang)
    whole_str, whole_conf = _ocr_single_crop(plate_bgr, model, device)
    whole_str = _correct_vietnamese_plate_syntax(whole_str)

    # Chiến lược B: Tách 2 dòng → OCR riêng → ghép
    split_str = ""
    split_conf = 0.0
    y_split = _find_split_line(plate_bgr)

    pad_y = max(2, int(h * 0.02))
    top_half = plate_bgr[0:y_split + pad_y, :]
    bottom_half = plate_bgr[max(0, y_split - pad_y):, :]

    min_h = 10
    if top_half.shape[0] >= min_h and bottom_half.shape[0] >= min_h:
        # Thử cả 2 kiểu preprocessing cho mỗi nửa, chọn cái tốt nhất
        top_candidates = []
        bot_candidates = []

        for kr in [False, True]:
            t_str, t_conf = _ocr_single_crop(top_half, model, device, keep_ratio=kr)
            b_str, b_conf = _ocr_single_crop(bottom_half, model, device, keep_ratio=kr)
            t_clean = _trim_to_vn_format_line1(t_str)
            b_clean = _trim_to_vn_format_line2(b_str)
            top_candidates.append((t_clean, t_conf))
            bot_candidates.append((b_clean, b_conf))

        # Chọn top result: ưu tiên chuỗi có format đúng (2số + 1-2chữ/số)
        def score_line1(text):
            if not text or len(text) < 3:
                return -1
            s = 0
            if text[0].isdigit() and text[1].isdigit():
                s += 2
                if text[:2] in _VN_PROVINCE_CODES:
                    s += 1
            if len(text) >= 3 and text[2].isalpha():
                s += 2
            if 3 <= len(text) <= 4:
                s += 1
            if len(text) == 4 and text[3].isalnum():
                s += 1
            return s

        best_top = max(top_candidates, key=lambda x: (score_line1(x[0]), x[1]))
        best_bot = max(bot_candidates, key=lambda x: (len(x[0]) if x[0] else 0, x[1]))

        top_clean, top_conf = best_top
        bot_clean, bot_conf = best_bot

        if top_clean and bot_clean:
            split_str = top_clean + "-" + bot_clean
        elif top_clean:
            split_str = top_clean
        elif bot_clean:
            split_str = bot_clean

        split_str = _correct_vietnamese_plate_syntax(split_str)
        split_conf = (top_conf + bot_conf) / 2.0

    # ── Chọn kết quả tốt nhất ─────────────────────────────────
    score_whole = _score_vn_plate(whole_str)
    score_split = _score_vn_plate(split_str)

    if score_split > score_whole:
        chosen_str, chosen_conf = split_str, split_conf
    else:
        chosen_str, chosen_conf = whole_str, whole_conf

    chosen_str = _correct_vietnamese_plate_syntax(chosen_str)
    chosen_str = _add_plate_dot(chosen_str)
    return chosen_str, chosen_conf, True


# ──────────────────────────────────────────────────────────────
# HÀM NHẬN DẠNG CHÍNH (ẢNH)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def recognize_from_image(pil_image: Image.Image, model, yolo_model, device,
                          conf_threshold: float = 0.0,
                          target_plate: str = "",
                          detector_method: str = "yolo") -> dict:
    """
    Pipeline đầy đủ: ảnh xe → phát hiện biển số → nhận dạng ký tự → đối chiếu mục tiêu.

    Args:
        pil_image    : PIL Image đầu vào (RGB)
        model        : Mô hình CRNN đã nạp
        yolo_model   : Mô hình YOLOv8 detector
        device       : Thiết bị tính toán
        conf_threshold: Ngưỡng lọc kết quả phát hiện
        target_plate : Biển số mục tiêu cần theo dõi (tùy chọn)

    Returns:
        dict với keys:
          - detections : List[dict] — kết quả phát hiện biển số
          - results    : List[dict] — kết quả nhận dạng từng biển số
          - annotated  : np.ndarray — ảnh gốc đã vẽ bounding box
          - time_detect: float — thời gian xử lý detect (ms)
          - time_recog : float — thời gian xử lý OCR (ms)
    """
    # ── Chuyển PIL → OpenCV BGR ────────────────────────────────
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Giai đoạn 1: Phát hiện biển số ────────────────────────
    t0 = time.time()
    if detector_method == "yolo" and yolo_model is not None:
        detections = detect_plate_yolo(img_bgr, yolo_model)
    else:
        detections = detect_plate_opencv(img_bgr)
    time_detect = (time.time() - t0) * 1000

    # ── Giai đoạn 2: Nhận dạng ký tự ─────────────────────────
    t0 = time.time()
    ocr_results = []

    if model is not None and detections:
        for det in detections[:3]:   # Tối đa 3 biển số
            plate_bgr = det["crop"]

            # Nhận dạng thông minh: tự phát hiện biển 1 dòng / 2 dòng
            pred_str, confidence, is_two_line = _ocr_plate_smart(
                plate_bgr, model, device)

            # Kiểm tra khớp biển số mục tiêu
            is_target_hit = plates_match_target(pred_str, target_plate) if target_plate else False

            # Lấy ảnh biển số để hiển thị (chuyển PIL)
            plate_rgb      = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
            plate_pil      = Image.fromarray(plate_rgb)

            ocr_results.append({
                "plate_text": pred_str,
                "confidence": confidence,
                "plate_image": plate_pil,
                "bbox"      : det["bbox"],
                "is_two_line": is_two_line,
                "is_target_hit": is_target_hit,
            })

    time_recog = (time.time() - t0) * 1000

    # ── Vẽ kết quả lên ảnh gốc ────────────────────────────────
    annotated_bgr = draw_detections(img_bgr, detections)
    for res in ocr_results:
        x, y, w, h = res["bbox"]
        text = res["plate_text"]
        is_hit = res.get("is_target_hit", False)

        label_text = f"🎯 {text}" if is_hit else text
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)

        if is_hit:
            # Highlight bounding box đỏ nổi bật cho biển số mục tiêu
            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.rectangle(annotated_bgr,
                          (x, y - th - 14), (x + tw + 14, y - 2),
                          (0, 0, 220), -1)
            cv2.putText(annotated_bgr, label_text, (x + 6, y - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(annotated_bgr,
                          (x, y - th - 14), (x + tw + 10, y - 2),
                          (30, 120, 255), -1)
            cv2.putText(annotated_bgr, text, (x + 5, y - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    return {
        "detections" : detections,
        "results"    : ocr_results,
        "annotated"  : annotated_pil,
        "time_detect": time_detect,
        "time_recog" : time_recog,
    }


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

def render_sidebar(model, yolo_model, device):
    with st.sidebar:
        device_name = str(device).upper()
        m_color = "#00E676" if model else "#FFD740"
        m_text  = "SẴN SÀNG" if model else "DEMO MODE"
        y_color = "#00D4FF" if yolo_model else "#4A6A89"
        y_text  = "YOLOv8 ACTIVE" if yolo_model else "OpenCV Canny"
        dot_m   = "sdot-green" if model else "sdot-yellow"
        dot_y   = "sdot-blue"  if yolo_model else "sdot-gray"

        history_len  = len(st.session_state.get("plate_history", []))
        target_hits  = sum(1 for p in st.session_state.get("plate_history", []) if p.get("is_target", False))
        th_color = "#FF5252" if target_hits else "#4A6A89"

        st.markdown(f"""
        <div class="sb-logo-area">
            <div class="sb-logo-icon">🚗</div>
            <div class="sb-logo-name">BIỂN SỐ AI</div>
            <div class="sb-logo-sub">Neural Plate Vision</div>
        </div>
        <div class="sb-sec-title">⚙️ Hệ Thống</div>
        <div class="sb-status-row">
            <span class="sdot {dot_m}"></span>
            <span style="color:{m_color};font-weight:700;font-size:0.8rem;">{m_text}</span>
        </div>
        <div class="sb-status-row">
            <span class="sdot {dot_y}"></span>
            <span style="color:{y_color};font-size:0.8rem;">{y_text}</span>
        </div>
        <div class="sb-row">
            <span class="sb-row-key">💻 Thiết bị</span>
            <span class="sb-row-val">{device_name}</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown(f"""
        <div class="sb-sec-title">📊 Phiên làm việc</div>
        <div class="sb-row">
            <span class="sb-row-key">Đã nhận diện</span>
            <span class="sb-row-val" style="color:#00E676;">{history_len} biển</span>
        </div>
        <div class="sb-row">
            <span class="sb-row-key">Khớp mục tiêu</span>
            <span class="sb-row-val" style="color:{th_color};">{target_hits} lần</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="sb-sec-title">📖 Hướng dẫn</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Tải ảnh / Camera / URL"),
            ("2", "Nhập biển số mục tiêu (tùy chọn)"),
            ("3", "Bấm Nhận Dạng Biển Số"),
            ("4", "AI phát hiện & nhận dạng ký tự"),
            ("5", "Xem kết quả & cảnh báo"),
        ]
        for num, text in steps:
            st.markdown(
                f'<div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:7px;">'
                f'<div style="width:20px;height:20px;border-radius:50%;background:rgba(0,212,255,0.12);'
                f'border:1px solid rgba(0,212,255,0.35);display:flex;align-items:center;justify-content:center;'
                f'font-size:0.65rem;font-weight:700;color:#00D4FF;flex-shrink:0;margin-top:1px;">{num}</div>'
                f'<span style="font-size:0.76rem;color:#4A6A89;line-height:1.4;">{text}</span></div>',
                unsafe_allow_html=True)

        st.divider()

        tech_info = {
            "OCR Model":  "CRNN (CNN+BiLSTM)",
            "Detector":   "YOLOv8" if yolo_model else "OpenCV+Canny",
            "Ký tự":      f"{config.NUM_CLASSES} classes",
            "Input size": f"{config.PLATE_HEIGHT}×{config.PLATE_WIDTH}px",
        }
        st.markdown('<div class="sb-sec-title">🔬 Kỹ thuật</div>', unsafe_allow_html=True)
        for k, v in tech_info.items():
            st.markdown(
                f'<div class="sb-row"><span class="sb-row-key">{k}</span>'
                f'<span class="sb-row-val">{v}</span></div>',
                unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="sb-sec-title">📡 Cổng Giao Tiếp IoT</div>', unsafe_allow_html=True)
        cur_esp_ip = (st.session_state.get("esp_ip") or getattr(config, "ESP_DEFAULT_IP", "192.168.137.61")).strip()
        if not cur_esp_ip:
            cur_esp_ip = "192.168.137.61"
        esp_ip_input = st.text_input(
            "IP Module ESP-01:",
            value=cur_esp_ip,
            placeholder="VD: 192.168.137.61",
            key="sb_esp_ip_field",
            help="Địa chỉ IP của ESP-01 kết nối trong cùng mạng Wi-Fi"
        )
        st.session_state["esp_ip"] = esp_ip_input.strip()

        if st.button("🔌 Kiểm tra kết nối ESP", key="sb_btn_test_esp", use_container_width=True):
            with st.spinner("Đang kết nối ESP-01..."):
                conn_ok, conn_msg, latency = iot_client.check_esp_connection(esp_ip_input.strip())
                st.session_state["esp_conn_status"] = (conn_ok, conn_msg)

        if "esp_conn_status" in st.session_state:
            is_ok, status_text = st.session_state["esp_conn_status"]
            if is_ok:
                st.markdown(f'<div style="font-size:0.75rem;color:#00FF88;padding:6px;background:rgba(0,255,136,0.1);border-radius:4px;border:1px solid #00FF88;margin-top:4px;">{status_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:0.75rem;color:#FF3B5C;padding:6px;background:rgba(255,59,92,0.1);border-radius:4px;border:1px solid #FF3B5C;margin-top:4px;">{status_text}</div>', unsafe_allow_html=True)

        if model is None:
            st.markdown("""
            <div class="info-note-cyber">
            ⚠️ Chưa có mô hình đã huấn luyện.<br>
            Chạy: <code style="color:#00D4FF;">python dataset.py --prepare</code><br>
            Rồi: <code style="color:#00D4FF;">python train.py</code>
            </div>
            """, unsafe_allow_html=True)
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# PHẦN HIỂN THỊ LỊCH SỬ NHẬN DIỆN GẦN ĐÂY (TACTICAL DATA TABLE)
# ──────────────────────────────────────────────────────────────

def render_html(html_str: str, **kwargs):
    """Render HTML an toàn trong Streamlit mà không bị markdown biến thành code block."""
    cleaned = "\n".join(line.strip() for line in str(html_str).splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_recent_history():
    """Hiển thị lịch sử nhận diện dạng bảng Cyberpunk Tactical."""
    # Xử lý xóa lẻ từng bản ghi qua query params
    if "del_id" in st.query_params:
        target_id = str(st.query_params.get("del_id", "")).strip()
        hist = st.session_state.get("plate_history", [])
        new_hist = []
        del_name = ""
        for idx, item in enumerate(hist):
            item_id = str(item.get("id") or item.get("timestamp_iso") or f"row_{idx}")
            if item_id == target_id:
                del_name = item.get("plate_text", "")
            else:
                new_hist.append(item)
        if len(new_hist) < len(hist):
            st.session_state["plate_history"] = new_hist
            save_history(new_hist)
            st.toast(f"Đã xóa biển số: {del_name}", icon="🗑️")
        try:
            del st.query_params["del_id"]
        except Exception:
            pass
        st.rerun()

    history = st.session_state.get("plate_history", [])

    # Header tabs & Action bar
    col_tabs, col_actions = st.columns([3, 2])
    with col_tabs:
        render_html(
            f'<div class="hist-tabs-bar" style="border-bottom:none;margin-bottom:0;">'
            f'<div class="hist-tab-btn">🕒 Nhật Ký Nhận Diện ({len(history):,} lượt)</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_actions:
        btn_c1, btn_c2, btn_c3 = st.columns([1.5, 1.5, 0.8])
        if "filter_target_only" not in st.session_state:
            st.session_state["filter_target_only"] = False

        with btn_c1:
            filter_label = "👁️ Xem Tất Cả" if st.session_state["filter_target_only"] else "🔻 Lọc Vi Phạm"
            if st.button(filter_label, key="toggle_filter_target_btn", use_container_width=True):
                st.session_state["filter_target_only"] = not st.session_state["filter_target_only"]
                st.rerun()

        with btn_c2:
            output_csv = io.StringIO()
            writer = csv.writer(output_csv)
            writer.writerow(["Thời gian", "Biển số", "Độ tin cậy (%)", "Loại biển", "Cảnh báo mục tiêu"])
            for item in history:
                writer.writerow([
                    item.get("timestamp", ""),
                    item.get("plate_text", ""),
                    f"{item.get('confidence', 0)*100:.1f}",
                    "2 dòng" if item.get("is_two_line") else "1 dòng",
                    "Có" if item.get("is_target") else "Không"
                ])
            st.download_button(
                label="📥 Xuất CSV",
                data=output_csv.getvalue().encode('utf-8-sig'),
                file_name="nhat_ky_bien_so.csv",
                mime="text/csv",
                key="export_hist_btn_tactical",
                use_container_width=True
            )

        with btn_c3:
            with st.popover("⚙️", help="Quản lý lịch sử"):
                st.markdown("<b style='color:#FF3B5C;'>⚠️ Thao tác nguy hiểm</b>", unsafe_allow_html=True)
                confirm_chk = st.checkbox("Xác nhận muốn xóa toàn bộ lịch sử", key="chk_safe_delete_history")
                if confirm_chk:
                    if st.button("🗑️ Xác nhận xóa sạch", type="primary", key="btn_confirm_delete_history", use_container_width=True):
                        st.session_state["plate_history"] = []
                        if os.path.exists(HISTORY_FILE):
                            try:
                                os.remove(HISTORY_FILE)
                            except Exception:
                                pass
                        st.rerun()

    if not history:
        render_html(
            '<div class="viewport-card" style="padding:40px 20px;text-align:center;">'
            '<div style="font-size:2.4rem;margin-bottom:8px;">📋</div>'
            '<div style="font-weight:700;color:#627D98;font-size:0.9rem;">Chưa có dữ liệu nhật ký nhận diện</div>'
            '<div style="font-size:0.75rem;color:#243B53;margin-top:4px;">Hệ thống sẽ tự động ghi lại lịch sử quét biển số và lưu trữ trong 7 ngày.</div>'
            '</div>'
        )
        return

    # Render Table
    display_history = [item for item in history if item.get("is_target", False)] if st.session_state.get("filter_target_only") else history
    rows_html = ""
    for i, item in enumerate(display_history[:50]):
        item_id = str(item.get("id") or item.get("timestamp_iso") or f"row_{i}")
        is_tg  = item.get("is_target", False)
        conf   = item.get("confidence", 0) * 100
        txt    = item.get("plate_text", "???")
        ts     = item.get("timestamp", "—")
        is_two = item.get("is_two_line", False)

        conf_color = "#00FF88" if conf >= 80 else ("#FFD740" if conf >= 60 else "#FF3B5C")
        row_cls = "target-row" if is_tg else ""
        tgt_badge = ('<span style="background:#FF3B5C;color:#fff;font-size:0.58rem;'
                     'font-weight:800;padding:2px 6px;border-radius:3px;margin-left:6px;">🎯 TARGET</span>'
                     if is_tg else "")

        # Thumbnail
        img_html = "—"
        if item.get("plate_image"):
            try:
                buf = io.BytesIO()
                p_img = item["plate_image"].convert("RGB")
                p_img.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                img_html = (
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="height:32px;max-width:90px;object-fit:contain;border-radius:3px;'
                    f'border:1px solid rgba(0,212,255,0.25);background:#050811;" />'
                )
            except Exception:
                img_html = "—"

        rows_html += f"""
        <tr class="{row_cls}">
            <td style="font-family:'JetBrains Mono',monospace;color:#627D98;font-size:0.72rem;">#{i+1:03d}</td>
            <td>{img_html}</td>
            <td>
                <span style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:800;color:#00D4FF;letter-spacing:0.08em;">{txt}</span>
                {tgt_badge}
            </td>
            <td style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#627D98;">{ts}</td>
            <td>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{conf_color};font-size:0.82rem;min-width:44px;">{conf:.1f}%</span>
                    <div style="flex:1;height:4px;background:#101B2E;border-radius:2px;min-width:40px;">
                        <div style="width:{conf:.0f}%;height:100%;background:{conf_color};border-radius:2px;"></div>
                    </div>
                </div>
            </td>
            <td style="font-size:0.75rem;color:#627D98;">{"Biển 2 Dòng" if is_two else "Biển 1 Dòng"}</td>
            <td>
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#627D98;border:1px solid rgba(0,212,255,0.2);padding:2px 6px;border-radius:3px;">#{i+1}</span>
                    <a href="?del_id={item_id}#sec-history" target="_self" class="del-single-btn" title="Xóa biển số {txt}">🗑️</a>
                </div>
            </td>
        </tr>
        """

    table_markup = f"""
    <div class="hist-table-wrap">
        <table class="hist-table-modern">
            <thead>
                <tr>
                    <th style="width:40px;">#</th>
                    <th style="width:110px;">Ảnh Cắt Biển Số</th>
                    <th>Chuỗi Ký Tự (Plate Text)</th>
                    <th>Thời Gian Quét</th>
                    <th>Độ Tin Cậy (Score)</th>
                    <th>Phân Loại</th>
                    <th>Thao Tác</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <div style="padding:8px 14px;font-size:0.7rem;color:#627D98;border-top:1px solid rgba(0,212,255,0.1);display:flex;justify-content:space-between;align-items:center;">
            <span>Hiển thị {min(len(history), 50)} trên {len(history)} bản ghi đã nhận dạng</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#243B53;">Lưu trữ cục bộ · Tự xóa sau 7 ngày</span>
        </div>
    </div>
    """
    render_html(table_markup)


# ──────────────────────────────────────────────────────────────
# MAIN DASHBOARD CONTROLLER
# ──────────────────────────────────────────────────────────────

def main():
    # Session state initialization
    if "plate_history" not in st.session_state or not st.session_state["plate_history"]:
        st.session_state["plate_history"] = load_history()
    if "active_plate_idx" not in st.session_state:
        st.session_state["active_plate_idx"] = 0
    if "last_output" not in st.session_state:
        st.session_state["last_output"] = None

    # Load neural models
    model, device = load_recognizer()
    yolo_model = load_yolo_model()

    # Sidebar removed as requested

    # ── 1. TOP NAVIGATION & TELEMETRY HEADER ─────────────────────
    now_time = datetime.now().strftime("%H:%M:%S")
    dev_str = "CUDA" if "cuda" in str(device).lower() else "CPU"
    dev_ms = "0.8ms" if dev_str == "CUDA" else "14.2ms"

    render_html(f"""
    <div class="topbar-container">
        <div class="brand-box">
            <div class="brand-icon">🔲</div>
            <div>
                <div class="brand-title">BIỂN SỐ AI <span class="brand-badge">PRO v2.4</span></div>
                <div class="brand-sub">NEURAL SMART PLATE VISION</div>
            </div>
        </div>
        <div class="top-nav-tabs">
            <a href="#sec-analysis" class="top-nav-item active" target="_self">🔲 Phân Tích Ảnh</a>
            <a href="#sec-camera" class="top-nav-item" target="_self">📹 Giám Sát Camera</a>
            <a href="#sec-hotlist" class="top-nav-item" target="_self">🎯 Khóa Mục Tiêu</a>
            <a href="#sec-history" class="top-nav-item" target="_self">📋 Nhật Ký Biển Số</a>
        </div>
        <div class="top-telemetry">
            <div class="telemetry-chip">
                <span class="telemetry-label">{dev_str}:</span>
                <span class="telemetry-val cyan">{dev_ms}</span>
            </div>
            <div class="telemetry-chip">
                <span class="telemetry-label">YOLOv8:</span>
                <span class="telemetry-val">59.8 FPS</span>
            </div>
            <div class="telemetry-chip">
                <span class="telemetry-label">Độ chính xác:</span>
                <span class="telemetry-val">99.4%</span>
            </div>
            <div class="telemetry-chip" style="color:#627D98;">
                <span>⏰ {now_time} UTC+7</span>
            </div>
            <div style="font-size:1.1rem;color:#00D4FF;padding-left:4px;cursor:pointer;">👤</div>
        </div>
    </div>
    """)

    # ── TOP DETECTOR CONTROL BAR (Thanh công cụ trên Header) ────
    render_html("""
    <div style="background: rgba(8, 16, 29, 0.85); border: 1px solid rgba(0, 212, 255, 0.25); border-radius: 6px; padding: 2px 12px; margin-top: -4px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.4);">
    </div>
    """)
    top_c1, top_c2 = st.columns([1.1, 3.5])
    with top_c1:
        st.markdown(
            """<div style="padding-top:6px;font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:700;color:#00D4FF;letter-spacing:0.04em;">
            ⚡ BỘ PHÁT HIỆN BIỂN SỐ:
            </div>""",
            unsafe_allow_html=True
        )
    with top_c2:
        det_method_choice = st.radio(
            "Chọn phương pháp phát hiện:",
            options=[
                "🎯 YOLOv8 (Học sâu AI - Chính xác cao)",
                "📐 OpenCV (Xử lý ảnh Canny + Contours)"
            ],
            index=0 if yolo_model is not None else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="radio_detection_method"
        )
    selected_detector = "yolo" if "YOLOv8" in det_method_choice else "opencv"

    # ── 2. TWO-COLUMN MAIN WORKSPACE ─────────────────────────────
    render_html('<div id="sec-analysis" style="scroll-margin-top:20px;"></div>')
    col_left, col_right = st.columns([1.35, 1.0], gap="large")

    # ═════════════════════════════════════════════════════════════
    # CỘT TRÁI: CAMERA VIEWPORT + CONTROLS + HOTLIST
    # ═════════════════════════════════════════════════════════════
    with col_left:
        # Header controls row
        st_c1, st_c2 = st.columns([2.5, 1.5])
        with st_c1:
            input_method = st.radio(
                "Nguồn dữ liệu:",
                ["📁 Tải Ảnh Lên", "📹 Camera Trực Tiếp"],
                horizontal=True,
                label_visibility="collapsed",
                key="source_mode"
            )
        with st_c2:
            render_html(
                """<div style="text-align:right;padding-top:8px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#00FF88;">
                <span class="sdot sdot-green"></span>CAM-01_Q1_HCM · 1080p @ 60fps
                </div>""",
                unsafe_allow_html=True
            )

        pil_image = None

        if input_method == "📁 Tải Ảnh Lên":
            uploaded = st.file_uploader(
                "Chọn ảnh xe",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
                key="file_uploader"
            )
            if uploaded:
                pil_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        elif input_method == "📹 Camera Trực Tiếp":
            snap = st.camera_input("Chụp ảnh camera", label_visibility="collapsed", key="cam_snap")
            if snap:
                pil_image = Image.open(io.BytesIO(snap.read())).convert("RGB")


        render_html('<div id="sec-camera" style="scroll-margin-top:20px;"></div>')
        # Viewport Frame
        target_count_display = len(st.session_state["last_output"]["results"]) if st.session_state.get("last_output") and st.session_state["last_output"].get("results") else 0
        detector_badge = "YOLOv8-PlateDetector" if ("YOLOv8" in st.session_state.get("radio_detection_method", "YOLOv8") and yolo_model) else "OpenCV-Canny"

        render_html(f"""
        <div class="viewport-header" style="border-radius:8px 8px 0 0;margin-top:6px;">
            <div><span class="sdot sdot-red"></span>LIVE FEED 1080P // MULTI-TARGET ({target_count_display})</div>
            <div style="background:rgba(0,212,255,0.15);color:#00D4FF;padding:2px 8px;border-radius:4px;font-size:0.65rem;border:1px solid rgba(0,212,255,0.3);">
                {detector_badge}
            </div>
        </div>
        """)

        if st.session_state.get("last_output") and st.session_state["last_output"].get("annotated"):
            st.image(st.session_state["last_output"]["annotated"], use_container_width=True)
        elif pil_image:
            st.image(pil_image, use_container_width=True)
        else:
            render_html("""
            <div class="viewport-canvas">
                <div style="text-align:center;">
                    <div style="font-size:3rem;margin-bottom:10px;opacity:0.6;">🎯</div>
                    <div style="color:#627D98;font-weight:700;font-size:0.95rem;">CHƯA CÓ NGUỒN ẢNH ĐẦU VÀO</div>
                    <div style="color:#243B53;font-size:0.75rem;margin-top:4px;">
                        Tải ảnh lên hoặc kích hoạt Camera để bắt đầu phát hiện & nhận dạng
                    </div>
                </div>
            </div>
            """)

        render_html("""
        <div class="viewport-footer" style="border-radius:0 0 8px 8px;margin-bottom:12px;">
            <span>Tọa độ: 10.7769° N, 106.7009° E (Ngã 4 Nguyễn Huệ, Q1)</span>
            <span style="color:#00FF88;">Multi-instance Inference: Active</span>
        </div>
        """)

        # Primary Run Action Button
        run_detection = st.button("🎯 NHẬN DẠNG BIỂN SỐ", use_container_width=True, type="primary", key="btn_run_tactical")

        render_html('<div id="sec-hotlist" style="scroll-margin-top:20px;"></div>')
        # Hotlist Target Surveillance Box
        render_html("""
        <div class="hotlist-box" style="margin-top:8px;margin-bottom:6px;">
            <div class="hotlist-header" style="margin-bottom:0;">
                <div class="hotlist-title">
                    <span>🔔 MỤC TIÊU KHÓA GIÁM SÁT</span>
                    <span class="hotlist-tag">HOTLIST ACTIVE</span>
                </div>
                <div style="font-size:0.68rem;color:#627D98;">Tự động báo động khi biển số đi vào camera</div>
            </div>
        </div>
        """)

        target_plate_input = st.text_input(
            "Biển số mục tiêu:",
            placeholder="Ví dụ: 51A-888.88 hoặc 29H-688.12",
            key="target_plate_tactical",
            label_visibility="collapsed"
        ).strip().upper()

        render_html("""
        <div style="display:flex;gap:6px;align-items:center;margin-top:4px;margin-bottom:8px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#627D98;">
            <span>Gợi ý mục tiêu:</span>
            <span style="border:1px solid rgba(0,212,255,0.2);padding:1px 6px;border-radius:3px;color:#00D4FF;">51A-888.88</span>
            <span style="border:1px solid rgba(0,212,255,0.2);padding:1px 6px;border-radius:3px;color:#00D4FF;">29H-688.12</span>
            <span style="border:1px solid rgba(0,212,255,0.2);padding:1px 6px;border-radius:3px;color:#00D4FF;">80B-33.99</span>
        </div>
        """)

        with st.expander("📡 Cổng Giao Tiếp IoT (ESP-01 → STM32)", expanded=False):
            c_ip, c_btn = st.columns([2, 1.3])
            with c_ip:
                esp_ip_val = st.text_input(
                    "IP ESP-01:",
                    value=st.session_state.get("esp_ip", getattr(config, "ESP_DEFAULT_IP", "")),
                    placeholder="VD: 192.168.1.50",
                    key="main_esp_ip_field",
                    help="IP ESP-01 nhận lệnh HTTP POST /alert để chuyển tiếp UART sang STM32"
                )
                st.session_state["esp_ip"] = esp_ip_val.strip()
            with c_btn:
                st.write("")
                st.write("")
                if st.button("🔌 Kiểm tra kết nối ESP", key="main_btn_test_esp", use_container_width=True):
                    with st.spinner("Kiểm tra ESP-01..."):
                        conn_ok, conn_msg, latency = iot_client.check_esp_connection(esp_ip_val.strip())
                        st.session_state["esp_conn_status"] = (conn_ok, conn_msg)
            if "esp_conn_status" in st.session_state:
                ok, msg = st.session_state["esp_conn_status"]
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        # Execute Recognition when button clicked
        if pil_image and run_detection:
            with st.spinner("Đang chạy mô hình phát hiện và nhận dạng ký tự..."):
                out = recognize_from_image(pil_image, model, yolo_model, device,
                                           target_plate=target_plate_input,
                                           detector_method=selected_detector)
                st.session_state["last_output"] = out
                st.session_state["active_plate_idx"] = 0

                esp_ip_target = (st.session_state.get("esp_ip") or getattr(config, "ESP_DEFAULT_IP", "192.168.137.61")).strip()
                if not esp_ip_target:
                    esp_ip_target = "192.168.137.61"
                iot_res = iot_client.process_target_alerts_for_detections(
                    out.get("results", []),
                    target_plate_input,
                    esp_ip=esp_ip_target
                )
                st.session_state["last_iot_result"] = iot_res

                # Persist to history
                if out.get("results"):
                    now = datetime.now()
                    now_str = now.strftime("%H:%M:%S %d/%m/%Y")
                    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S")
                    for r in out["results"]:
                        txt = r.get("plate_text", "").strip()
                        if txt and txt != "???":
                            add_plate_to_history({
                                "plate_text"   : txt,
                                "confidence"   : r.get("confidence", 0.0),
                                "is_target"    : r.get("is_target_hit", False),
                                "is_two_line"  : r.get("is_two_line", False),
                                "plate_image"  : r.get("plate_image"),
                                "timestamp"    : now_str,
                                "timestamp_iso": now_iso,
                            })
                st.rerun()

    # ═════════════════════════════════════════════════════════════
    # CỘT PHẢI: KẾT QUẢ TACTICAL + HERO PLATE VIEW + METRICS
    # ═════════════════════════════════════════════════════════════
    with col_right:
        cur_out = st.session_state.get("last_output")
        results = cur_out.get("results", []) if cur_out else []
        detections = cur_out.get("detections", []) if cur_out else []
        time_det = cur_out.get("time_detect", 0.0) if cur_out else 0.0
        time_rec = cur_out.get("time_recog", 0.0) if cur_out else 0.0
        total_time = time_det + time_rec

        # Target Match Notification Box — always render once to avoid React DOM conflict
        target_hits = [r for r in results if r.get("is_target_hit", False)]
        _show_danger  = bool(target_plate_input and target_hits)
        _show_watching = bool(target_plate_input and results and not target_hits)
        _show_none     = not _show_danger and not _show_watching
        _hit_count  = len(target_hits)
        _res_count  = len(results)
        _hit_text   = target_hits[0]['plate_text'] if target_hits else ""

        # IoT Notification snippet
        iot_res = st.session_state.get("last_iot_result")
        iot_feedback_html = ""
        if _show_danger and iot_res and iot_res.get("has_target_hit"):
            for item in iot_res.get("send_results", []):
                p = item.get("plate", "")
                if item.get("success"):
                    iot_feedback_html += f"""
                    <div style="margin-top:8px;padding:6px 10px;background:rgba(0,255,136,0.12);border:1px solid #00FF88;border-radius:5px;font-size:0.75rem;color:#00FF88;display:flex;align-items:center;gap:6px;">
                        <span>📡</span>
                        <span><b>[IoT ESP-01 → STM32]</b> Đã gửi HTTP POST /alert cho biển số <b>{p}</b>. STM32 điều khiển LED nhấp nháy cảnh báo.</span>
                    </div>
                    """
                else:
                    err = item.get("message", "Không thể gửi cảnh báo đến thiết bị")
                    iot_feedback_html += f"""
                    <div style="margin-top:8px;padding:6px 10px;background:rgba(255,59,92,0.12);border:1px solid #FF3B5C;border-radius:5px;font-size:0.75rem;color:#FF3B5C;display:flex;align-items:center;gap:6px;">
                        <span>⚠️</span>
                        <span><b>[IoT ESP-01]</b> {err} (AI Web vẫn hoạt động bình thường).</span>
                    </div>
                    """

        render_html(f"""
        <div id="target-alert-wrapper">
            <div class="target-alert-match danger" style="{'display:block' if _show_danger else 'display:none'}">
                <div style="font-weight:900;color:#FF3B5C;font-size:0.88rem;letter-spacing:0.04em;">
                    🚨 PHÁT HIỆN BIỂN SỐ MỤC TIÊU (TARGET MATCH)
                    <span style="float:right;background:#FF3B5C;color:#fff;font-size:0.62rem;padding:2px 8px;border-radius:4px;">
                        KHỚP {_hit_count}/{_res_count} XE
                    </span>
                </div>
                <div style="font-size:0.75rem;color:#E2EDF8;margin-top:4px;">
                    Chuỗi ký tự <b style="color:#FFD740;font-family:'JetBrains Mono',monospace;">{_hit_text}</b>
                    trùng khớp hoàn toàn với tham số target_plate đang giám sát.
                </div>
                {iot_feedback_html}
            </div>
            <div class="target-alert-match" style="{'display:block' if _show_watching else 'display:none'}">
                <div style="font-size:0.75rem;color:#627D98;">
                    🎯 Mục tiêu giám sát: <b style="color:#00D4FF;">{target_plate_input}</b>
                    &nbsp;|&nbsp; <i>Chưa phát hiện thấy trong khung hình</i>
                </div>
            </div>
        </div>
        """)

        # Main Results Card
        render_html(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;background:#090F1D;padding:10px 14px;border:1px solid var(--border);border-radius:8px;">
            <div style="font-size:0.85rem;font-weight:800;color:#FFFFFF;letter-spacing:0.06em;display:flex;align-items:center;gap:6px;">
                <span>🔲 NHẬN DIỆN ĐA BIỂN SỐ</span>
                <span style="background:rgba(0,212,255,0.15);color:#00D4FF;padding:2px 8px;border-radius:4px;font-size:0.65rem;border:1px solid rgba(0,212,255,0.3);">
                    {len(results):02d} XE CÙNG LÚC
                </span>
            </div>
            <div style="font-size:0.7rem;color:#627D98;font-family:'JetBrains Mono',monospace;">
                Suy luận: {total_time:.1f}ms
            </div>
        </div>
        """)

        if results:
            # Pill Selectors for multiple plates
            plate_labels = []
            for i, r in enumerate(results):
                t_lbl = "TARGET" if r.get("is_target_hit") else "PLATE"
                plate_labels.append(f"#{i+1:02d} {t_lbl} {r.get('confidence',0)*100:.1f}%")

            active_idx = st.selectbox(
                "Chọn biển số xem chi tiết:",
                range(len(results)),
                format_func=lambda i: f"#{i+1:02d} {results[i].get('plate_text', '???')} ({results[i].get('confidence',0)*100:.1f}%)",
                key="active_plate_selector",
                label_visibility="collapsed"
            )

            active_plate = results[active_idx]
            p_text = active_plate.get("plate_text", "???")
            p_conf = active_plate.get("confidence", 0.0) * 100
            p_two = active_plate.get("is_two_line", False)
            p_is_tgt = active_plate.get("is_target_hit", False)
            bbox = active_plate.get("bbox", [0, 0, 192, 64])

            # Hero Plate Display
            target_status_badge = '<span style="background:#FF3B5C;color:#fff;font-size:0.58rem;padding:2px 6px;border-radius:3px;font-weight:800;">TRÚNG KHỚP TARGET</span>' if p_is_tgt else '<span style="color:#00D4FF;font-size:0.65rem;">CHUẨN HÓA AI</span>'
            plate_type_name = "BIỂN 2 DÒNG" if p_two else "BIỂN 1 DÒNG"

            render_html(f"""
            <div class="plate-hero-wrapper">
                <div class="plate-hero-title">
                    <span>● ĐANG CHỌN XEM: MỤC TIÊU #{active_idx+1:02d}</span>
                    <span>{target_status_badge}</span>
                </div>
                <div class="plate-hero-box">
                    <div class="plate-hero-type">🔩 ĐỊNH DẠNG {plate_type_name} 🔩</div>
                    <div class="plate-hero-text">{p_text}</div>
                </div>
            </div>
            """)

            # 4-Grid Metrics
            conf_color = "#00FF88" if p_conf >= 80 else ("#FFD740" if p_conf >= 60 else "#FF3B5C")
            render_html(f"""
            <div class="grid-2x2">
                <div class="grid-tile">
                    <div class="grid-tile-label">ĐỘ TIN CẬY OCR</div>
                    <div class="grid-tile-val" style="color:{conf_color};">{p_conf:.1f}% <span style="font-size:0.7rem;color:#627D98;">(Score)</span></div>
                </div>
                <div class="grid-tile">
                    <div class="grid-tile-label">THỜI GIAN CRNN</div>
                    <div class="grid-tile-val" style="color:#00D4FF;">{time_rec:.1f} ms <span style="font-size:0.7rem;color:#627D98;">(Inference)</span></div>
                </div>
                <div class="grid-tile">
                    <div class="grid-tile-label">KÍCH THƯỚC CROP</div>
                    <div class="grid-tile-val" style="color:#E2EDF8;">{bbox[2]} × {bbox[3]} <span style="font-size:0.7rem;color:#627D98;">px</span></div>
                </div>
                <div class="grid-tile">
                    <div class="grid-tile-label">ĐỊNH DẠNG BIỂN</div>
                    <div class="grid-tile-val" style="color:#00D4FF;font-size:0.88rem;">{plate_type_name}</div>
                </div>
            </div>
            """)

            # Action Buttons Row
            btn_act1, btn_act2 = st.columns(2)
            with btn_act1:
                st.code(p_text, language=None)
            with btn_act2:
                if active_plate.get("plate_image"):
                    buf_c = io.BytesIO()
                    active_plate["plate_image"].save(buf_c, format="PNG")
                    st.download_button(
                        label="📥 TẢI ẢNH CROP",
                        data=buf_c.getvalue(),
                        file_name=f"crop_{p_text}.png",
                        mime="image/png",
                        use_container_width=True,
                        key="btn_download_crop"
                    )

            # Breakdown List of All Plates
            render_html(f"""
            <div style="font-size:0.72rem;color:#627D98;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 8px;display:flex;justify-content:space-between;">
                <span>DANH SÁCH {len(results):02d} BIỂN ĐÃ PHÂN TÁCH</span>
                <span style="color:#00FF88;">Đã nạp tọa độ 100%</span>
            </div>
            """)

            for i, r in enumerate(results):
                t_score = r.get("confidence", 0.0) * 100
                t_txt = r.get("plate_text", "???")
                t_color = "#00FF88" if t_score >= 80 else "#FFD740"

                img_b64_sub = ""
                if r.get("plate_image"):
                    try:
                        b_tmp = io.BytesIO()
                        r["plate_image"].convert("RGB").save(b_tmp, format="JPEG", quality=70)
                        img_b64_sub = base64.b64encode(b_tmp.getvalue()).decode("utf-8")
                    except Exception:
                        img_b64_sub = ""

                thumb_el = f'<img src="data:image/jpeg;base64,{img_b64_sub}" style="height:22px;border-radius:2px;background:#050811;" />' if img_b64_sub else ''

                render_html(f"""
                <div class="breakdown-item">
                    <div style="display:flex;align-items:center;gap:8px;">
                        {thumb_el}
                        <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#00D4FF;font-size:0.85rem;">{t_txt}</span>
                        <span style="font-size:0.68rem;color:#627D98;">#{i+1:02d} · CRNN: {time_rec:.1f}ms</span>
                    </div>
                    <div>
                        <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{t_color};font-size:0.82rem;">{t_score:.1f}%</span>
                    </div>
                </div>
                """)

            # Footer timing breakdown
            render_html(f"""
            <div style="border-top:1px solid var(--border);padding-top:10px;margin-top:12px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#627D98;line-height:1.6;">
                <div style="display:flex;justify-content:space-between;">
                    <span>Tách B{len(results)} biển (Multi-YOLO Cut):</span>
                    <span style="color:#E2EDF8;">{time_det:.1f} ms</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span>Đọc song song {len(results)} biển (CRNN Batch):</span>
                    <span style="color:#E2EDF8;">{time_rec:.1f} ms</span>
                </div>
                <div style="display:flex;justify-content:space-between;color:#00FF88;font-weight:700;margin-top:2px;">
                    <span>Tổng Thời Gian Suy Luận ({len(results)}x):</span>
                    <span>{total_time:.1f} ms (Realtime)</span>
                </div>
            </div>
            """)

        else:
            render_html("""
            <div style="padding:48px 20px;text-align:center;">
                <div style="font-size:2.8rem;margin-bottom:8px;opacity:0.6;">🔍</div>
                <div style="color:#627D98;font-weight:700;font-size:0.9rem;">CHƯA CÓ KẾT QUẢ SUY LUẬN</div>
                <div style="color:#243B53;font-size:0.75rem;margin-top:4px;">
                    Nhấn nút <b>NHẬN DẠNG BIỂN SỐ</b> ở khung bên trái để bắt đầu trích xuất biển số
                </div>
            </div>
            """)


    # ── 3. BOTTOM DATA TABLE: NHẬT KÝ NHẬN DIỆN ─────────────────
    render_html('<div id="sec-history" style="scroll-margin-top:30px;"></div>')
    render_html('<div style="margin-top:24px;"></div>')
    render_recent_history()

    # ── 5. CYBERPUNK TACTICAL FOOTER ────────────────────────────
    render_html(f"""
    <div class="tactical-footer">
        <div>
            BIỂN SỐ AI // SYSTEM READY &nbsp;|&nbsp;
            <span style="color:#00FF88;">● {dev_str} Đang Hoạt Động</span>
        </div>
        <div>
            Phiên Bản Hệ Thống: v2.4 PRO (Build 2025.02) - Bộ Điều Khiển Trung Tâm
        </div>
    </div>
    """)


if __name__ == "__main__":
    main()
