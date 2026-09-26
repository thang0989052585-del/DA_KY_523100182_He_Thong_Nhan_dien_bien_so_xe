"""
model.py — Kiến Trúc Mô Hình Nhận Dạng Biển Số Xe (Phiên Bản Nâng Cấp)
==========================================================================
File này xây dựng mạng CRNN nâng cấp để nhận dạng ký tự biển số xe.

KIẾN TRÚC NÂNG CẤP (CRNN v2):
┌─────────────────────────────────────────────────────────────────────┐
│  Ảnh biển số (1 × 64 × 192)                                        │
│         ↓                                                           │
│  [ResNet-style CNN + SE Attention]                                  │
│  7 lớp Conv với Residual Connections + Squeeze-Excitation blocks    │
│  → Feature Map (512 × 1 × 48)                                      │
│         ↓                                                           │
│  [Map-to-Sequence + Positional Encoding]                            │
│  Thêm thông tin vị trí cho mỗi bước thời gian                      │
│         ↓                                                           │
│  [Bi-LSTM] — 2 lớp, 256 hidden, Bidirectional                      │
│         ↓                                                           │
│  [Attention Layer] — Tập trung vào vị trí ký tự quan trọng         │
│         ↓                                                           │
│  [Linear Head] → Log-Softmax                                        │
│         ↓                                                           │
│  [CTC Decode] → "51A-12345"                                         │
└─────────────────────────────────────────────────────────────────────┘

NÂNG CẤP SO VỚI V1:
  ✓ Residual Connections: Gradient chảy dễ hơn → train sâu hơn, hội tụ nhanh
  ✓ SE Attention: Mạng tự học "kênh nào quan trọng" → bỏ nhiễu, giữ ký tự
  ✓ Deeper CNN (7 layers): Trích xuất đặc trưng phong phú hơn
  ✓ Positional Encoding: Giúp model biết vị trí ký tự trên biển số
  ✓ Attention sau LSTM: Tập trung vào vùng chứa ký tự, bỏ qua nền trống

TẠI SAO DÙNG CTC (Connectionist Temporal Classification)?
  Biển số xe có độ dài ký tự KHÔNG cố định (VD: "29A-1234" vs "51G-00123").
  CTC giải quyết vấn đề này: Không cần căn chỉnh từng ký tự với từng pixel —
  Mạng tự học cách sắp xếp và hội tụ đúng vị trí ký tự.

TẠI SAO BI-LSTM?
  Ký tự "1" đứng sau "5" và trước "A" mang ý nghĩa khác nhau theo ngữ cảnh.
  LSTM thuận (→) học ngữ cảnh trái-sang-phải, LSTM ngược (←) học phải-sang-trái.
  Kết hợp 2 chiều giúp nhận dạng chính xác hơn trong điều kiện nhiễu.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ──────────────────────────────────────────────────────────────
# SQUEEZE-AND-EXCITATION BLOCK — Attention trên kênh
# ──────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block — Cơ chế attention tự động trên kênh.

    Ý tưởng:
      1. Squeeze: Nén mỗi feature map thành 1 số (Global Avg Pool)
      2. Excitation: Dùng 2 FC layers để học "kênh nào quan trọng"
      3. Scale: Nhân lại vào feature map gốc

    Kết quả: Mạng tự động tăng weight cho kênh chứa ký tự,
             giảm weight cho kênh chứa nhiễu/nền.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return x * scale


# ──────────────────────────────────────────────────────────────
# RESIDUAL BLOCK — Kết nối tắt để train sâu hơn
# ──────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual Block với 2 lớp Conv + SE Attention.

    y = SEBlock(Conv(BN(ReLU(Conv(BN(ReLU(x))))))) + shortcut(x)

    Residual connection giúp gradient chảy trực tiếp → train được
    mạng sâu hơn mà không bị vanishing gradient.
    """

    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        # Shortcut: nếu in_ch ≠ out_ch, dùng 1x1 conv để match dimensions
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out = F.relu(out + identity, inplace=True)  # Residual connection!
        return out


# ──────────────────────────────────────────────────────────────
# CNN BACKBONE NÂNG CẤP — ResNet-style + SE Attention
# ──────────────────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """
    Mạng CNN nâng cấp với Residual Connections và SE Attention.

    So với v1 (4 Conv đơn giản):
      ✓ 7 lớp Conv (sâu hơn → trích xuất đặc trưng tốt hơn)
      ✓ Residual connections (train ổn định, không vanishing gradient)
      ✓ SE blocks (tự động focus vào kênh quan trọng)
      ✓ Dropout2d (giảm overfitting)

    Input : (B, 1, 64, 192)
    Output: (B, 512, 1, 48)
    """

    def __init__(self):
        super().__init__()

        # Lớp đầu vào: mở rộng từ 1 kênh (xám) → 64 kênh
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: (B, 64, 64, 192) → (B, 64, 32, 96)
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, use_se=False),
            nn.MaxPool2d(kernel_size=2, stride=2),   # /2H, /2W
        )

        # Stage 2: (B, 64, 32, 96) → (B, 128, 16, 48)
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, use_se=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # /2H, /2W
        )

        # Stage 3: (B, 128, 16, 48) → (B, 256, 8, 48)
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, use_se=True),
            nn.MaxPool2d(kernel_size=(2, 1)),         # /2H only
            nn.Dropout2d(0.1),
        )

        # Stage 4: (B, 256, 8, 48) → (B, 512, 4, 48)
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, use_se=True),
            nn.MaxPool2d(kernel_size=(2, 1)),         # /2H only
            nn.Dropout2d(0.1),
        )

        # Nén chiều dọc còn lại về 1
        self.pool_final = nn.AdaptiveAvgPool2d((1, None))  # height → 1

    def forward(self, x):
        x = self.stem(x)     # (B, 64, 64, 192)
        x = self.stage1(x)   # (B, 64, 32, 96)
        x = self.stage2(x)   # (B, 128, 16, 48)
        x = self.stage3(x)   # (B, 256, 8, 48)
        x = self.stage4(x)   # (B, 512, 4, 48)
        x = self.pool_final(x)  # (B, 512, 1, 48)
        return x


# ──────────────────────────────────────────────────────────────
# POSITIONAL ENCODING — Thêm thông tin vị trí
# ──────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Thêm thông tin vị trí sinusoidal cho chuỗi thời gian.

    Giúp model biết "ký tự này ở vị trí thứ mấy trên biển số".
    Quan trọng vì: ký tự đầu (mã tỉnh) và ký tự cuối (số thứ tự)
    có ý nghĩa rất khác nhau.
    """

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T, B, D)
        return x + self.pe[:x.size(0)]


# ──────────────────────────────────────────────────────────────
# ATTENTION LAYER — Tập trung vào ký tự quan trọng
# ──────────────────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    """
    Cơ chế Attention đơn giản sau BiLSTM.

    Tại mỗi bước thời gian, tính trọng số attention dựa trên nội dung,
    giúp model "nhìn" vào các bước thời gian lân cận khi quyết định ký tự.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (T, B, hidden_size)
        Returns:
            attended: (T, B, hidden_size) — đã tăng cường bởi attention
        """
        # Tính attention weights
        attn_weights = self.attention(lstm_output)  # (T, B, 1)
        attn_weights = F.softmax(attn_weights, dim=0)  # softmax theo chiều T

        # Context vector (weighted sum)
        context = (attn_weights * lstm_output).sum(dim=0, keepdim=True)  # (1, B, H)
        context = context.expand_as(lstm_output)  # broadcast về (T, B, H)

        # Kết hợp: output gốc + context → giúp mỗi bước "thấy" toàn cục
        return lstm_output + context * 0.1  # scale nhỏ để không phá gradient


# ──────────────────────────────────────────────────────────────
# MÔ HÌNH CRNN NÂNG CẤP
# ──────────────────────────────────────────────────────────────

class CRNN(nn.Module):
    """
    CRNN v2 — Nâng cấp toàn diện từ v1:
      - ResNet CNN backbone thay cho plain CNN
      - SE Attention blocks trong CNN
      - Positional Encoding
      - Attention Layer sau BiLSTM
      - Deeper architecture (nhiều tham số hơn nhưng train tốt hơn)

    Luồng xử lý:
      Input (B,1,64,192) → CNN → PosEnc → BiLSTM → Attention → Head → (T,B,C)
    """

    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 lstm_hidden: int = config.LSTM_HIDDEN,
                 lstm_layers: int = config.LSTM_LAYERS,
                 lstm_dropout: float = config.LSTM_DROPOUT):
        super().__init__()

        self.cnn = CNNBackbone()

        # CNN output channels (512 cho phiên bản nâng cấp)
        cnn_out_channels = 512

        # Positional Encoding
        self.pos_enc = PositionalEncoding(cnn_out_channels)

        # Dropout sau CNN
        self.cnn_dropout = nn.Dropout(0.2)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # Layer Normalization sau LSTM (ổn định hơn BatchNorm cho chuỗi)
        self.ln = nn.LayerNorm(lstm_hidden * 2)

        # Attention layer
        self.attention = AttentionLayer(lstm_hidden * 2)

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor ảnh biển số (B, C, H, W) — C=1 (xám), H=64, W=192
        Returns:
            log_probs: (T, B, NUM_CLASSES) — dùng trực tiếp cho CTCLoss
        """
        # ── CNN ────────────────────────────────────────────────
        feat = self.cnn(x)            # (B, 512, 1, W') với W'=48

        # ── Map-to-Sequence ─────────────────────────────────────
        B, C, H, W = feat.shape
        feat = feat.squeeze(2)        # (B, 512, 48)
        feat = feat.permute(2, 0, 1)  # (48, B, 512)

        # ── Positional Encoding ─────────────────────────────────
        feat = self.pos_enc(feat)
        feat = self.cnn_dropout(feat)

        # ── Bi-LSTM ───────────────────────────────────────────
        feat, _ = self.lstm(feat)     # (48, B, 512)  [hidden*2]
        feat = self.ln(feat)          # Layer Normalization

        # ── Attention ─────────────────────────────────────────
        feat = self.attention(feat)   # (48, B, 512)

        # ── Linear Head ──────────────────────────────────────
        log_probs = self.head(feat)   # (48, B, NUM_CLASSES)
        log_probs = log_probs.log_softmax(dim=2)

        return log_probs              # (T=48, B, C=NUM_CLASSES)


# ──────────────────────────────────────────────────────────────
# GIẢI MÃ CTC (CTC GREEDY DECODER)
# ──────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int = 0) -> list:
    """
    Giải mã đầu ra CTC bằng thuật toán Greedy (chọn ký tự có xác suất cao nhất
    tại mỗi bước, sau đó loại blank và ký tự trùng lặp liên tiếp).

    Args:
        log_probs: Tensor (T, B, C) — đầu ra log-softmax của mạng
        blank_idx: Index của ký tự trống CTC (mặc định 0)

    Returns:
        List[str]: Danh sách chuỗi ký tự được giải mã cho từng ảnh trong batch
    """
    # greedy: lấy argmax theo chiều C ở mỗi bước thời gian
    pred_indices = log_probs.argmax(dim=2)  # (T, B)
    pred_indices = pred_indices.permute(1, 0)  # (B, T)

    decoded_strings = []
    for b in range(pred_indices.shape[0]):
        indices = pred_indices[b].tolist()
        chars = []
        prev = blank_idx
        for idx in indices:
            if idx != blank_idx and idx != prev:
                chars.append(config.IDX2CHAR.get(idx, "?"))
            prev = idx
        decoded_strings.append("".join(chars))

    return decoded_strings


# ──────────────────────────────────────────────────────────────
# CÁC HÀM TIỆN ÍCH
# ──────────────────────────────────────────────────────────────

def build_model() -> CRNN:
    """Khởi tạo mô hình CRNN mới."""
    return CRNN()


def get_device() -> torch.device:
    """Select device: GPU NVIDIA or CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU (No GPU found)")
    return dev


def load_model(path: str, device: torch.device) -> CRNN:
    """
    Load model weights from .pth file.

    Args:
        path  : Path to .pth file
        device: Device (cpu / cuda)
    Returns:
        Model with loaded weights, in eval() mode
    """
    model = build_model().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[Model] Loaded from: {path}")
    return model


def count_parameters(model: nn.Module) -> dict:
    """Thống kê số tham số (tế bào não) của mô hình."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ──────────────────────────────────────────────────────────────
# KIỂM TRA NHANH
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CRNN v2 Model Test - License Plate Recognition")
    print("=" * 60)

    model = build_model()
    stats = count_parameters(model)
    print(f"\nParameter stats:")
    print(f"  Total     : {stats['total']:,}")
    print(f"  Trainable : {stats['trainable']:,}")

    # Test forward pass with dummy batch
    device = get_device()
    model = model.to(device)
    dummy = torch.randn(4, 1, config.PLATE_HEIGHT, config.PLATE_WIDTH).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"\nForward pass test:")
    print(f"  Input : {tuple(dummy.shape)}  (B, C, H, W)")
    print(f"  Output: {tuple(out.shape)}    (T, B, NUM_CLASSES)")
    print(f"  T = {out.shape[0]} timesteps (width / 4)")
    print(f"  NUM_CLASSES = {config.NUM_CLASSES}")


