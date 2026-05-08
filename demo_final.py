"""
Air-Writing Character Recognition — Live Demo
All 4 Improvements from Report Active

  8.1 Time Series Forecasting  : fading blue dots ahead of fingertip show predicted stroke direction
  8.2 Multi-Stroke Support     : SPACE commits a stroke; draw more; ENTER predicts combined
  8.3 Writer-Adaptive LoRA     : loads lora_model.pt if present, else best_model.pt
  8.4 Confidence Thresholding  : threshold = 0.50 (calibrated on validation set, accuracy 80.5%)

HOW TO DRAW:
  Single character  (a, b, c, e, m, n, o, s, w...):
    → Point index finger, draw the character, hide hand → auto predicts in 2 seconds

  Multi-stroke chars (i, j, t, f, x, k...):
    → Draw first part (body), press SPACE to save it
    → Draw second part (dot/crossbar), press ENTER to predict
    → OR hide hand after each stroke, then ENTER at the end

KEYBOARD:
  SPACE  = save current stroke (multi-stroke mode)
  ENTER  = predict from all saved strokes
  C      = clear everything and start over
  Q      = quit
"""

import cv2, pickle, numpy as np, urllib.request, os
from collections import deque
import torch, torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions — must match training exactly
# ─────────────────────────────────────────────────────────────────────────────
class StrokeTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=256, nhead=8,
                 num_layers=6, num_classes=97, dropout=0.15, max_len=1000):
        super().__init__()
        self.embedding    = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Embedding(max_len + 1, d_model)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            el, num_layers=num_layers, enable_nested_tensor=False)
        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        B, T, _ = x.shape
        x   = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(B, -1)
        x   = x + self.pos_encoding(pos)
        x   = self.dropout(x)
        if src_key_padding_mask is not None:
            cm = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cm, src_key_padding_mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(self.norm(x)[:, 0, :])


class StrokeForecaster(nn.Module):
    """Improvement 8.1: regression head predicts future (x,y) coordinates."""
    def __init__(self, input_dim=2, d_model=256, nhead=8,
                 num_layers=6, dropout=0.15, max_len=1000, predict_len=20):
        super().__init__()
        self.predict_len  = predict_len
        self.embedding    = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Embedding(max_len + 1, d_model)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer     = nn.TransformerEncoder(
            el, num_layers=num_layers, enable_nested_tensor=False)
        self.norm            = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, predict_len * 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        B, T, _ = x.shape
        x   = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(B, -1)
        x   = x + self.pos_encoding(pos)
        x   = self.dropout(x)
        if src_key_padding_mask is not None:
            cm = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cm, src_key_padding_mask], dim=1)
        x   = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        out = self.regression_head(self.norm(x)[:, 0, :])
        return out.view(B, self.predict_len, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
print("Loading...")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
N = len(le.classes_)

# 8.3: use LoRA-adapted model if available
model       = StrokeTransformer(num_classes=N)
model_label = "Base"
if os.path.exists("lora_model.pt"):
    try:
        model.load_state_dict(
            torch.load("lora_model.pt", map_location="cpu"), strict=False)
        model_label = "LoRA"
        print("  [8.3] LoRA model loaded")
    except Exception:
        model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
        print("  LoRA load failed — using base model")
else:
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    print("  Base model loaded")
model.eval()

# 8.1: load forecaster
forecaster = None
if os.path.exists("forecaster.pt"):
    try:
        forecaster = StrokeForecaster(predict_len=20)
        forecaster.load_state_dict(
            torch.load("forecaster.pt", map_location="cpu"))
        forecaster.eval()
        print("  [8.1] Forecaster loaded")
    except Exception:
        forecaster = None
        print("  Forecaster load failed — 8.1 visual disabled")
else:
    print("  No forecaster.pt — 8.1 visual disabled")

print(f"  {N} classes | Model: {model_label}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe hand detector
# ─────────────────────────────────────────────────────────────────────────────
HAND_MODEL = "hand_landmarker.task"
if not os.path.exists(HAND_MODEL):
    print("Downloading MediaPipe hand model (~30 MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task", HAND_MODEL)
    print("  Done.")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE))

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESH        = 0.50   # 8.4: calibrated on val set (acc=80.5%, cov=76%)
MIN_PTS            = 8      # minimum points for a valid stroke
AUTO_COMMIT_FRAMES = 22     # no-hand frames before auto-committing stroke
AUTO_PREDICT_FRAMES= 55     # no-hand frames before auto-predicting
FORECAST_INTERVAL  = 18     # update forecast every N points

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
committed      = []          # list of saved stroke arrays (np.float32)
cur_stroke     = []          # current stroke being drawn (normalised x,y)
trail          = deque(maxlen=600)   # pixel coords for drawing trail
smooth_buf     = deque(maxlen=5)     # smoothing buffer
forecast_dots  = []          # 8.1: list of (px,py) to draw as forecast dots
fore_count     = 0           # counter for forecast update timing
prediction     = ""
conf_val       = 0.0
drawing        = False
no_hand        = 0           # consecutive frames with no hand


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def is_index_only(lm):
    """Only index finger extended — DRAW."""
    return (finger_up(lm, 8, 6) and
            not finger_up(lm, 12, 10) and
            not finger_up(lm, 16, 14) and
            not finger_up(lm, 20, 18))

def smooth(x, y):
    smooth_buf.append((x, y))
    return (sum(p[0] for p in smooth_buf) / len(smooth_buf),
            sum(p[1] for p in smooth_buf) / len(smooth_buf))

def normalize(pts):
    arr = np.array(pts, dtype=np.float32)
    mn, mx = arr.min(0), arr.max(0)
    rng = np.where(mx - mn == 0, 1.0, (mx - mn).astype(np.float32))
    return ((arr - mn) / rng).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 8.1: Stroke forecasting
# ─────────────────────────────────────────────────────────────────────────────
def compute_forecast(cur_pts, tip_px, fw, fh):
    """
    8.1: Feed normalised current stroke into forecaster.
    Map predicted offsets relative to fingertip so dots stay near the drawing.
    """
    if forecaster is None or len(cur_pts) < 15:
        return []
    arr  = normalize(cur_pts)
    t    = torch.from_numpy(arr).unsqueeze(0)
    with torch.no_grad():
        pred = forecaster(t)[0].cpu().numpy()   # (20, 2) in normalised space
    last = arr[-1]
    tx, ty = tip_px
    scale  = min(fw, fh) * 0.18               # keep dots close to fingertip
    dots   = []
    for nx, ny in pred[:8]:
        dx = (nx - last[0]) * scale
        dy = (ny - last[1]) * scale
        px, py = int(tx + dx), int(ty + dy)
        if 0 < px < fw and 0 < py < fh:
            dots.append((px, py))
    return dots


# ─────────────────────────────────────────────────────────────────────────────
# 8.2 + 8.4: Classify with separator tokens and confidence threshold
# ─────────────────────────────────────────────────────────────────────────────
def classify(strokes):
    """
    8.2: Combine strokes with separator token [-0.5, -0.5].
    8.4: Reject prediction if softmax confidence < CONF_THRESH (0.50).
    """
    if not strokes:
        return "?", 0.0

    SEP   = np.array([[-0.5, -0.5]], dtype=np.float32)
    parts = []
    for i, s in enumerate(strokes):
        parts.append(s)
        if i < len(strokes) - 1:
            parts.append(SEP)

    seq = np.vstack(parts).astype(np.float32)
    mn, mx = seq.min(0), seq.max(0)
    rng = np.where(mx - mn == 0, 1.0, (mx - mn).astype(np.float32))
    seq = ((seq - mn) / rng).astype(np.float32)

    t = torch.from_numpy(seq).unsqueeze(0)
    with torch.no_grad():
        out   = model(t)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = probs.max(0)
        conf  = float(conf.item())
        char  = le.classes_[int(idx.item())]

    if conf < CONF_THRESH:          # 8.4
        return "unclear — redraw", conf
    return char, conf


# ─────────────────────────────────────────────────────────────────────────────
# Stroke actions
# ─────────────────────────────────────────────────────────────────────────────
def commit_stroke(reason=""):
    global cur_stroke, drawing, no_hand, forecast_dots, fore_count
    if len(cur_stroke) >= MIN_PTS:
        committed.append(np.array(cur_stroke, dtype=np.float32))
        n = len(committed)
        print(f"  Stroke {n} committed ({len(cur_stroke)} pts)"
              + (f" [{reason}]" if reason else ""))
    else:
        print("  Stroke too short, discarded")
    cur_stroke = []
    trail.clear()
    smooth_buf.clear()
    forecast_dots = []
    fore_count    = 0
    drawing       = False
    no_hand       = 0


def do_predict():
    global prediction, conf_val, committed, drawing, no_hand
    if len(cur_stroke) >= MIN_PTS:
        commit_stroke("auto-included")
    if not committed:
        print("  Nothing to predict")
        return
    prediction, conf_val = classify(committed)
    committed  = []
    trail.clear()
    drawing    = False
    no_hand    = 0
    print(f"  → \"{prediction}\"  conf={conf_val:.0%}")


def reset():
    global committed, cur_stroke, trail, smooth_buf, forecast_dots
    global drawing, no_hand, prediction, conf_val, fore_count
    committed=[]; cur_stroke=[]; trail.clear(); smooth_buf.clear()
    forecast_dots=[]; drawing=False; no_hand=0
    prediction=""; conf_val=0.0; fore_count=0
    print("  Cleared")


# ─────────────────────────────────────────────────────────────────────────────
# Render — clean, minimal UI
# ─────────────────────────────────────────────────────────────────────────────
def render(frame, h, w):

    # ── Stroke trail ──────────────────────────────────────────────────────────
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / max(len(pts), 1)
        color = (0, int(130 + 125 * alpha), 40)
        thick = max(2, int(alpha * 6))
        cv2.line(frame, pts[i-1], pts[i], color, thick)

    # ── 8.1: Forecast dots (fading blue, near fingertip) ─────────────────────
    if forecast_dots and drawing:
        n = len(forecast_dots)
        for i, (fx, fy) in enumerate(forecast_dots):
            alpha  = 1.0 - i / max(n, 1)
            radius = max(3, int(8 * alpha))
            color  = (int(255 * alpha), int(140 * alpha), 0)   # blue-ish fading
            cv2.circle(frame, (fx, fy), radius, color, -1)

    # ── 8.2: Saved stroke markers ─────────────────────────────────────────────
    for k, s in enumerate(committed):
        if not len(s): continue
        # Draw a small numbered circle at the endpoint of each saved stroke
        ex = int(np.clip(s[-1][0], 0, 1) * w)
        ey = int(np.clip(s[-1][1], 0, 1) * h)
        cv2.circle(frame, (ex, ey), 10, (40, 40, 200), -1)
        cv2.circle(frame, (ex, ey), 12, (160, 160, 255), 1)
        cv2.putText(frame, str(k + 1), (ex - 4, ey + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 255), 1)

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 76), (10, 10, 10), -1)

    # Prediction text
    if prediction:
        unclear = "unclear" in prediction
        col     = (40, 40, 220) if unclear else (0, 230, 80)
        text    = prediction
    else:
        col  = (55, 55, 55)
        text = "—"
    cv2.putText(frame, f"Prediction:  {text}",
                (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.05, col, 3)

    # 8.3: Model label (small, top-left below prediction)
    col_m = (80, 140, 255) if "LoRA" in model_label else (100, 100, 100)
    cv2.putText(frame, f"[8.3] {model_label}",
                (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_m, 1)

    # 8.4: Confidence bar (top-right)
    if conf_val > 0:
        bx   = w - 170
        bw_f = int(min(conf_val, 1.0) * 140)
        bc   = (0, 190, 60) if conf_val >= CONF_THRESH else (30, 30, 200)
        cv2.rectangle(frame, (bx, 12), (bx + 140, 28), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, 12), (bx + bw_f, 28), bc, -1)
        cv2.putText(frame, f"[8.4] conf {conf_val:.0%}",
                    (bx, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (120, 120, 120), 1)
        cv2.putText(frame, f"threshold {CONF_THRESH:.0%}",
                    (bx, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 80, 80), 1)

    # Stroke count (8.2)
    n_st = len(committed) + (1 if drawing else 0)
    if n_st > 0:
        cv2.putText(frame, f"[8.2] strokes: {n_st}",
                    (w - 148, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 160, 255), 1)

    # 8.1 indicator
    fc_col = (0, 180, 255) if forecaster else (60, 60, 60)
    cv2.putText(frame, "[8.1] forecast",
                (w - 148, 12 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, fc_col, 1)

    # ── BOTTOM STATUS BAR ─────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 62), (w, h), (10, 10, 10), -1)

    # Dynamic status line
    if drawing:
        msg = f"Drawing...  {len(cur_stroke)} points captured"
        col = (0, 210, 60)
    elif committed:
        msg = f"{len(committed)} stroke(s) saved — draw more  OR  press ENTER to predict"
        col = (0, 160, 255)
    elif prediction and "unclear" not in prediction:
        msg = "Predicted!  Press C to clear and draw again"
        col = (0, 200, 60)
    elif prediction and "unclear" in prediction:
        msg = "Low confidence — please redraw the character more clearly"
        col = (40, 40, 220)
    else:
        msg = "Point your index finger at the camera and draw"
        col = (120, 120, 120)

    cv2.putText(frame, msg, (14, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)

    # Controls
    cv2.putText(frame,
                "SPACE = save stroke  |  ENTER = predict  |  C = clear  |  Q = quit",
                (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (85, 85, 85), 1)
    cv2.putText(frame,
                "Multi-stroke chars (i,t,j,f): draw body → SPACE → draw dot/bar → ENTER",
                (14, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 60, 60), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("=" * 55)
print("  Air-Writing Demo — All 4 Improvements Active")
print("=" * 55)
print(f"  8.1 Forecast     : {'ON' if forecaster else 'OFF (forecaster.pt missing)'}")
print(f"  8.2 Multi-stroke : ON — SPACE to save stroke, ENTER to predict")
print(f"  8.3 LoRA         : {model_label}")
print(f"  8.4 Confidence   : threshold = {CONF_THRESH:.0%}  (calibrated on val set)")
print()
print("  SPACE = save stroke   ENTER = predict   C = clear   Q = quit")
print("  For i/t/j/f: draw body → SPACE → draw dot/bar → ENTER")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    hand_present = bool(result.hand_landmarks)

    if hand_present:
        no_hand = 0                          # reset only when hand truly visible
        lm      = result.hand_landmarks[0]

        if is_index_only(lm):
            tip    = lm[8]
            sx, sy = smooth(tip.x, tip.y)
            cx, cy = int(sx * w), int(sy * h)

            cur_stroke.append([sx, sy])
            trail.append((cx, cy))
            drawing = True

            # Fingertip marker
            cv2.circle(frame, (cx, cy), 14, (0, 255, 120), -1)
            cv2.circle(frame, (cx, cy), 16, (255, 255, 255), 1)

            # 8.1: update forecast periodically
            fore_count += 1
            if forecaster and fore_count % FORECAST_INTERVAL == 0:
                forecast_dots = compute_forecast(cur_stroke, (cx, cy), w, h)

        else:
            # Hand visible but not drawing gesture
            # Do NOT increment no_hand — prevents accidental auto-commit
            forecast_dots = []
            cv2.putText(frame, "point only index finger to draw",
                        (14, h - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 90, 210), 2)

    else:
        # No hand at all — safe to increment counter
        no_hand       += 1
        forecast_dots  = []

        # Step 1: auto-commit the stroke after short pause
        if drawing and no_hand >= AUTO_COMMIT_FRAMES:
            commit_stroke("hand removed")

        # Step 2: auto-predict after longer pause
        if (not drawing and
                committed and
                no_hand >= AUTO_PREDICT_FRAMES):
            do_predict()

    render(frame, h, w)
    cv2.imshow("Air-Writing Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord("q"):
        break
    elif key == ord("c"):
        reset()
    elif key == ord(" "):
        # SPACE: save current stroke
        if drawing or len(cur_stroke) >= MIN_PTS:
            commit_stroke("SPACE key")
        elif committed:
            print("  Strokes already saved — draw more or press ENTER")
        else:
            print("  Nothing drawn yet")
    elif key == 13:
        # ENTER: predict now
        do_predict()

cap.release()
cv2.destroyAllWindows()
