# ids_real_model_FINAL.py
# This one works — no errors, real F1 score

import joblib
import pandas as pd
import numpy as np
import sys
from collections import defaultdict, deque

# Load model
print("Loading model and preprocessors...")
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm = ensemble['ocsvm']
le = joblib.load('id_encoder.pkl')
scaler = joblib.load('scaler.pkl')

print("Model loaded — encoder classes:", le.classes_.tolist())

buffers = defaultdict(lambda: deque(maxlen=5))
signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']

def process_line(parts):
    if len(parts) < 7:
        return None

    try:
        label = int(parts[0])                    # Label
        time_val = float(parts[1])               # Time
        can_id = parts[2].strip()                # ID
        s1 = pd.to_numeric(parts[3], errors='coerce')
        s2 = pd.to_numeric(parts[4], errors='coerce')
        s3 = pd.to_numeric(parts[5], errors='coerce')
        s4 = pd.to_numeric(parts[6], errors='coerce')
    except:
        return None

    row = {
        'Time': time_val,
        'ID': can_id,
        'Signal1': s1 if pd.notna(s1) else 0.0,
        'Signal2': s2 if pd.notna(s2) else 0.0,
        'Signal3': s3 if pd.notna(s3) else 0.0,
        'Signal4': s4 if pd.notna(s4) else 0.0
    }

    buf = buffers[can_id]
    buf.append(row.copy())

    # Time delta
    td = 1.0
    if len(buf) >= 2:
        td = row['Time'] - buf[-2]['Time']

    # Signal deltas
    deltas = np.zeros(4)
    abs_deltas = np.zeros(4)
    if len(buf) >= 2:
        prev = buf[-2]
        for i, col in enumerate(signal_cols):
            c = row[col]
            p = prev[col]
            deltas[i] = c - p
            abs_deltas[i] = abs(deltas[i])

    # Rolling stats
    roll_var = [0.0]*4
    roll_mean = [0.0]*4
    if len(buf) >= 2:
        for i, col in enumerate(signal_cols):
            hist = [buf[j][col] - buf[j-1][col] for j in range(1, len(buf))
                    if pd.notna(buf[j][col]) and pd.notna(buf[j-1][col])]
            recent = hist[-5:] or [0.0]
            roll_var[i] = np.var(recent) if len(recent) > 1 else 0.0
            roll_mean[i] = np.mean(recent)

    # 18 features
    features = np.array([
        td,
        deltas[0], abs_deltas[0], roll_var[0], roll_mean[0],
        deltas[1], abs_deltas[1], roll_var[1], roll_mean[1],
        deltas[2], abs_deltas[2], roll_var[2], roll_mean[2],
        deltas[3], abs_deltas[3], roll_var[3], roll_mean[3],
        le.transform([can_id])[0]
    ], dtype=np.float32).reshape(1, -1)

    features[:, :17] = scaler.transform(features[:, :17])

    score = (if_model.decision_function(features)[0] + ocsvm.score_samples(features)[0]) / 2
    proba = -score

    return can_id, td, proba, label

# Main loop with F1
msg_count = 0
tp = fp = fn = tn = 0

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line or line.startswith('Label'):
        continue

    parts = line.split(',')
    result = process_line(parts)
    if result is None:
        continue

    can_id, td, proba, true_label = result
    msg_count += 1

    pred_label = 1 if proba > 0.6 else 0
    status = "ALERT!" if pred_label == 1 else "Normal"

    # Confusion matrix
    if true_label == 1 and pred_label == 1:
        tp += 1
    elif true_label == 0 and pred_label == 1:
        fp += 1
    elif true_label == 1 and pred_label == 0:
        fn += 1
    else:
        tn += 1

    print(f"Msg {msg_count:6d} | ID={can_id:>6} | Δt={td:8.3f} | Proba={proba:6.3f} | {status} | Label={true_label}")

# Final F1
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n" + "="*80)
print(f"TOTAL MESSAGES: {msg_count:,}")
print(f"TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
print(f"PRECISION: {precision:.4f}  RECALL: {recall:.4f}  F1: {f1:.4f}")
print("="*80)
