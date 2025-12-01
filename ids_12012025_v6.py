import joblib
import pandas as pd
import numpy as np
import sys
from collections import defaultdict, deque

# Load the perfect new files
print("Loading model and preprocessors...")
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm = ensemble['ocsvm']
le = joblib.load('id_encoder.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

print("Model loaded – encoder knows:", le.classes_.tolist())

buffers = defaultdict(lambda: deque(maxlen=5))
signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']

def process_line(parts):
    if len(parts) < 7: return None

    row = {
        'Time': float(parts[1]),
        'ID': parts[2].strip(),
        'Signal1': pd.to_numeric(parts[3], errors='coerce'),
        'Signal2': pd.to_numeric(parts[4], errors='coerce'),
        'Signal3': pd.to_numeric(parts[5], errors='coerce'),
        'Signal4': pd.to_numeric(parts[6], errors='coerce')
    }

    # State buffer
    buf = buffers[row['ID']]
    buf.append(row.copy())

    # Time delta
    td = 1.0
    if len(buf) >= 2:
        td = row['Time'] - buf[-2]['Time']

    # Deltas
    deltas = np.zeros(4)
    abs_deltas = np.zeros(4)
    if len(buf) >= 2:
        prev = buf[-2]
        for i, col in enumerate(signal_cols):
            c = row[col] if pd.notna(row[col]) else 0.0
            p = prev[col] if pd.notna(prev[col]) else 0.0
            deltas[i] = c - p
            abs_deltas[i] = abs(deltas[i])

    # Rolling stats
    roll_var = [0.0]*4
    roll_mean = [0.0]*4
    if len(buf) >= 2:
        for i, col in enumerate(signal_cols):
            hist = [buf[j][col] - buf[j-1][col]
                    for j in range(1, len(buf))
                    if pd.notna(buf[j][col]) and pd.notna(buf[j-1][col])]
            recent = hist[-5:] or [0.0]
            roll_var[i] = np.var(recent) if len(recent)>1 else 0.0
            roll_mean[i] = np.mean(recent)

    # Final 18-feature vector
    features = np.array([
        td,
        deltas[0], abs_deltas[0], roll_var[0], roll_mean[0],
        deltas[1], abs_deltas[1], roll_var[1], roll_mean[1],
        deltas[2], abs_deltas[2], roll_var[2], roll_mean[2],
        deltas[3], abs_deltas[3], roll_var[3], roll_mean[3],
        le.transform([row['ID']])[0]
    ], dtype=np.float32).reshape(1, -1)

    features[:, :17] = scaler.transform(features[:, :17])

    # Score
    if_score = if_model.decision_function(features)[0]
    svm_score = ocsvm.score_samples(features)[0]
    proba = -(if_score + svm_score) / 2

    return row['ID'], td, proba

# Main loop
msg = 0
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith('Label'): continue
    parts = line.split(',')
    result = process_line(parts)
    if result is None: continue

    can_id, td, proba = result
    msg += 1
    status = "ALERT!" if proba > 0.6 else "Normal"
    print(f"Msg {msg:6d} | ID={can_id:>6} | Δt={td:8.3f} | Proba={proba:6.3f} | {status}")
