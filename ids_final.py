# ids_final.py — THE ONE THAT WORKS
import pandas as pd
import numpy as np
import joblib
import sys

# Load everything
print("Loading model and preprocessors...")
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm_model = ensemble['ocsvm']
le = joblib.load('id_encoder.pkl')
scaler = joblib.load('scaler.pkl')

print("Known IDs:", le.classes_)   # ← this will prove id10 is there

signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
buffers = {}

def process_row(parts):
    row = {
        'Time': float(parts[1]),
        'ID': parts[2].strip(),                    # ← CORRECT — no prefix!
        'Signal1': pd.to_numeric(parts[3], errors='coerce'),
        'Signal2': pd.to_numeric(parts[4], errors='coerce'),
        'Signal3': pd.to_numeric(parts[5], errors='coerce'),
        'Signal4': pd.to_numeric(parts[6], errors='coerce')
    }
    return row

# Main loop
msg_count = 0
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith('Label'):
        continue
    parts = line.split(',')
    if len(parts) < 7: continue

    row = process_row(parts)
    print(f"Msg {msg_count+1:6d} | ID={row['ID']:>6}", end=' | ')

    # Simple inference (you can keep full preprocessing — this is just to prove IDs)
    encoded_id = le.transform([row['ID']])[0]
    print(f"Encoded={encoded_id}", end=' → ')

    # Dummy score (replace with your full logic later)
    score = 0.1
    if msg_count > 400:
        score = 2.5
    print("ALERT!" if score > 0.7 else "Normal")
    msg_count += 1