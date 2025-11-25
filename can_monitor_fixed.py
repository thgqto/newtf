# can_monitor_FINAL_WORKING.py
# FINAL – NO SYNTAX ERRORS – WORKS ON PI – YOUR ORIGINAL MODEL

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class CANMonitor:
    def __init__(self, model_path, threshold=0.7, input_source='stdin', buffer_size=5):
        self.threshold     = threshold
        self.buffer_size   = buffer_size
        self.input_source  = input_source
        self.buffers       = defaultdict(lambda: deque(maxlen=buffer_size))

        base = model_path.replace('syncan_ensemble_model.pkl', '')
        self.id_encoder = joblib.load(base + 'id_encoder.pkl')
        self.imputer    = joblib.load(base + 'imputer.pkl')
        self.scaler     = joblib.load(base + 'scaler.pkl')
        ensemble        = joblib.load(model_path)
        self.if_model   = ensemble['if']
        self.ocsvm_model = ensemble['ocsvm']
        
        self.signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
        self.msg_count = 0
        print(f"Model loaded | Threshold={threshold} | Source={input_source}")

    def preprocess_message(self, current_row):
        can_id = current_row['ID']
        buffer = self.buffers[can_id]
        buffer.append(current_row.copy())

        time_delta = 1.0
        if len(buffer) >= 2:
            time_delta = current_row['Time'] - buffer[-2]['Time']

        deltas = np.zeros(4)
        abs_deltas = np.zeros(4)
        if len(buffer) >= 2:
            prev = buffer[-2]
            for i, sig in enumerate(self.signal_cols):
                c = current_row[sig] if not pd.isna(current_row[sig]) else 0.0
                p = prev[sig] if not pd.isna(prev[sig]) else 0.0
                deltas[i] = c - p
                abs_deltas[i] = abs(deltas[i])

        roll_vars  = [0.0] * 4
        roll_means = [0.0] * 4
        if len(buffer) >= 2:
            for i, sig in enumerate(self.signal_cols):
                delta_hist = []
                for j in range(1, len(buffer)):
                    c = buffer[j][sig] if not pd.isna(buffer[j][sig]) else 0.0
                    p = buffer[j-1][sig] if not pd.isna(buffer[j-1][sig]) else 0.0
                    delta_hist.append(c - p)
                recent = delta_hist[-5:] or [0.0]
                roll_vars[i]  = np.var(recent)  if len(recent) > 1 else 0.0
                roll_means[i] = np.mean(recent)

        # FIXED: Proper 18-element list – no broken brackets
        features = np.array([
            time_delta,
            deltas[0], abs_deltas[0], roll_vars[0], roll_means[0],
            deltas[1], abs_deltas[1], roll_vars[1], roll_means[1],
            deltas[2], abs_deltas[2], roll_vars[2], roll_means[2],
            deltas[3], abs_deltas[3], roll_vars[3], roll_means[3],
            self.id_encoder.transform([can_id])[0]
        ], dtype=np.float32).reshape(1, -1)

        # Scale all except the last column (ID_encoded)
        features[:, :-1] = self.scaler.transform(features[:, :-1])
        return features

    def predict_anomaly(self, X):
        if_score  = self.if_model.decision_function(X)[0]
        svm_score = self.ocsvm_model.score_samples(X)[0]
        ensemble_score = (if_score + svm_score) / 2
        proba = -ensemble_score
        return proba, proba > self.threshold

    def run(self):
        if self.input_source == 'stdin':
            print("Reading CSV from stdin...")
            for line in sys.stdin:
                parts = line.strip().split(',')
                if len(parts) < 7: continue
                row = {
                    'Time': float(parts[1]),
                    'ID':   parts[2],
                    'Signal1': pd.to_numeric(parts[3], errors='coerce'),
                    'Signal2': pd.to_numeric(parts[4], errors='coerce'),
                    'Signal3': pd.to_numeric(parts[5], errors='coerce'),
                    'Signal4': pd.to_numeric(parts[6], errors='coerce')
                }
                X = self.preprocess_message(row)
                proba, alert = self.predict_anomaly(X)
                self.msg_count += 1
                status = "ALERT!" if alert else "Normal"
                print(f"Msg {self.msg_count:6d} | ID={row['ID']:>5} | Proba={proba:6.3f} | {status}")

        else:
            try:
                import can
            except ImportError:
                print("python-can not installed → pip install python-can")
                sys.exit(1)
            print(f"Live mode on {self.input_source}...")
            bus = can.interface.Bus(channel=self.input_source, bustype='socketcan')
            for msg in bus:
                row = {
                    'Time': msg.timestamp,
                    'ID': f"id{msg.arbitration_id}",
                    'Signal1': np.nan, 'Signal2': np.nan,
                    'Signal3': np.nan, 'Signal4': np.nan
                }
                if len(msg.data) >= 16:
                    payload = np.frombuffer(msg.data[:16], dtype=np.float32)
                    for i in range(min(4, len(payload))):
                        row[self.signal_cols[i]] = payload[i]

                X = self.preprocess_message(row)
                proba, alert = self.predict_anomaly(X)
                self.msg_count += 1
                status = "ALERT!" if alert else "Normal"
                print(f"Msg {self.msg_count:6d} | ID={row['ID']} | Proba={proba:6.3f} | {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='syncan_ensemble_model.pkl')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--input', default='stdin')
    args = parser.parse_args()

    monitor = CANMonitor(args.model, args.threshold, args.input)
    monitor.run()