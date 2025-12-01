from collections import defaultdict
import sys
import pandas as pd

# CONFIG — tune only this
FLOOD_THRESHOLD_MS = 8.0        # Anything faster than 8 ms = attack (adjustable)
MIN_MESSAGES_FOR_ALERT = 10     # Need at least X fast messages to trigger

# Per-ID state
last_time = {}
fast_count = defaultdict(int)
total_count = 0

print("SIMPLE IDS RUNNING — waiting for flooding attack...")
print(f"Threshold: {FLOOD_THRESHOLD_MS} ms | Need {MIN_MESSAGES_FOR_ALERT} fast messages")

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line or line.startswith('Label'): 
        continue
    
    parts = line.split(',')
    if len(parts) < 7: 
        continue

    try:
        timestamp = float(parts[1])      # in seconds → convert to ms
        can_id = parts[2].strip()
    except:
        continue

    current_ms = timestamp * 1000.0

    if can_id in last_time:
        delta_ms = current_ms - last_time[can_id]
        
        if delta_ms < FLOOD_THRESHOLD_MS:
            fast_count[can_id] += 1
        else:
            fast_count[can_id] = 0   # reset on normal gap

        if fast_count[can_id] >= MIN_MESSAGES_FOR_ALERT:
            print(f"ALERT! FLOODING DETECTED! ID={can_id} | Δt={delta_ms:6.3f} ms | "
                  f"Fast streak: {fast_count[can_id]}")
    else:
        print(f"First message from ID={can_id}")

    last_time[can_id] = current_ms
    total_count += 1

print(f"\nDONE — processed {total_count} messages")
