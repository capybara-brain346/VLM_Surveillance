import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os
import re
import sqlite3
from datetime import datetime  # ✅ added for real-time timestamp

# ==== CONFIG ====
VIDEO_PATH = "Untitled.mp4"
FRAME_DIR = "frames"
FRAME_EVERY_N_SECONDS = 1
MODEL_NAME = "Salesforce/blip-image-captioning-base"
DB_PATH = "vlm_log.db"
TABLE_NAME = "caption_logs"
# ================

# High-priority keywords to trigger alerts
ALERT_KEYWORDS = ["loitering", "crash", "harm", "accident", "assault", "fight"]

# Make sure output dir exists
os.makedirs(FRAME_DIR, exist_ok=True)

# Load BLIP model
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

# Video setup
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps * FRAME_EVERY_N_SECONDS
logs = []
frame_id = 0
last_caption = None  # for change detection

def parse_caption(caption):
    vehicle = re.search(r"(car|truck|motorcycle|vehicle|van|bike)", caption)
    person = re.search(r"(man|woman|person|driver|individual)", caption)
    action = re.search(r"(parked|entering|exiting|driving|walking|standing|moving|seen)", caption)
    location = re.search(r"(garage|warehouse|street|hallway|parking|driveway|tunnel|gate)", caption)

    return {
        "vehicle": vehicle.group(0) if vehicle else None,
        "person": person.group(0) if person else None,
        "action": action.group(0) if action else None,
        "location": location.group(0) if location else None
    }

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # === ✅ Use real system time instead of video timestamp ===
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if frame_number % frame_interval == 0:
        frame_file = f"frame_{frame_id:04d}.jpg"
        frame_path = os.path.join(FRAME_DIR, frame_file)
        cv2.imwrite(frame_path, frame)

        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs)

        caption = processor.decode(out[0], skip_special_tokens=True)
        parsed = parse_caption(caption)

        # Skip if caption hasn't changed from previous frame
        if caption == last_caption:
            continue
        last_caption = caption

        # Check for alert keywords in caption
        alert = None
        for word in ALERT_KEYWORDS:
            if word in caption.lower():
                alert = f"🚨 Alert: {word.capitalize()} detected"
                break

        logs.append({
            "frame": frame_file,
            "timestamp": timestamp,  # ✅ use real time
            "caption": caption,
            "location": parsed["location"],
            "alert": alert
        })

        print(f"[{frame_id}] {caption} {'👉 ' + alert if alert else ''}")
        frame_id += 1

cap.release()

# Final log summary
print("\n🧾 Final Captions Log:")
for log in logs:
    print(f"{log['frame']} @ {log['timestamp']}: {log['caption']} {'👉 ' + log['alert'] if log['alert'] else ''}")

# ========================
# Export logs to SQLite DB
# ========================
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create table if not exists
cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame TEXT,
    timestamp TEXT,
    caption TEXT,
    location TEXT,
    alert TEXT
)
""")

# Insert data
for log in logs:
    cursor.execute(f"""
    INSERT INTO {TABLE_NAME} (frame, timestamp, caption, location, alert)
    VALUES (?, ?, ?, ?, ?)
    """, (log['frame'], log['timestamp'], log['caption'], log['location'], log['alert']))

conn.commit()
conn.close()

print(f"\n✅ Logs stored in database: {DB_PATH} (table: {TABLE_NAME})")


# Example at bottom of t3.py
def get_logs():
    return logs  # or load from DB if logs is not accessible

import sqlite3

def fetch_logs(db_path="vlm_log.db", table="caption_logs"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT frame, timestamp, caption, location, alert FROM {table}")
    results = cursor.fetchall()
    conn.close()
    return results
