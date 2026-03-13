#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
EVENTS_DIR = ROOT / "data" / "events"
FEATURES_DIR = ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLIPBOARD_TYPES = ["empty", "text", "url", "image", "code"]

def get_id(mapping, key):
    if key not in mapping:
        mapping[key] = len(mapping) + 1
    return mapping[key]

def extract_features():
    print("🚀 Extracting features from raw events...")
    
    app_mapping = {}
    url_mapping = {}
    
    files = sorted(EVENTS_DIR.glob("*.jsonl"))
    if not files:
        print("❌ No event files found.")
        return

    for f in files:
        date_str = f.stem
        features = []
        labels = [] # We'll store actions if present, though mainly for two_tower.py
        
        with open(f, "r") as json_file:
            events = [json.loads(line) for line in json_file]
        
        if not events:
            continue

        for e in events:
            # Normalized features (dim 24)
            vec = np.zeros(24, dtype=np.float32)
            
            vec[0] = e.get("hour", 0) / 24.0
            try:
                wd = e.get("weekday", "Monday")
                vec[1] = WEEKDAYS.index(wd) / 7.0
            except:
                vec[1] = 0.0
                
            vec[2] = min(e.get("wpm", 0) / 200.0, 1.0)
            vec[3] = min(e.get("backspace_rate", 0) / 30.0, 1.0)
            vec[4] = min(e.get("modifier_rate", 0) / 30.0, 1.0)
            vec[5] = min(e.get("mouse_distance_px", 0) / 5000.0, 1.0)
            vec[6] = min(e.get("mouse_clicks", 0) / 20.0, 1.0)
            vec[7] = min(e.get("scroll_events", 0) / 100.0, 1.0)
            vec[8] = min(e.get("mouse_idle_s", 0) / 300.0, 1.0)
            vec[9] = min(e.get("tab_count", 0) / 20.0, 1.0)
            vec[10] = min(e.get("window_count", 0) / 10.0, 1.0)
            vec[11] = min(e.get("cpu_percent", 0) / 100.0, 1.0)
            vec[12] = min(e.get("ram_percent", 0) / 100.0, 1.0)
            
            bat = e.get("battery") or {}
            vec[13] = (e.get("bat_percent") or bat.get("percent", 100)) / 100.0
            vec[14] = 1.0 if (e.get("bat_charging") or bat.get("plugged", True)) else 0.0
            
            vec[15] = e.get("focus_score", 50) / 100.0
            vec[16] = min(e.get("app_switch_rate", 0) / 10.0, 1.0)
            vec[17] = 1.0 if e.get("net_active") else 0.0
            vec[18] = 1.0 if e.get("dark_mode") else 0.0
            vec[19] = min(e.get("duration", 0) / 3600.0, 1.0)
            
            vec[20] = get_id(app_mapping, e.get("app", "unknown"))
            
            # URL pattern (simple domain extraction or hash)
            url = e.get("url", "")
            domain = url.split("//")[-1].split("/")[0] if "//" in url else url
            vec[21] = get_id(url_mapping, domain)
            
            clip = e.get("clipboard_type", "empty")
            try:
                vec[22] = CLIPBOARD_TYPES.index(clip)
            except:
                vec[22] = 0
            
            vec[23] = min(e.get("screens", 1) / 3.0, 1.0)
            
            features.append(vec)
            
        # Save as npy
        np_features = np.array(features, dtype=np.float32)
        np.save(FEATURES_DIR / f"{date_str}.npy", np_features)
        print(f"   Saved {len(features)} events to {date_str}.npy")

    # Save mappings for agent to use
    with open(FEATURES_DIR / "mappings.json", "w") as m:
        json.dump({
            "apps": app_mapping,
            "urls": url_mapping,
            "clip": CLIPBOARD_TYPES
        }, m, indent=2)

if __name__ == "__main__":
    extract_features()
