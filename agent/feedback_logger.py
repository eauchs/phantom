#!/usr/bin/env python3
import json
import time
from datetime import datetime
from pathlib import Path

ROOT         = Path(__file__).parent.parent
FEEDBACK_DIR = ROOT / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

def log_feedback(action: str, accepted: bool, context_tokens: list):
    """
    Log user accept/reject for a predicted action.
    Used for future reward signal / fine-tuning.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = FEEDBACK_DIR / f"{date_str}.jsonl"
    
    entry = {
        "ts":             datetime.now().isoformat(),
        "action":         action,
        "accepted":       accepted,
        "context_tokens": context_tokens
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    status = "✅ ACCEPTED" if accepted else "❌ REJECTED"
    print(f"[FEEDBACK] {status}: {action}")
