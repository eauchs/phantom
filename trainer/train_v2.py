#!/usr/bin/env python3
"""
Phantom v2.1 — Two-Tower Orchestrator
1. Run feature_extractor.py
2. Train two_tower.py with RLHF feedback
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import trainer.feature_extractor as fe
import trainer.two_tower as tt

FEEDBACK_DIR = ROOT / "data" / "feedback"

def load_feedback():
    all_feedback = []
    n_accepted = 0
    n_rejected = 0
    
    if not FEEDBACK_DIR.exists():
        return [], 0, 0
        
    for f in FEEDBACK_DIR.glob("*.jsonl"):
        try:
            lines = f.read_text().strip().split("\n")
            for line in lines:
                if not line: continue
                entry = json.loads(line)
                all_feedback.append(entry)
                if entry.get("accepted"):
                    n_accepted += 1
                else:
                    n_rejected += 1
        except:
            continue
    return all_feedback, n_accepted, n_rejected

def main():
    print("🛸 Phantom v2.1 — Training Pipeline\n")
    
    # 1. Feature Extraction
    fe.extract_features()
    
    # 2. Feedback Integration
    feedback, n_pos, n_neg = load_feedback()
    print(f"   Feedback incorporated: {n_pos} positive, {n_neg} negative")
    
    # 3. Two-Tower Training
    tt.train(feedback=feedback)
    
    print("\n✓ Pipeline complete. Agent will now use the new model.")

if __name__ == "__main__":
    main()
