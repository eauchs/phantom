#!/usr/bin/env python3
"""
Phantom v2.1 — Two-Tower Orchestrator
1. Run feature_extractor.py
2. Train two_tower.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import trainer.feature_extractor as fe
import trainer.two_tower as tt

def main():
    print("🛸 Phantom v2.1 — Training Pipeline\n")
    
    # 1. Feature Extraction
    fe.extract_features()
    
    # 2. Two-Tower Training
    tt.train()
    
    print("\n✓ Pipeline complete. Agent will now use the new model.")

if __name__ == "__main__":
    main()
