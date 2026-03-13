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
import requests
import time
from agent.context_builder import build_context

FEEDBACK_DIR = ROOT / "data" / "feedback"
LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

def get_session_modifier():
    context = build_context()
    profile = context.get("profile", {})
    mod = profile.get("current_session_modifier", {})
    if mod and time.time() - mod.get("ts", 0) < 1800: # 30 mins
        return mod.get("value", 0.0), mod.get("ts", 0)
    return 0.0, 0

def get_qwen_reward(entry, session_mod=0.0):
    """
    Call Qwen to score a feedback event.
    """
    system_prompt = (
        "You are a reward model for a personal AI OS. "
        "Analyze the behavioral context and return ONLY a JSON: "
        "{\"reward\": float between -1.0 and 1.0, \"reason\": str}"
    )
    
    # Use centralized context
    context = build_context()
    # Override with feedback-specific entry
    context["feedback_entry"] = {
        "action": entry.get("action"),
        "accepted": entry.get("accepted"),
        "ts": entry.get("ts"),
        "context_tokens": entry.get("context_tokens", [])[-10:]
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context)}
        ],
        "max_tokens": 500,
        "temperature": 0.0
    }
    
    try:
        r = requests.post(LLAMA_URL, json=payload, timeout=30)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"].strip()
            # Strip <think>...</think> tags
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            
            # Find the JSON part
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                reward = data.get("reward", 0.0)
                
                # Apply session modifier (clamped)
                reward = max(-1.0, min(1.0, reward + session_mod))
                
                return reward, data.get("reason", "No reason provided")
    except Exception as e:
        print(f"      ⚠️ Reward model error: {e}")
    
    # Fallback to binary + session_mod
    fallback_reward = 1.0 if entry.get("accepted") else -1.0
    fallback_reward = max(-1.0, min(1.0, fallback_reward + session_mod))
    return fallback_reward, "Fallback binary reward"

def load_feedback():
    all_feedback = []
    n_accepted = 0
    n_rejected = 0
    
    if not FEEDBACK_DIR.exists():
        return [], 0, 0
    
    session_mod, mod_ts = get_session_modifier()
    if session_mod != 0.0:
        print(f"   ℹ️ Session reward modifier active: {session_mod:+.2f}")
        
    print("   Scoring feedback with Qwen reward model...")
    files = sorted(FEEDBACK_DIR.glob("*.jsonl"))
    for f in files:
        try:
            lines = f.read_text().strip().split("\n")
            for line in lines:
                if not line: continue
                entry = json.loads(line)
                
                # Apply session_mod only if entry is newer than the modifier or within reasonable window
                # For simplicity, we apply to all if session_mod is active
                reward, reason = get_qwen_reward(entry, session_mod=session_mod)
                entry["reward"] = reward
                entry["reward_reason"] = reason
                
                all_feedback.append(entry)
                if entry.get("accepted"):
                    n_accepted += 1
                else:
                    n_rejected += 1
                
                print(f"      Action: {entry.get('action'):<15} | Accepted: {entry.get('accepted'):<5} | Reward: {reward:+.2f} | {reason}")
        except Exception as e:
            print(f"      Error processing {f.name}: {e}")
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
