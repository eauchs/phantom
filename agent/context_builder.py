#!/usr/bin/env python3
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PROFILE_DIR = DATA_DIR / "profile"
PROFILE_FILE = PROFILE_DIR / "profile.json"
ANSWERS_FILE = PROFILE_DIR / "answers.jsonl"
FEEDBACK_DIR = DATA_DIR / "feedback"
EVENTS_DIR = DATA_DIR / "events"

def get_last_tokens(n=10):
    from tokenizer.tokenizer_v2 import tokenize_event
    today = datetime.now().strftime("%Y-%m-%d")
    f = EVENTS_DIR / f"{today}.jsonl"
    if not f.exists(): return []
    
    tokens = []
    try:
        lines = f.read_text().strip().split("\n")
        for l in lines[-n:]:
            if l.strip():
                tokens.extend(tokenize_event(json.loads(l)).split())
    except:
        pass
    return tokens[-n:]

def get_recent_answers(n=3):
    if not ANSWERS_FILE.exists(): return []
    try:
        lines = ANSWERS_FILE.read_text().strip().split("\n")
        answers = []
        for l in lines[-n:]:
            if l.strip():
                answers.append(json.loads(l))
        return answers
    except:
        return []

def get_feedback_history(n=5):
    if not FEEDBACK_DIR.exists(): return []
    feedback = []
    files = sorted(FEEDBACK_DIR.glob("*.jsonl"))
    for f in reversed(files):
        try:
            lines = f.read_text().strip().split("\n")
            for l in reversed(lines):
                if l.strip():
                    feedback.append(json.loads(l))
                if len(feedback) >= n: break
        except:
            continue
        if len(feedback) >= n: break
    return feedback

def get_last_event():
    today = datetime.now().strftime("%Y-%m-%d")
    f = EVENTS_DIR / f"{today}.jsonl"
    if not f.exists(): return {}
    try:
        lines = f.read_text().strip().split("\n")
        if lines:
            return json.loads(lines[-1])
    except:
        pass
    return {}

from agent.session_state import load_state

def build_context() -> dict:
    profile = {}
    if PROFILE_FILE.exists():
        try:
            profile = json.loads(PROFILE_FILE.read_text())
        except:
            pass
            
    now = datetime.now()
    last_event = get_last_event()
    
    return {
        "profile": profile,
        "session": load_state(),
        "session_modifier": profile.get("current_session_modifier", {}).get("value", 0.0),
        "last_10_tokens": get_last_tokens(10),
        "window_title": last_event.get("window_title", ""),
        "current_hour": now.hour,
        "weekday": now.strftime("%A"),
        "recent_answers": get_recent_answers(3),
        "feedback_history": get_feedback_history(5)
    }
