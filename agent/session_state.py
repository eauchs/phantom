#!/usr/bin/env python3
import json
import time
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).parent.parent
SESSION_DIR = ROOT / "data" / "session"
STATE_FILE = SESSION_DIR / "state.json"

def load_state() -> dict:
    if not SESSION_DIR.exists():
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
    
    if not STATE_FILE.exists():
        initial_state = {
            "last_started": None,
            "last_stopped": None,
            "last_focus_peak": None,
            "current_project": "UNKNOWN",
            "last_action_accepted": None,
            "last_action_rejected": None,
            "session_count_today": 0,
            "total_focus_minutes_today": 0,
            "peak_hour_today": None,
            "uptime_start": None,
            "last_update_date": str(date.today())
        }
        save_state(initial_state)
        return initial_state
    
    try:
        state = json.loads(STATE_FILE.read_text())
        # Reset daily counters if date changed
        if state.get("last_update_date") != str(date.today()):
            state["session_count_today"] = 0
            state["total_focus_minutes_today"] = 0
            state["last_update_date"] = str(date.today())
            save_state(state)
        return state
    except:
        return {}

def save_state(updates: dict):
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except: pass
    
    state.update(updates)
    STATE_FILE.write_text(json.dumps(state, indent=2))

def update_focus_peak(score, app, tokens):
    state = load_state()
    current_peak = state.get("last_focus_peak")
    
    if not current_peak or score >= current_peak.get("score", 0):
        save_state({
            "last_focus_peak": {
                "ts": datetime.now().isoformat(),
                "score": score,
                "app": app,
                "tokens": tokens
            }
        })
        # Check for peak hour
        hour = datetime.now().hour
        save_state({"peak_hour_today": hour})

def update_project(tokens: list):
    for t in tokens:
        if t.startswith("PROJECT:"):
            save_state({"current_project": t})
            break

def on_startup():
    state = load_state()
    now = datetime.now().isoformat()
    save_state({
        "last_started": now,
        "uptime_start": now,
        "session_count_today": state.get("session_count_today", 0) + 1
    })
    print(f"   ✓ Session state initialized. Today's sessions: {state.get('session_count_today', 0) + 1}")

def on_shutdown():
    state = load_state()
    if not state.get("uptime_start"): return
    
    start_dt = datetime.fromisoformat(state["uptime_start"])
    end_dt = datetime.now()
    duration_mins = (end_dt - start_dt).total_seconds() / 60
    
    save_state({
        "last_stopped": end_dt.isoformat(),
        "total_focus_minutes_today": state.get("total_focus_minutes_today", 0) + duration_mins,
        "uptime_start": None
    })
    print(f"   ✓ Session state saved. Focus added: {duration_mins:.1f}m")
