#!/usr/bin/env python3
"""
Phantom Daemon — Behavioral Event Collector
Observes: active app, browser URL, active file, clipboard
Logs to: data/events/YYYY-MM-DD.jsonl
"""

import json
import time
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────
POLL_INTERVAL   = 3       # secondes entre chaque poll
DATA_DIR        = Path(__file__).parent.parent / "data" / "events"
LOG_DIR         = Path(__file__).parent.parent / "logs"
MIN_DURATION    = 3       # durée minimale (sec) pour logguer un event
BROWSERS        = {"Safari", "Google Chrome", "Firefox", "Arc", "Brave Browser"}

# ── Helpers AppleScript ───────────────────────────────
def run_applescript(script: str) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip()
    except Exception:
        return ""

def get_active_app() -> str:
    return run_applescript(
        'tell application "System Events" to get name of first process whose frontmost is true'
    )

def get_browser_url(app: str) -> str:
    if app == "Safari":
        return run_applescript(
            'tell application "Safari" to get URL of current tab of front window'
        )
    elif app in {"Google Chrome", "Arc", "Brave Browser"}:
        return run_applescript(
            f'tell application "{app}" to get URL of active tab of front window'
        )
    return ""

def get_active_file(app: str) -> str:
    editors = {"Cursor", "Code", "Xcode", "PyCharm", "Nova", "Sublime Text", "BBEdit"}
    if app in editors:
        result = run_applescript(
            f'tell application "{app}" to get path of document 1'
        )
        return result
    return ""

def get_clipboard_hash() -> str:
    """Hash du clipboard pour détecter les changements sans stocker le contenu."""
    try:
        result = subprocess.run(
            ["pbpaste"], capture_output=True, text=True, timeout=1
        )
        content = result.stdout[:200]  # premiers 200 chars seulement
        return hashlib.md5(content.encode()).hexdigest()[:8] if content else ""
    except Exception:
        return ""

# ── Logger ────────────────────────────────────────────
def log_event(event: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = DATA_DIR / f"{date_str}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

# ── Main loop ─────────────────────────────────────────
def main():
    print(f"👻 Phantom daemon started — polling every {POLL_INTERVAL}s")
    print(f"   Logging to: {DATA_DIR}")
    print("   Press Ctrl+C to stop\n")

    current_event = None
    current_start = None

    while True:
        try:
            now = datetime.now()
            app = get_active_app()

            # Infos contextuelles selon l'app
            url        = get_browser_url(app) if app in BROWSERS else ""
            file_path  = get_active_file(app)
            clip_hash  = get_clipboard_hash()

            # Résumé de l'état courant
            state = {
                "app":       app,
                "url":       url,
                "file":      file_path,
                "clip_hash": clip_hash,
            }

            # Nouveau contexte détecté ?
            state_key = f"{app}|{url}|{file_path}"

            if current_event is None:
                current_event = state
                current_start = now

            elif state_key != f"{current_event['app']}|{current_event['url']}|{current_event['file']}":
                # Contexte changé → log l'event précédent si assez long
                duration = (now - current_start).total_seconds()
                if duration >= MIN_DURATION:
                    event = {
                        "ts_start":  current_start.isoformat(),
                        "ts_end":    now.isoformat(),
                        "duration":  round(duration, 1),
                        "hour":      current_start.hour,
                        "weekday":   current_start.strftime("%A"),
                        **current_event,
                    }
                    log_event(event)
                    print(f"[{now.strftime('%H:%M:%S')}] {app:<20} {round(duration)}s  {url or file_path or ''}")

                current_event = state
                current_start = now

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n👻 Phantom stopped.")
            break
        except Exception as e:
            with open(LOG_DIR / "errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} ERROR: {e}\n")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
