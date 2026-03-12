#!/usr/bin/env python3
"""
Phantom Tokenizer v2 — handles all new behavioral signals
"""

import json
from pathlib import Path

ROOT          = Path(__file__).parent.parent
EVENTS_DIR    = ROOT / "data" / "events"
SEQUENCES_DIR = ROOT / "data" / "sequences_v2"

# ── Bucketing functions ─────────────────────────────────

def session_token(hour):
    if 5  <= hour < 9:  return "SESSION:MORNING"
    if 9  <= hour < 12: return "SESSION:WORK_AM"
    if 12 <= hour < 14: return "SESSION:LUNCH"
    if 14 <= hour < 18: return "SESSION:WORK_PM"
    if 18 <= hour < 22: return "SESSION:EVENING"
    return "SESSION:NIGHT"

def day_token(weekday):
    return f"DAY:{weekday[:3].upper()}"

def app_token(app):
    mapping = {
        "Google Chrome": "APP:BROWSER", "Arc": "APP:BROWSER",
        "Safari": "APP:BROWSER", "Firefox": "APP:BROWSER",
        "Terminal": "APP:TERMINAL", "iTerm2": "APP:TERMINAL",
        "Cursor": "APP:EDITOR", "Visual Studio Code": "APP:EDITOR",
        "PyCharm": "APP:EDITOR", "Xcode": "APP:EDITOR",
        "Slack": "APP:COMM", "WhatsApp": "APP:COMM",
        "Telegram": "APP:COMM", "Messages": "APP:COMM",
        "Mail": "APP:MAIL",
        "Notion": "APP:NOTES", "Obsidian": "APP:NOTES",
        "Spotify": "APP:MEDIA", "Music": "APP:MEDIA",
        "Figma": "APP:DESIGN", "Sketch": "APP:DESIGN",
        "Preview": "APP:VIEWER", "Finder": "APP:FINDER",
    }
    return mapping.get(app, f"APP:OTHER")

def url_token(url):
    if not url:
        return None
    u = url.lower()
    if "claude.ai"      in u: return "WEB:CLAUDE"
    if "chat.openai"    in u: return "WEB:CHATGPT"
    if "github.com"     in u: return "WEB:GITHUB"
    if "localhost"      in u or "127.0.0.1" in u: return "WEB:LOCALHOST"
    if "stackoverflow"  in u: return "WEB:STACKOVERFLOW"
    if "reddit.com"     in u: return "WEB:REDDIT"
    if "arxiv.org"      in u: return "WEB:ARXIV"
    if "youtube.com"    in u: return "WEB:YOUTUBE"
    if "twitter.com"    in u or "x.com" in u: return "WEB:TWITTER"
    if "linkedin.com"   in u: return "WEB:LINKEDIN"
    if "notion.so"      in u: return "WEB:NOTION"
    if "docs.google"    in u: return "WEB:GDOCS"
    if "mail.google"    in u: return "WEB:GMAIL"
    if "huggingface"    in u: return "WEB:HUGGINGFACE"
    if "deezer"         in u or "spotify" in u: return "WEB:MUSIC"
    if "perplexity"     in u: return "WEB:PERPLEXITY"
    return "WEB:OTHER"

def duration_token(d):
    if d < 10:  return "DUR:GLANCE"
    if d < 60:  return "DUR:SHORT"
    if d < 300: return "DUR:MEDIUM"
    if d < 1800:return "DUR:LONG"
    return "DUR:DEEP"

def wpm_token(wpm):
    if wpm == 0:   return "TYPING:IDLE"
    if wpm < 20:   return "TYPING:SLOW"
    if wpm < 50:   return "TYPING:MEDIUM"
    if wpm < 80:   return "TYPING:FAST"
    return "TYPING:BURST"

def error_token(backspace_rate):
    if backspace_rate > 15: return "ERROR:HIGH"
    if backspace_rate > 5:  return "ERROR:MED"
    if backspace_rate > 0:  return "ERROR:LOW"
    return None

def mouse_token(distance, idle_s):
    if idle_s > 120:        return "MOUSE:IDLE"
    if distance < 100:      return "MOUSE:STILL"
    if distance < 1000:     return "MOUSE:LIGHT"
    return "MOUSE:ACTIVE"

def tabs_token(count):
    if count == 0:  return None
    if count <= 3:  return "TABS:FEW"
    if count <= 10: return "TABS:MEDIUM"
    return "TABS:MANY"

def windows_token(count):
    if count <= 1:  return "WIN:SINGLE"
    if count <= 3:  return "WIN:MULTI"
    return "WIN:SCATTERED"

def cpu_token(cpu):
    if cpu > 70:    return "CPU:HIGH"
    if cpu > 30:    return "CPU:MED"
    return "CPU:LOW"

def focus_token(score):
    if score >= 70: return "FOCUS:DEEP"
    if score >= 40: return "FOCUS:MEDIUM"
    return "FOCUS:SHALLOW"

def switch_token(rate):
    if rate >= 3:   return "SWITCH:FAST"
    if rate >= 1:   return "SWITCH:MED"
    return "SWITCH:STABLE"

def net_token(active):
    return "NET:ACTIVE" if active else "NET:IDLE"

def action_tokens(action_str):
    if not action_str:
        return []
    
    tokens = []
    actions = action_str.split(",")
    
    mapping = {
        "ACT:OPEN_CHROME": "ACT:OPEN_BROWSER", "ACT:OPEN_ARC": "ACT:OPEN_BROWSER",
        "ACT:OPEN_SAFARI": "ACT:OPEN_BROWSER", "ACT:OPEN_FIREFOX": "ACT:OPEN_BROWSER",
        "ACT:OPEN_TERMINAL": "ACT:OPEN_TERMINAL", "ACT:OPEN_ITERM2": "ACT:OPEN_TERMINAL",
        "ACT:OPEN_CURSOR": "ACT:OPEN_EDITOR", "ACT:OPEN_VISUAL_STUDIO_CODE": "ACT:OPEN_EDITOR",
        "ACT:OPEN_XCODE": "ACT:OPEN_EDITOR",
        "ACT:OPEN_SLACK": "ACT:OPEN_COMM", "ACT:OPEN_WHATSAPP": "ACT:OPEN_COMM",
        "ACT:OPEN_TELEGRAM": "ACT:OPEN_COMM", "ACT:OPEN_MESSAGES": "ACT:OPEN_COMM",
    }
    
    for a in actions:
        if a in mapping:
            tokens.append(mapping[a])
        else:
            # Fallback for GIT and FILE actions which are already in correct format or need passthrough
            if a.startswith("ACT:GIT_") or a.startswith("ACT:FILE_"):
                tokens.append(a)
            elif a.startswith("ACT:OPEN_"):
                # Generic app open if not in mapping
                tokens.append(a)
    
    return tokens

# ── Tokenize one event ──────────────────────────────────
def tokenize_event(ev):
    tokens = []
    
    # ── Context ──
    tokens.append(session_token(ev.get("hour", 12)))
    tokens.append(day_token(ev.get("weekday", "Monday")))
    tokens.append(app_token(ev.get("app", "")))
    
    u = url_token(ev.get("url", ""))
    if u: tokens.append(u)
    
    tokens.append(duration_token(ev.get("duration", 0)))
    
    # ── Behavior ──
    wpm = ev.get("wpm")
    if wpm is not None:
        tokens.append(wpm_token(wpm))
    
    br = ev.get("backspace_rate")
    if br is not None:
        err = error_token(br)
        if err: tokens.append(err)
    
    dist = ev.get("mouse_distance_px", 0)
    idle = ev.get("mouse_idle_s", 0)
    tokens.append(mouse_token(dist, idle))
    
    tabs = ev.get("tab_count_total") or ev.get("tab_count", 0)
    t = tabs_token(tabs)
    if t: tokens.append(t)
    
    wins = ev.get("window_count", 1)
    tokens.append(windows_token(wins))
    
    cpu = ev.get("cpu_percent")
    if cpu is not None:
        tokens.append(cpu_token(cpu))
    
    focus = ev.get("focus_score")
    if focus is not None:
        tokens.append(focus_token(focus))
    
    rate = ev.get("app_switch_rate")
    if rate is not None:
        tokens.append(switch_token(rate))
    
    net = ev.get("net_active")
    if net is not None:
        tokens.append(net_token(net))

    # ── Actions ──
    act = ev.get("action")
    if act:
        tokens.extend(action_tokens(act))
    
    return " ".join(t for t in tokens if t)

# ── Main ────────────────────────────────────────────────
def main():
    print("🔤 Phantom Tokenizer v2")
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    
    files = sorted(EVENTS_DIR.glob("*.jsonl"))
    if not files:
        print("   No event files found.")
        return
    
    total = 0
    for f in files:
        seqs = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    ev  = json.loads(line)
                    seq = tokenize_event(ev)
                    if seq: seqs.append(seq)
                except Exception:
                    continue
        
        out = SEQUENCES_DIR / f"{f.stem}.txt"
        with open(out, "w") as fh:
            fh.write("\n".join(seqs))
        
        print(f"   {f.name} → {len(seqs)} sequences")
        total += len(seqs)
    
    print(f"\n   ✓ {total} sequences → data/sequences_v2/")
    print(f"\n   Sample:")
    sample = SEQUENCES_DIR / f"{files[-1].stem}.txt"
    if sample.exists():
        with open(sample) as fh:
            for i, line in enumerate(fh):
                if i >= 5: break
                print(f"     {line.strip()}")

if __name__ == "__main__":
    main()
