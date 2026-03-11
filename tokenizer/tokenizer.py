#!/usr/bin/env python3
"""
Phantom Tokenizer — Event → Sequence Representation
Transforms raw JSONL events into structured token sequences
for transformer training.

Input:  data/events/YYYY-MM-DD.jsonl
Output: data/sequences/YYYY-MM-DD.txt
"""

import json
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

DATA_DIR  = Path(__file__).parent.parent / "data"
EVENTS_DIR    = DATA_DIR / "events"
SEQUENCES_DIR = DATA_DIR / "sequences"

# ── URL → label sémantique ─────────────────────────────
URL_PATTERNS = [
    (r"claude\.ai",                 "AI:CLAUDE"),
    (r"chat\.openai\.com",          "AI:CHATGPT"),
    (r"github\.com/([^/]+)/([^/]+)","CODE:GITHUB"),
    (r"github\.com",                "CODE:GITHUB"),
    (r"reddit\.com/r/(\w+)",        "SOCIAL:REDDIT"),
    (r"twitter\.com|x\.com",        "SOCIAL:X"),
    (r"linkedin\.com",              "SOCIAL:LINKEDIN"),
    (r"youtube\.com|youtu\.be",     "MEDIA:YOUTUBE"),
    (r"huggingface\.co",            "AI:HUGGINGFACE"),
    (r"arxiv\.org",                 "RESEARCH:ARXIV"),
    (r"notion\.so|notion\.com",     "PRODUCTIVITY:NOTION"),
    (r"mail\.google\.com|gmail",    "COMM:GMAIL"),
    (r"calendar\.google\.com",      "PRODUCTIVITY:CALENDAR"),
    (r"docs\.google\.com",          "PRODUCTIVITY:GDOCS"),
    (r"localhost|127\.0\.0\.1",     "DEV:LOCAL"),
    (r"substack\.com",              "WRITING:SUBSTACK"),
]

APP_PATTERNS = [
    (r"^(Cursor|Code|PyCharm|Xcode|Nova|Sublime)$", "APP:EDITOR"),
    (r"^(Terminal|iTerm|Warp|Alacritty)$",           "APP:TERMINAL"),
    (r"^(Google Chrome|Safari|Firefox|Arc|Brave)$",  "APP:BROWSER"),
    (r"^(Slack|Discord|Telegram|WhatsApp)$",         "APP:MESSAGING"),
    (r"^(Spotify|Music)$",                           "APP:MUSIC"),
    (r"^(Finder)$",                                  "APP:FILES"),
    (r"^(Notes|Obsidian|Bear|Notion)$",              "APP:NOTES"),
    (r"^(Mail|Mimestream|Spark)$",                   "APP:MAIL"),
    (r"^(zoom\.us|Teams|Meet)$",                     "APP:MEETING"),
]

# ── Durée → bucket ─────────────────────────────────────
def duration_bucket(secs: float) -> str:
    if secs < 10:   return "DUR:GLANCE"    # <10s  — coup d'oeil
    if secs < 60:   return "DUR:SHORT"     # <1min — court
    if secs < 300:  return "DUR:MEDIUM"    # <5min — moyen
    if secs < 1800: return "DUR:LONG"      # <30min — long
    return "DUR:DEEP"                       # >30min — deep work

# ── Heure → session ────────────────────────────────────
def time_session(hour: int) -> str:
    if 5  <= hour < 9:  return "SESSION:MORNING"
    if 9  <= hour < 12: return "SESSION:WORK_AM"
    if 12 <= hour < 14: return "SESSION:LUNCH"
    if 14 <= hour < 18: return "SESSION:WORK_PM"
    if 18 <= hour < 22: return "SESSION:EVENING"
    return "SESSION:NIGHT"

# ── Tokenise une URL ───────────────────────────────────
def tokenize_url(url: str) -> str:
    if not url:
        return ""
    for pattern, label in URL_PATTERNS:
        m = re.search(pattern, url)
        if m:
            # Pour GitHub, ajoute le repo si possible
            if label == "CODE:GITHUB" and len(m.groups()) >= 2:
                return f"CODE:GITHUB/{m.group(2).upper()}"
            # Pour Reddit, ajoute le subreddit
            if label == "SOCIAL:REDDIT" and m.lastindex:
                return f"SOCIAL:REDDIT/{m.group(1).upper()}"
            return label
    # Domaine inconnu → extrait le hostname
    try:
        host = urlparse(url).netloc.replace("www.", "")
        return f"WEB:{host.upper().split('.')[0]}"
    except Exception:
        return "WEB:UNKNOWN"

# ── Tokenise une app ───────────────────────────────────
def tokenize_app(app: str) -> str:
    if not app:
        return "APP:UNKNOWN"
    for pattern, label in APP_PATTERNS:
        if re.match(pattern, app):
            return label
    return f"APP:{app.upper().replace(' ', '_')}"

# ── Tokenise un fichier ────────────────────────────────
def tokenize_file(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    ext = p.suffix.lower()
    ext_map = {
        ".py": "FILE:PYTHON", ".js": "FILE:JS", ".ts": "FILE:TS",
        ".rs": "FILE:RUST",   ".cpp": "FILE:CPP", ".c": "FILE:C",
        ".md": "FILE:MARKDOWN", ".txt": "FILE:TEXT",
        ".json": "FILE:JSON", ".yaml": "FILE:YAML", ".toml": "FILE:TOML",
        ".sh": "FILE:SHELL",
    }
    return ext_map.get(ext, f"FILE:OTHER")

# ── Tokenise un event complet ──────────────────────────
def tokenize_event(event: dict) -> str:
    tokens = []

    # Contexte temporel
    hour    = event.get("hour", 0)
    weekday = event.get("weekday", "")
    tokens.append(time_session(hour))
    tokens.append(f"DAY:{weekday.upper()[:3]}")

    # App
    app = event.get("app", "")
    tokens.append(tokenize_app(app))

    # URL ou fichier (contexte)
    url  = event.get("url", "")
    file = event.get("file", "")
    url_token  = tokenize_url(url)
    file_token = tokenize_file(file)
    if url_token:
        tokens.append(url_token)
    if file_token:
        tokens.append(file_token)

    # Durée
    duration = event.get("duration", 0)
    tokens.append(duration_bucket(duration))

    return " ".join(t for t in tokens if t)

# ── Traite un fichier JSONL ────────────────────────────
def process_file(events_file: Path) -> int:
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = events_file.stem
    out_file = SEQUENCES_DIR / f"{date_str}.txt"

    sequences = []
    with open(events_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                seq   = tokenize_event(event)
                if seq:
                    sequences.append(seq)
            except json.JSONDecodeError:
                continue

    with open(out_file, "w") as f:
        f.write("\n".join(sequences))

    return len(sequences)

# ── Main ───────────────────────────────────────────────
def main():
    print("🔤 Phantom Tokenizer")
    files = sorted(EVENTS_DIR.glob("*.jsonl"))
    if not files:
        print("   No event files found in data/events/")
        return

    total = 0
    for f in files:
        count = process_file(f)
        print(f"   {f.name} → {count} sequences")
        total += count

    print(f"\n   ✓ {total} total sequences written to data/sequences/")
    print(f"   Sample output:")
    sample_file = SEQUENCES_DIR / f"{files[-1].stem}.txt"
    if sample_file.exists():
        with open(sample_file) as sf:
            for i, line in enumerate(sf):
                if i >= 5: break
                print(f"     {line.strip()}")

if __name__ == "__main__":
    main()
