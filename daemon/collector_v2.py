#!/usr/bin/env python3
"""
Phantom Collector v2 — full behavioral sensor suite
Captures: keyboard, mouse, windows, tabs, system metrics
"""

import json
import time
import threading
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from collections import deque

import psutil
import os
import requests

try:
    from pynput import keyboard as kb, mouse as ms
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False
    print("⚠️  pynput not found — keyboard/mouse disabled")

try:
    import Quartz
    QUARTZ_OK = True
except ImportError:
    QUARTZ_OK = False
    print("⚠️  Quartz not found — window layout disabled")

# ── Config ─────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "events"
LOG_DIR     = ROOT / "logs"
CACHE_FILE  = ROOT / "data" / "tokens_cache.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL = 5
LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

# ── Contextual Tokenizer ───────────────────────────────
TOKENS_CACHE = {}
if CACHE_FILE.exists():
    try:
        TOKENS_CACHE = json.loads(CACHE_FILE.read_text())
    except:
        pass

def get_contextual_token(app_name, url):
    cache_key = f"{app_name}|{url}"
    if cache_key in TOKENS_CACHE:
        return TOKENS_CACHE[cache_key]

    system_prompt = (
        "You are a behavioral tokenizer. Given this app/url, "
        "return ONE token in format CATEGORY:NAME (uppercase, no spaces). "
        "Examples: WEB:NOTION, APP:FIGMA, WEB:ARXIV "
        "Return ONLY the token, nothing else."
    )
    user_prompt = f"app={app_name} url={url}"
    
    # ── Guards ──
    if not app_name.strip() and not url.strip():
        return None
    
    # Avoid too short/useless queries
    if len(user_prompt.replace("app=","").replace("url=","").strip()) < 3:
        return None

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 20,
        "temperature": 0.0,
        "enable_thinking": False
    }
    
    # ── Guard: verify user query is not empty ──
    user_msg = next((m["content"] for m in payload["messages"] if m["role"] == "user"), None)
    if not user_msg or not user_msg.strip():
        return None

    print(f"[LLM DEBUG] get_contextual_token sending {len(payload['messages'])} messages")
    for m in payload["messages"]:
        print(f"  role={m['role']} content_len={len(str(m.get('content','')))}")

    try:
        r = requests.post(LLAMA_URL, json=payload, timeout=30)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            
            if not content or not content.strip():
                return None

            token = content.split()[0].upper() if content else None
            if token and ":" in token:
                TOKENS_CACHE[cache_key] = token
                CACHE_FILE.write_text(json.dumps(TOKENS_CACHE, indent=2))
                return token
    except:
        pass
    return None

# ── State partagé thread-safe ───────────────────────────

MIN_DURATION  = 5       # durée min pour logguer un event
BROWSERS      = {"Google Chrome", "Safari", "Arc", "Firefox", "Brave Browser"}

# ── State partagé thread-safe ───────────────────────────
class BehaviorState:
    def __init__(self):
        self._lock = threading.Lock()
        
        # Keyboard
        self.key_times      = deque(maxlen=200)   # timestamps des frappes
        self.backspaces     = 0
        self.total_keys     = 0
        self.last_key_time  = None
        self.key_intervals  = deque(maxlen=100)   # inter-key delays (ms)
        self.word_chars     = 0                   # chars depuis dernier espace
        self.modifier_count = 0                   # cmd/ctrl/alt usage
        
        # Mouse
        self.mouse_positions  = deque(maxlen=50)
        self.mouse_clicks     = 0
        self.scroll_events    = 0
        self.last_mouse_time  = None
        self.mouse_idle_since = time.time()
        self.pending_actions  = []
        
    def record_action(self, action):
        with self._lock:
            if action not in self.pending_actions:
                self.pending_actions.append(action)
    
    def record_key(self, key_name):
        with self._lock:
            now = time.time()
            self.key_times.append(now)
            self.total_keys += 1
            
            if key_name in ('backspace', 'delete'):
                self.backspaces += 1
            
            if key_name in ('space', 'return', 'tab'):
                self.word_chars = 0
            else:
                self.word_chars += 1
            
            if key_name in ('cmd', 'ctrl', 'alt', 'shift'):
                self.modifier_count += 1
            
            if self.last_key_time:
                interval_ms = (now - self.last_key_time) * 1000
                if interval_ms < 2000:  # ignore pauses > 2s
                    self.key_intervals.append(interval_ms)
            self.last_key_time = now
    
    def record_mouse_move(self, x, y):
        with self._lock:
            self.mouse_positions.append((x, y, time.time()))
            self.mouse_idle_since = time.time()
    
    def record_mouse_click(self):
        with self._lock:
            self.mouse_clicks += 1
            self.mouse_idle_since = time.time()
    
    def record_scroll(self):
        with self._lock:
            self.scroll_events += 1
    
    def snapshot(self):
        """Retourne un snapshot et remet à zéro les compteurs delta"""
        with self._lock:
            now = time.time()
            
            # WPM — frappes dans les 30 dernières secondes
            cutoff = now - 30
            recent_keys = sum(1 for t in self.key_times if t > cutoff)
            wpm = (recent_keys / 5) * 2  # approx: 5 chars/mot, fenêtre 30s → ×2 pour /min
            
            # Inter-key velocity (médiane en ms)
            if self.key_intervals:
                sorted_intervals = sorted(self.key_intervals)
                n = len(sorted_intervals)
                key_velocity_ms = sorted_intervals[n // 2]
            else:
                key_velocity_ms = 0
            
            # Backspace rate
            backspace_rate = (self.backspaces / max(self.total_keys, 1)) * 100
            
            # Mouse distance depuis dernier snapshot
            positions = list(self.mouse_positions)
            mouse_distance = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                mouse_distance += (dx**2 + dy**2) ** 0.5
            
            # Mouse idle (secondes depuis dernier mouvement)
            mouse_idle_s = now - self.mouse_idle_since
            
            snap = {
                "wpm":              round(wpm, 1),
                "key_velocity_ms":  round(key_velocity_ms, 1),
                "backspace_rate":   round(backspace_rate, 1),
                "total_keys_delta": self.total_keys,
                "modifier_rate":    round((self.modifier_count / max(self.total_keys, 1)) * 100, 1),
                "mouse_distance_px": round(mouse_distance),
                "mouse_clicks":     self.mouse_clicks,
                "scroll_events":    self.scroll_events,
                "mouse_idle_s":      round(mouse_idle_s),
                "action":            ",".join(self.pending_actions) if self.pending_actions else None,
            }
            
            # Reset deltas
            self.backspaces     = 0
            self.total_keys     = 0
            self.modifier_count = 0
            self.mouse_positions.clear()
            self.mouse_clicks   = 0
            self.scroll_events  = 0
            self.pending_actions = []
            
            return snap

class ActionSensor:
    def __init__(self):
        self.seen_apps = set()
        self.last_dir_state = {} 
        self.git_keywords = {"push", "commit", "pull"}

    def detect(self, current_app, current_file):
        actions = []
        try:
            # 1. App launches (check if current active app is new this session)
            if current_app and current_app not in self.seen_apps:
                token_app = current_app.replace(" ", "_").replace(".app", "").upper()
                actions.append(f"ACT:OPEN_{token_app}")
                self.seen_apps.add(current_app)
            
            # 2. Git operations (scan processes for git commands)
            for p in psutil.process_iter(['name', 'cmdline']):
                if p.info['name'] == 'git' and p.info['cmdline']:
                    cmd = " ".join(p.info['cmdline'])
                    for kw in self.git_keywords:
                        if kw in cmd:
                            actions.append(f"ACT:GIT_{kw.upper()}")
            
            # 3. File changes
            if current_file:
                # Optimized: only check top level of agent for now
                dir_path = str(ROOT)
                files = set()
                try:
                    files = set(os.listdir(ROOT))
                    if dir_path in self.last_dir_state:
                        old_files = self.last_dir_state[dir_path]
                        if len(files) > len(old_files): actions.append("ACT:FILE_CREATE")
                        if len(files) < len(old_files): actions.append("ACT:FILE_DELETE")
                    self.last_dir_state[dir_path] = files
                except:
                    pass
        except Exception:
            pass
        return list(set(actions))

STATE = BehaviorState()

# ── Listeners pynput ────────────────────────────────────
def start_listeners():
    if not PYNPUT_OK:
        return
    
    def on_press(key):
        try:
            name = key.char if hasattr(key, 'char') and key.char else key.name
            STATE.record_key(name or 'unknown')
        except Exception:
            pass
    
    def on_move(x, y):
        STATE.record_mouse_move(x, y)
    
    def on_click(x, y, button, pressed):
        if pressed:
            STATE.record_mouse_click()
    
    def on_scroll(x, y, dx, dy):
        STATE.record_scroll()
    
    kb_listener = kb.Listener(on_press=on_press, suppress=False)
    ms_listener = ms.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    
    kb_listener.daemon = True
    ms_listener.daemon = True
    kb_listener.start()
    ms_listener.start()
    print("   ✓ Keyboard + mouse listeners active")

# ── AppleScript helpers ─────────────────────────────────
def run_applescript(script):
    try:
        r = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True, text=True, timeout=3
        )
        return r.stdout.strip()
    except Exception:
        return ""

def get_active_app():
    return run_applescript(
        'tell application "System Events" to get name of first application process whose frontmost is true'
    )

def get_browser_url(app):
    scripts = {
        "Google Chrome": 'tell application "Google Chrome" to get URL of active tab of front window',
        "Arc":           'tell application "Arc" to get URL of active tab of front window',
        "Safari":        'tell application "Safari" to get URL of current tab of front window',
    }
    script = scripts.get(app)
    return run_applescript(script) if script else ""

def get_browser_tab_count(app):
    scripts = {
        "Google Chrome": 'tell application "Google Chrome" to get count of tabs of front window',
        "Arc":           'tell application "Arc" to get count of tabs of front window',
        "Safari":        'tell application "Safari" to get count of tabs of front window',
    }
    script = scripts.get(app)
    if not script:
        return 0
    result = run_applescript(script)
    try:
        return int(result)
    except Exception:
        return 0

def get_all_tab_count():
    """Total tabs across all Chrome windows"""
    script = '''
    tell application "Google Chrome"
        set total to 0
        repeat with w in windows
            set total to total + (count of tabs of w)
        end repeat
        return total
    end tell
    '''
    result = run_applescript(script)
    try:
        return int(result)
    except Exception:
        return 0

def get_active_file(app):
    editors = {"Cursor", "Visual Studio Code", "Xcode", "BBEdit", "Nova", "Sublime Text", "PyCharm"}
    if app not in editors:
        return ""
    script = f'''
    tell application "System Events"
        tell process "{app}"
            try
                return value of attribute "AXTitle" of window 1
            end try
        end tell
    end tell
    '''
    return run_applescript(script)

def get_file_ext(title):
    if not title or "." not in title:
        return "none"
    ext = title.split(".")[-1].lower().strip()
    # Nettoyage si le titre contient des suffixes (ex: "main.py — phantom")
    ext = ext.split(" ")[0]
    mapping = {
        "py": "py", "cpp": "cpp", "hpp": "cpp", "c": "cpp", "h": "cpp",
        "md": "md", "markdown": "md",
        "json": "json", "yaml": "json", "yml": "json",
        "tsx": "tsx", "ts": "tsx", "js": "tsx", "jsx": "tsx", "html": "tsx", "css": "tsx"
    }
    return mapping.get(ext, "other")

def get_clipboard_type():
    """Détecte le type de contenu du presse-papiers sans loguer le contenu"""
    try:
        # pbpaste pour le texte, sips/osascript pour le reste
        content = run_applescript('the clipboard')
        if not content:
            # Vérifier si c'est une image (via AppleScript)
            img_check = run_applescript('try\nget (the clipboard as «class PNGf»)\nreturn "image"\non error\nreturn "empty"\nend try')
            return img_check if img_check == "image" else "empty"
        
        c = content.strip()
        if not c: return "empty"
        
        # URL detection
        if c.startswith(("http://", "https://", "www.")):
            return "url"
        
        # Code detection (minimal heuristics)
        code_keywords = {"def ", "function", "import ", "class ", "const ", "let ", "var ", "void ", "public:", "private:", "extern "}
        if any(kw in c for kw in code_keywords) or ("{" in c and "}" in c):
            return "code"
            
        return "text"
    except Exception:
        return "empty"

def is_dark_mode():
    """Détecte si macOS est en mode sombre"""
    try:
        res = run_applescript('tell application "System Events" to tell appearance preferences to get dark mode')
        return res.lower() == "true"
    except Exception:
        return False

def get_clipboard_hash():
    script = 'the clipboard'
    content = run_applescript(script)
    if content:
        return hashlib.md5(content.encode()).hexdigest()[:8]
    return ""

def get_window_count(app):
    """Nb de fenêtres de l'app active"""
    script = f'''
    tell application "System Events"
        tell process "{app}"
            return count of windows
        end tell
    end tell
    '''
    result = run_applescript(script)
    try:
        return int(result)
    except Exception:
        return 1

def get_window_layout():
    """Info layout via Quartz — fenêtres de toutes les apps"""
    if not QUARTZ_OK:
        return {"window_count_total": 0, "screens": 1}
    
    try:
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        visible = [w for w in windows if w.get('kCGWindowLayer', 999) == 0]
        
        # Nb d'écrans
        screens = len(Quartz.CGDisplayCopyAllDisplayModes(None, None) or [1])
        
        return {
            "window_count_total": len(visible),
            "screens": min(screens, 3),
        }
    except Exception:
        return {"window_count_total": 0, "screens": 1}

def get_ssid():
    """Récupère le nom du réseau WiFi actuel"""
    try:
        # Sur macOS: networksetup -getairportnetwork en0
        # Output type: Current Wi-Fi Network: NomDuReseau
        r = subprocess.run(
            ['networksetup', '-getairportnetwork', 'en0'],
            capture_output=True, text=True, timeout=2
        )
        if "Current Wi-Fi Network:" in r.stdout:
            return r.stdout.split("Current Wi-Fi Network:")[1].strip()
        return "OFFLINE"
    except Exception:
        return "UNKNOWN"

def get_system_metrics():
    """CPU, RAM, batterie, réseau"""
    try:
        cpu    = psutil.cpu_percent(interval=0.1)
        ram    = psutil.virtual_memory().percent
        
        battery = None
        bat = psutil.sensors_battery()
        if bat:
            battery = {
                "percent": round(bat.percent),
                "plugged": bat.power_plugged
            }
        
        # Bytes réseau delta (approx actif/idle)
        net = psutil.net_io_counters()
        net_bytes = net.bytes_sent + net.bytes_recv
        
        return {
            "cpu_percent":  round(cpu, 1),
            "ram_percent":  round(ram, 1),
            "battery":      battery,
            "net_bytes":    net_bytes,
        }
    except Exception:
        return {"cpu_percent": 0, "ram_percent": 0}

# ── Focus score ─────────────────────────────────────────
def compute_focus_score(behavior_snap, tab_count, window_count, app_switch_rate):
    """
    Score cognitif 0-100 basé sur :
    - WPM élevé + key_velocity faible → focus
    - Mouse idle élevé → focus
    - Peu de tabs + peu de fenêtres → focus
    - Peu de switches app → focus
    """
    score = 50  # baseline
    
    wpm = behavior_snap.get("wpm", 0)
    mouse_idle = behavior_snap.get("mouse_idle_s", 0)
    
    # Typing signal
    if wpm > 60:
        score += 20
    elif wpm > 30:
        score += 10
    elif wpm < 5:
        score -= 10
    
    # Mouse idle (si tu tapes sans bouger la souris → deep focus)
    if mouse_idle > 30 and wpm > 20:
        score += 15
    
    # Tab scatter
    if tab_count > 15:
        score -= 15
    elif tab_count > 8:
        score -= 5
    elif tab_count <= 3:
        score += 10
    
    # App switching
    if app_switch_rate > 3:   # > 3 switches/min → distraction
        score -= 20
    elif app_switch_rate < 0.5:
        score += 10
    
    return max(0, min(100, score))

# ── Logger ──────────────────────────────────────────────
def log_event(event):
    date_str  = datetime.now().strftime("%Y-%m-%d")
    log_file  = DATA_DIR / f"{date_str}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

# ── Main loop ───────────────────────────────────────────
def main():
    print("👻 Phantom Collector v2")
    print(f"   Polling every {POLL_INTERVAL}s")
    print(f"   Logging to: {DATA_DIR}\n")
    
    start_listeners()
    
    sensor        = ActionSensor()
    current_app   = None
    current_start = None
    current_url   = ""
    current_file  = ""
    current_ext   = "none"
    current_ssid  = "UNKNOWN"
    current_clip  = "empty"
    current_dark  = False
    current_bat_p = 100
    current_bat_c = True
    
    app_switch_times = deque(maxlen=20)  # pour calculer switch rate
    last_net_bytes   = 0
    
    while True:
        try:
            now = datetime.now()
            
            # ── App context ──
            app  = get_active_app()
            url  = get_browser_url(app) if app in BROWSERS else ""
            file = get_active_file(app)
            ext  = get_file_ext(file)
            ssid = get_ssid()
            clip = get_clipboard_type()
            dark = is_dark_mode()
            
            # ── Actions ──
            actions = sensor.detect(app, file)
            for a in actions:
                STATE.record_action(a)
            
            # ── Tab count ──
            tab_count    = get_browser_tab_count(app) if app in BROWSERS else 0
            all_tabs     = get_all_tab_count() if app in BROWSERS else 0
            window_count = get_window_count(app)
            
            # ── Window layout ──
            layout = get_window_layout()
            
            # ── System ──
            sys_metrics = get_system_metrics()
            net_delta   = sys_metrics.get("net_bytes", 0) - last_net_bytes
            last_net_bytes = sys_metrics.get("net_bytes", 0)
            
            # ── Behavior snapshot ──
            behavior = STATE.snapshot()
            
            # ── App switch rate (switches/min) ──
            t_now = time.time()
            if app != current_app:
                app_switch_times.append(t_now)
            
            recent_switches = sum(1 for t in app_switch_times if t > t_now - 60)
            app_switch_rate = recent_switches  # per minute
            
            # ── Focus score ──
            focus = compute_focus_score(behavior, all_tabs or tab_count, window_count, app_switch_rate)
            
            # ── Battery state ──
            bat = sys_metrics.get("battery")
            bat_p = bat["percent"] if bat else 100
            bat_c = bat["plugged"] if bat else True

            # ── Détection changement de contexte ──
            state_key = f"{app}|{url}|{file}|{ssid}|{clip}|{dark}|{bat_c}"
            
            if current_app is None:
                current_app   = app
                current_start = now
                current_url   = url
                current_file  = file
                current_ext   = ext
                current_ssid  = ssid
                current_clip  = clip
                current_dark  = dark
                current_bat_p = bat_p
                current_bat_c = bat_c
            
            elif state_key != f"{current_app}|{current_url}|{current_file}|{current_ssid}|{current_clip}|{current_dark}|{current_bat_c}":
                duration = (now - current_start).total_seconds()
                
                if duration >= MIN_DURATION:
                    event = {
                        # ── Contexte de base ──
                        "ts_start":  current_start.isoformat(),
                        "ts_end":    now.isoformat(),
                        "duration":  round(duration, 1),
                        "hour":      current_start.hour,
                        "weekday":   current_start.strftime("%A"),
                        "app":       current_app,
                        "url":       current_url,
                        "file":      current_file,
                        "active_file_ext": current_ext,
                        "ssid":      current_ssid,
                        "clipboard_type": current_clip,
                        "dark_mode": current_dark,
                        "bat_percent": current_bat_p,
                        "bat_charging": current_bat_c,
                        "contextual_token": get_contextual_token(current_app, current_url),
                        
                        # ── Keyboard ──
                        "wpm":              behavior["wpm"],
                        "key_velocity_ms":  behavior["key_velocity_ms"],
                        "backspace_rate":   behavior["backspace_rate"],
                        "modifier_rate":    behavior["modifier_rate"],
                        "keys_total":       behavior["total_keys_delta"],
                        
                        # ── Mouse ──
                        "mouse_distance_px": behavior["mouse_distance_px"],
                        "mouse_clicks":      behavior["mouse_clicks"],
                        "scroll_events":     behavior["scroll_events"],
                        "mouse_idle_s":      behavior["mouse_idle_s"],
                        
                        # ── Tabs & Windows ──
                        "tab_count":          tab_count,
                        "tab_count_total":    all_tabs,
                        "window_count":       window_count,
                        "window_count_total": layout.get("window_count_total", 0),
                        "screens":            layout.get("screens", 1),
                        
                        # ── System ──
                        "cpu_percent":   sys_metrics.get("cpu_percent", 0),
                        "ram_percent":   sys_metrics.get("ram_percent", 0),
                        "battery":       sys_metrics.get("battery"),
                        "net_active":    net_delta > 50000,  # > 50KB delta = actif
                        
                        # ── Composite ──
                        "focus_score":     focus,
                        "app_switch_rate": round(app_switch_rate, 1),
                        "action":          behavior["action"],
                    }
                    
                    log_event(event)
                    
                    # Log console compact
                    focus_icon = "🟢" if focus > 70 else ("🟡" if focus > 40 else "🔴")
                    print(f"[{now.strftime('%H:%M:%S')}] {current_app:<22} "
                          f"{round(duration)}s  "
                          f"WPM:{behavior['wpm']:<5} "
                          f"tabs:{all_tabs or tab_count:<3} "
                          f"{focus_icon}focus:{focus}")
                
                current_app   = app
                current_start = now
                current_url   = url
                current_file  = file
                current_ext   = ext
                current_ssid  = ssid
                current_clip  = clip
                current_dark  = dark
                current_bat_p = bat_p
                current_bat_c = bat_c
            
            time.sleep(POLL_INTERVAL)
        
        except KeyboardInterrupt:
            print("\n👻 Phantom v2 stopped.")
            break
        except Exception as e:
            with open(LOG_DIR / "errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} ERROR: {e}\n")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
