#!/usr/bin/env python3
"""
Phantom Agent v2 — Proactive Action Layer
Reads live events, predicts next actions based on V2 tokens.
"""
import json
import time
import subprocess
import sys
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Import interviewer ──
try:
    from agent.interviewer import interviewer_loop
except ImportError as e:
    print(f"   ⚠️  Could not import interviewer: {e}")
    interviewer_loop = None

try:
    import mlx.core as mx
    import mlx.nn as nn
    BACKEND = "mlx"
except ImportError:
    BACKEND = "numpy"

ROOT       = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer_v2 import tokenize_event
from agent.feedback_logger import log_feedback

DATA_DIR   = ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"
FEATURES_DIR = DATA_DIR / "features"
VOCAB_FILE = DATA_DIR / "vocab.json"
MODELS_DIR = ROOT / "models"
PROFILE_FILE = DATA_DIR / "profile" / "profile.json"
TOKEN_INJECTION_FILE = DATA_DIR / "profile" / "injected_tokens.json"

# ── Profile Loading ───────────────────────────────────
def load_profile():
    if not PROFILE_FILE.exists():
        return {"preferences": {}, "avoid": {}, "context": {}, "action_feedback": {}}
    try:
        return json.loads(PROFILE_FILE.read_text())
    except:
        return {"preferences": {}, "avoid": {}, "context": {}, "action_feedback": {}}

def get_injected_tokens():
    if not TOKEN_INJECTION_FILE.exists():
        return []
    try:
        data = json.loads(TOKEN_INJECTION_FILE.read_text())
        # Only inject if tokens are fresh (last 10 mins)
        if time.time() - data.get("ts", 0) < 600:
            return data.get("tokens", [])
    except:
        pass
    return []

def get_profile_summary(profile):
    summary = []
    if profile.get("preferences"):
        summary.append(f"Likes: {list(profile['preferences'].keys())}")
    if profile.get("avoid"):
        summary.append(f"Avoids: {list(profile['avoid'].keys())}")
    if profile.get("context", {}).get("last_interaction"):
        last = profile["context"]["last_interaction"]
        summary.append(f"Recent: {last.get('a')}")
    return " ".join(summary)

# ── Two-Tower Config ─────────────────────────────────
ACTION_VOCAB = [
  "ACT:OPEN_BROWSER", "ACT:OPEN_TERMINAL", "ACT:OPEN_EDITOR",
  "ACT:OPEN_COMM", "ACT:GIT_PUSH", "ACT:GIT_COMMIT", "ACT:GIT_PULL",
  "ACT:FILE_CREATE", "ACT:DND_ON", "ACT:MUSIC_PLAY",
  "FOCUS:DEEP", "FOCUS:SHALLOW", "SWITCH:FAST",
  "WEB:GITHUB", "WEB:CLAUDE", "WEB:YOUTUBE"
]
N_ACTIONS = len(ACTION_VOCAB)
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_VOCAB)}

class UserTower(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        h_seq, _ = self.lstm(h)
        h_last = h_seq[:, -1, :]
        out = self.fc2(h_last)
        return out / mx.linalg.norm(out, axis=-1, keepdims=True)

class CandidateTower(nn.Module):
    def __init__(self, n_actions, output_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, output_dim)

    def __call__(self, action_indices):
        out = self.embedding(action_indices)
        return out / mx.linalg.norm(out, axis=-1, keepdims=True)

def load_two_tower():
    npz_path = MODELS_DIR / "two_tower.npz"
    mapping_path = FEATURES_DIR / "mappings.json"
    if not npz_path.exists() or not mapping_path.exists():
        return None, None
    
    with open(mapping_path) as f:
        mappings = json.load(f)
    
    u_tower = UserTower()
    c_tower = CandidateTower(N_ACTIONS)
    
    weights = mx.load(str(npz_path))
    u_weights = {k.replace("user_tower.", ""): v for k, v in weights.items() if k.startswith("user_tower.")}
    c_weights = {k.replace("candidate_tower.", ""): v for k, v in weights.items() if k.startswith("candidate_tower.")}
    
    u_tower.load_weights(list(u_weights.items()))
    c_tower.load_weights(list(c_weights.items()))
    
    return (u_tower, c_tower), mappings

def events_to_feature_matrix(events, mappings, seq_len=512):
    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    CLIPBOARD_TYPES = mappings.get("clip", ["empty", "text", "url", "image", "code"])
    
    matrix = []
    for e in events:
        vec = np.zeros(24, dtype=np.float32)
        vec[0] = e.get("hour", 0) / 24.0
        try:
            wd = e.get("weekday", "Monday")
            vec[1] = WEEKDAYS.index(wd) / 7.0
        except: vec[1] = 0.0
        
        vec[2] = min(e.get("wpm", 0) / 200.0, 1.0)
        vec[3] = min(e.get("backspace_rate", 0) / 30.0, 1.0)
        vec[4] = min(e.get("modifier_rate", 0) / 30.0, 1.0)
        vec[5] = min(e.get("mouse_distance_px", 0) / 5000.0, 1.0)
        vec[6] = min(e.get("mouse_clicks", 0) / 20.0, 1.0)
        vec[7] = min(e.get("scroll_events", 0) / 100.0, 1.0)
        vec[8] = min(e.get("mouse_idle_s", 0) / 300.0, 1.0)
        vec[9] = min(e.get("tab_count", 0) / 20.0, 1.0)
        vec[10] = min(e.get("window_count", 0) / 10.0, 1.0)
        vec[11] = min(e.get("cpu_percent", 0) / 100.0, 1.0)
        vec[12] = min(e.get("ram_percent", 0) / 100.0, 1.0)
        
        bat = e.get("battery") or {}
        vec[13] = (e.get("bat_percent") or bat.get("percent", 100)) / 100.0
        vec[14] = 1.0 if (e.get("bat_charging") or bat.get("plugged", True)) else 0.0
        
        vec[15] = e.get("focus_score", 50) / 100.0
        vec[16] = min(e.get("app_switch_rate", 0) / 10.0, 1.0)
        vec[17] = 1.0 if e.get("net_active") else 0.0
        vec[18] = 1.0 if e.get("dark_mode") else 0.0
        vec[19] = min(e.get("duration", 0) / 3600.0, 1.0)
        
        app = e.get("app", "unknown")
        vec[20] = mappings["apps"].get(app, 0)
        
        url = e.get("url", "")
        domain = url.split("//")[-1].split("/")[0] if "//" in url else url
        vec[21] = mappings["urls"].get(domain, 0)
        
        clip = e.get("clipboard_type", "empty")
        try: vec[22] = CLIPBOARD_TYPES.index(clip)
        except: vec[22] = 0
        
        vec[23] = min(e.get("screens", 1) / 3.0, 1.0)
        matrix.append(vec)
    
    if len(matrix) < seq_len:
        pad = [np.zeros(24, dtype=np.float32)] * (seq_len - len(matrix))
        matrix = pad + matrix
    return np.array(matrix[-seq_len:], dtype=np.float32)

# ── Config ────────────────────────────────────────────
POLL_INTERVAL   = 5
CONFIDENCE_THRESHOLD = 0.55
CONTEXT_WINDOW  = 512 # Extended for Two-Tower
COOLDOWN        = 120

# ── Notif macOS ───────────────────────────────────────
def notify(title: str, message: str, subtitle: str = "👻 Phantom"):
    script = f'''
    display notification "{message}" with title "{title}" subtitle "{subtitle}"
    '''
    subprocess.run(["osascript", "-e", script], capture_output=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔔 {title} — {message}")

def load_vocab() -> dict:
    if not VOCAB_FILE.exists():
        return {}
    with open(VOCAB_FILE) as f:
        data = json.load(f)
    return data["token2id"]

def predict(model, token2id: dict, id2token: dict, context: list[str], top_k=3):
    seq_len = 32
    ids = [token2id.get(t, 1) for t in context]
    pad = seq_len - len(ids)
    ids = [0] * max(0, pad) + ids[-seq_len:]

    if BACKEND == "mlx":
        x = mx.array([ids])
        logits = model(x)
        probs = mx.softmax(logits[0]).tolist()
        top = sorted(enumerate(probs), key=lambda x: -x[1])[:top_k]
        return [(id2token.get(i, "<UNK>"), p) for i, p in top]
    else:
        last_id = ids[-1]
        row = model[last_id]
        top = sorted(enumerate(row), key=lambda x: -x[1])[:top_k]
        return [(id2token.get(i, "<UNK>"), float(p)) for i, p in top]

def load_model(token2id):
    if BACKEND == "mlx":
        npz = MODELS_DIR / "phantom_latest.npz"
        if not npz.exists():
            return None
        vocab_size = len(token2id)
        
        class PhantomTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed     = nn.Embedding(vocab_size, 128)
                self.pos_embed = nn.Embedding(32, 128)
                self.layers    = [nn.TransformerEncoderLayer(128, 4, 512) for _ in range(4)]
                self.norm      = nn.LayerNorm(128)
                self.head      = nn.Linear(128, vocab_size)

            def __call__(self, x):
                T = x.shape[1]
                positions = mx.arange(T)
                h = self.embed(x) + self.pos_embed(positions)
                for layer in self.layers:
                    h = layer(h, mask=None)
                h = self.norm(h)
                return self.head(h[:, -1, :])

        model = PhantomTransformer()
        model.load_weights(str(npz))
        return model
    else:
        npy = MODELS_DIR / "phantom_latest.npy"
        if not npy.exists():
            return None
        return np.load(str(npy))

def get_recent_events(n=10) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    f = EVENTS_DIR / f"{today}.jsonl"
    if not f.exists(): return []
    
    lines = f.read_text().strip().split("\n")
    events = []
    for l in lines[-n:]:
        if l.strip():
            try: events.append(json.loads(l))
            except: pass
    return events

def event_to_tokens(event: dict) -> list[str]:
    # ATTENTION: Utilise le tokenizer v2
    return tokenize_event(event).split()

# ── Prédiction → message lisible (V2) ─────────────────
READABLE = {
    "AI:CLAUDE":          ("Claude", "Besoin d'aide sur ton code ?"),
    "CODE:GITHUB":        ("GitHub", "Tu t'apprêtes à check tes PRs ou push"),
    "APP:TERMINAL":       ("Terminal", "Retour au shell"),
    "APP:EDITOR":         ("Éditeur", "Retour dans la codebase"),
    "APP:BROWSER":        ("Browser", "Tu vas faire une recherche"),
    "DUR:DEEP":           ("Deep work", "Longue session en approche"),
    "FOCUS:DEEP":         ("Zone", "Focus absolu détecté. J'arrête les notifs."),
    "FOCUS:SHALLOW":      ("Distraction", "Ton focus baisse. Ferme des onglets ?"),
    "TYPING:FAST":        ("Flow State", "WPM très élevé, belle dynamique !"),
    "SWITCH:FAST":        ("Zapping", "Tu changes beaucoup d'app, besoin d'une pause ?"),
    "WEB:YOUTUBE":        ("YouTube", "Petite pause vidéo prévue"),
    "ERROR:HIGH":         ("Bugs ?", "Beaucoup de retours arrière. Bloqué sur un bug ?")
}

def token_to_readable(token: str) -> tuple[str, str]:
    return READABLE.get(token, (token, f"Prédit : {token}"))

# ── Actions execution ─────────────────────────────────
ACTION_MAP = {
    "ACT:OPEN_BROWSER":  ("Browser", "Ouvrir Claude.ai ?", "open https://claude.ai"),
    "ACT:OPEN_TERMINAL": ("Terminal", "Lancer le Terminal ?", "open -a Terminal"),
    "ACT:GIT_PUSH":      ("Git", "Pousser les changements ?", "osascript -e 'tell app \"Terminal\" to do script \"git push\"'"),
    "ACT:OPEN_COMM":     ("Chat", "Ouvrir WhatsApp ?", "open -a WhatsApp"),
}

def ask_confirmation(title, message):
    # Utilise osascript pour un dialogue bloquant (UI macOS)
    script = f'display dialog "{message}" with title "{title}" buttons {{"Annuler", "OK"}} default button "OK" giving up after 15'
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return "OK" in r.stdout
    except:
        return False

def execute_action(token, context_tokens):
    if token not in ACTION_MAP:
        return False
    
    title, prompt, command = ACTION_MAP[token]
    
    # Demande confirmation via dialogue macOS
    confirmed = ask_confirmation(f"👻 Phantom: {title}", prompt)
    
    # Log feedback (important pour le futur RL / reward)
    log_feedback(token, confirmed, context_tokens)
    
    if confirmed:
        try:
            cwd = str(ROOT) if "git" in command and "osascript" not in command else None
            subprocess.run(command, shell=True, cwd=cwd)
            notify("Action exécutée", f"{token} a été lancé avec succès.")
            return True
        except Exception as e:
            notify("Erreur", f"Échec de l'action {token}: {e}")
    
    return False

# ── Main loop ─────────────────────────────────────────
def main():
    print("👻 Phantom Agent v2 started")
    print(f"   Backend: {BACKEND.upper()}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"   Watching: {EVENTS_DIR}\n")

    # ── Load Profile ──
    profile = load_profile()
    print(f"   ✓ Profile loaded: {len(profile.get('action_feedback', {}))} feedback points.")

    # ── Start Interviewer Thread ──
    if interviewer_loop:
        thread = threading.Thread(target=interviewer_loop, daemon=True)
        thread.start()
        print("   ✓ Interviewer thread active.")

    token2id = load_vocab()
    if not token2id:
        print("   ⚠️  No vocab found. Run trainer first.")
        return

    id2token = {v: k for k, v in token2id.items()}
    model    = load_model(token2id)
    
    # ── Two-Tower Load ──
    tt_model, tt_mappings = load_two_tower()
    if tt_model:
        print("   ✓ Two-Tower model loaded.")
    
    if model is None and tt_model is None:
        print("   ⚠️  No model found. Run trainer first.")
        return

    print("   ✓ Monitoring behavior...\n")
    notify("Phantom v2.1", "Agent démarré — Two-Tower architecture active 🚀")

    last_notif  = {}
    last_events = []

    while True:
        try:
            # Refresh profile periodically
            if time.time() % 300 < 10:
                profile = load_profile()

            events = get_recent_events(CONTEXT_WINDOW)
            if events == last_events or not events:
                time.sleep(POLL_INTERVAL)
                continue

            last_events = events
            
            # 1. Try Two-Tower Prediction
            tt_preds = []
            if tt_model:
                # Note: Injecting profile summary context logically before calling
                profile_summary = get_profile_summary(profile)
                
                u_tower, c_tower = tt_model
                x = events_to_feature_matrix(events, tt_mappings)
                # Ensure x is a numpy array before conversion to MLX
                x = np.array(x, dtype=np.float32)
                x = mx.array(np.expand_dims(x, axis=0))
                
                user_emb = u_tower(x)
                action_embs = c_tower(mx.arange(N_ACTIONS))
                scores = (user_emb @ action_embs.T)[0]
                
                # Apply profile-based bias if any (e.g. avoid actions user rejected)
                for action, fb in profile.get("action_feedback", {}).items():
                    if action in ACTION_TO_ID and fb.get("reason"):
                        idx = ACTION_TO_ID[action]
                        scores[idx] -= 2.0 # Penalize rejected actions
                
                probs = mx.sigmoid(scores * 10.0).tolist()
                
                for i, p in enumerate(probs):
                    if p > 0.75:
                        tt_preds.append((ACTION_VOCAB[i], p))
                
                # Sort by confidence
                tt_preds.sort(key=lambda x: -x[1])

            # 2. Transformer Fallback
            trans_preds = []
            context_tokens = []
            if model:
                # Inject profile tokens as prefix
                profile_summary = get_profile_summary(profile)
                if profile_summary:
                    context_tokens.append(f"PRF:{profile_summary}")
                
                # NLP: Inject tokens from last interviewer answer
                injected = get_injected_tokens()
                if injected:
                    context_tokens.extend([f"NLP:{t}" for t in injected])

                for ev in events[-16:]: # Keep window small for transformer
                    context_tokens.extend(event_to_tokens(ev))
                context_tokens = context_tokens[-16:]
                trans_preds = predict(model, token2id, id2token, context_tokens)

            # 3. Decision Logic
            final_preds = tt_preds if tt_preds else trans_preds

            for token, confidence in final_preds:
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                if token in {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}:
                    continue

                now = time.time()
                if token in last_notif and now - last_notif[token] < COOLDOWN:
                    continue

                # ── Exécution Proactive (Actions) ──
                if token.startswith("ACT:") and confidence > 0.70:
                    executed = execute_action(token, context_tokens)
                    if executed:
                        last_notif[token] = now
                        break

                # ── Notification Simple (Observations) ──
                title, message = token_to_readable(token)
                notify(title, message, subtitle=f"👻 {confidence:.0%} de confiance ({'TT' if tt_preds else 'Trans'})")
                last_notif[token] = now
                break

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n👻 Phantom Agent stopped.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
