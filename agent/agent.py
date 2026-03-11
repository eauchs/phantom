#!/usr/bin/env python3
"""
Phantom Agent v2 — Proactive Action Layer
Reads live events, predicts next actions based on V2 tokens.
"""
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import mlx.core as mx
    import mlx.nn as nn
    BACKEND = "mlx"
except ImportError:
    BACKEND = "numpy"

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"
VOCAB_FILE = DATA_DIR / "vocab.json"
MODELS_DIR = ROOT / "models"

# ── Config ────────────────────────────────────────────
POLL_INTERVAL   = 5
CONFIDENCE_THRESHOLD = 0.55
CONTEXT_WINDOW  = 6
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
    seq_len = 16
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
                self.embed     = nn.Embedding(vocab_size, 64)
                self.pos_embed = nn.Embedding(16, 64)
                self.layers    = [nn.TransformerEncoderLayer(64, 4, 256) for _ in range(2)]
                self.norm      = nn.LayerNorm(64)
                self.head      = nn.Linear(64, vocab_size)

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
    import sys
    sys.path.insert(0, str(ROOT))
    from tokenizer.tokenizer_v2 import tokenize_event
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

# ── Main loop ─────────────────────────────────────────
def main():
    print("👻 Phantom Agent v2 started")
    print(f"   Backend: {BACKEND.upper()}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"   Watching: {EVENTS_DIR}\n")

    token2id = load_vocab()
    if not token2id:
        print("   ⚠️  No vocab found. Run trainer first.")
        return

    id2token = {v: k for k, v in token2id.items()}
    model    = load_model(token2id)
    if model is None:
        print("   ⚠️  No model found. Run trainer first.")
        return

    print("   ✓ Model loaded. Watching your behavior...\n")
    notify("Phantom v2", "Agent démarré — analyse du focus et clavier activée 👀")

    last_notif  = {}
    last_events = []

    while True:
        try:
            events = get_recent_events(CONTEXT_WINDOW)
            if events == last_events or not events:
                time.sleep(POLL_INTERVAL)
                continue

            last_events = events
            
            context_tokens = []
            for ev in events:
                context_tokens.extend(event_to_tokens(ev))
            context_tokens = context_tokens[-16:]

            preds = predict(model, token2id, id2token, context_tokens)

            for token, confidence in preds:
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                if token in {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}:
                    continue

                now = time.time()
                if token in last_notif and now - last_notif[token] < COOLDOWN:
                    continue

                title, message = token_to_readable(token)
                notify(title, message, subtitle=f"👻 {confidence:.0%} de confiance")
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
