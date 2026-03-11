#!/usr/bin/env python3
"""
Phantom Agent — Proactive Action Layer
Reads live events from the daemon, runs predictions,
and fires macOS notifications when confident.

Run alongside the daemon:
  Terminal 1: python3 daemon/collector.py
  Terminal 2: python3 agent/agent.py
"""

import json
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

try:
    import mlx.core as mx
    BACKEND = "mlx"
except ImportError:
    BACKEND = "numpy"

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"
VOCAB_FILE = DATA_DIR / "vocab.json"
MODELS_DIR = ROOT / "models"

# ── Config ────────────────────────────────────────────
POLL_INTERVAL   = 5      # secondes entre chaque check
CONFIDENCE_THRESHOLD = 0.55  # seuil pour notifier
CONTEXT_WINDOW  = 6      # derniers events à garder en contexte
COOLDOWN        = 120    # secondes entre deux notifs identiques

# ── Notif macOS ───────────────────────────────────────
def notify(title: str, message: str, subtitle: str = "👻 Phantom"):
    script = f'''
    display notification "{message}" with title "{title}" subtitle "{subtitle}"
    '''
    subprocess.run(["osascript", "-e", script], capture_output=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔔 {title} — {message}")

# ── Vocab ─────────────────────────────────────────────
def load_vocab() -> dict:
    if not VOCAB_FILE.exists():
        return {}
    with open(VOCAB_FILE) as f:
        data = json.load(f)
    return data["token2id"]

# ── Prediction ────────────────────────────────────────
def predict(model, token2id: dict, id2token: dict, context: list[str], top_k=3):
    seq_len = 16
    ids = [token2id.get(t, 1) for t in context]
    pad = seq_len - len(ids)
    ids = [0] * max(0, pad) + ids[-seq_len:]

    if BACKEND == "mlx":
        import mlx.nn as nn
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

# ── Load model ────────────────────────────────────────
def load_model(token2id):
    if BACKEND == "mlx":
        npz = MODELS_DIR / "phantom_latest.npz"
        if not npz.exists():
            return None
        # Reconstruit le modèle et charge les poids
        import mlx.nn as nn
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

# ── Lire derniers events ──────────────────────────────
def get_recent_events(n=10) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    f = EVENTS_DIR / f"{today}.jsonl"
    if not f.exists():
        return []
    lines = f.read_text().strip().split("\n")
    lines = [l for l in lines if l.strip()]
    events = []
    for l in lines[-n:]:
        try:
            events.append(json.loads(l))
        except Exception:
            pass
    return events

# ── Event → tokens ────────────────────────────────────
def event_to_tokens(event: dict) -> list[str]:
    from tokenizer.tokenizer import tokenize_event
    return tokenize_event(event).split()

# ── Prédiction → message lisible ─────────────────────
READABLE = {
    "AI:CLAUDE":          ("Claude", "Tu vas probablement ouvrir Claude"),
    "CODE:GITHUB":        ("GitHub", "Tu vas probablement aller sur GitHub"),
    "APP:TERMINAL":       ("Terminal", "Tu vas probablement ouvrir le Terminal"),
    "APP:EDITOR":         ("Éditeur", "Tu vas probablement coder"),
    "APP:BROWSER":        ("Browser", "Tu vas probablement ouvrir le browser"),
    "DUR:DEEP":           ("Deep work", "Session longue détectée — coupe les notifs ?"),
    "DUR:GLANCE":         ("Coup d'œil", "Passage rapide prévu"),
    "SESSION:NIGHT":      ("Nuit", "Tu travailles tard ce soir"),
    "SOCIAL:REDDIT":      ("Reddit", "Tu vas probablement aller sur Reddit"),
    "RESEARCH:ARXIV":     ("Arxiv", "Tu vas probablement lire un paper"),
    "APP:MESSAGING":      ("Message", "Tu vas probablement checker tes messages"),
}

def token_to_readable(token: str) -> tuple[str, str]:
    return READABLE.get(token, (token, f"Prédit : {token}"))

# ── Main loop ─────────────────────────────────────────
def main():
    print("👻 Phantom Agent started")
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
    notify("Phantom", "Agent démarré — je t'observe 👀")

    last_notif  = {}   # token → timestamp dernière notif
    last_events = []   # events vus au dernier poll

    import sys
    sys.path.insert(0, str(ROOT))

    while True:
        try:
            events = get_recent_events(CONTEXT_WINDOW)

            if events == last_events or not events:
                time.sleep(POLL_INTERVAL)
                continue

            last_events = events

            # Construit le contexte courant
            context_tokens = []
            for ev in events:
                context_tokens.extend(event_to_tokens(ev))
            context_tokens = context_tokens[-16:]  # garde les 16 derniers

            # Prédit
            preds = predict(model, token2id, id2token, context_tokens)

            for token, confidence in preds:
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                if token in {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}:
                    continue

                # Cooldown — pas deux fois la même notif en 2 min
                now = time.time()
                if token in last_notif and now - last_notif[token] < COOLDOWN:
                    continue

                title, message = token_to_readable(token)
                notify(title, message, subtitle=f"👻 {confidence:.0%} de confiance")
                last_notif[token] = now
                break  # une seule notif par cycle

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n👻 Phantom Agent stopped.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
