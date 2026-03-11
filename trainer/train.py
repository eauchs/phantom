#!/usr/bin/env python3
"""
Phantom Trainer — Online Behavioral Transformer
Trains a small transformer on your tokenized sequences.
Predicts next token → learns your patterns.

Input:  data/sequences/*.txt
Output: models/phantom_latest.npz
"""

import json
import math
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    BACKEND = "mlx"
except ImportError:
    import warnings
    warnings.warn("MLX not found, falling back to numpy (slow). Install: pip install mlx")
    BACKEND = "numpy"

DATA_DIR   = Path(__file__).parent.parent / "data" / "sequences"
MODELS_DIR = Path(__file__).parent.parent / "models"
VOCAB_FILE = Path(__file__).parent.parent / "data" / "vocab.json"

# ── Hyperparams ────────────────────────────────────────
CONFIG = {
    "d_model":    64,    # dimension du modèle (petit = rapide)
    "n_heads":    4,     # têtes d'attention
    "n_layers":   2,     # couches transformer
    "seq_len":    16,    # longueur de séquence de contexte
    "batch_size": 32,
    "lr":         3e-4,
    "epochs":     20,
    "dropout":    0.1,
}

# ── Vocab ──────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self.token2id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id2token = {v: k for k, v in self.token2id.items()}

    def build(self, sequences: list[list[str]], min_freq: int = 1):
        counter = Counter(t for seq in sequences for t in seq)
        for token, freq in sorted(counter.items()):
            if freq >= min_freq and token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx]   = token
        print(f"   Vocabulary: {len(self.token2id)} tokens")

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, 1) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(i, "<UNK>") for i in ids]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id}, f, indent=2)
        print(f"   Vocab saved → {path}")

    def load(self, path: Path):
        with open(path) as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(v): k for k, v in self.token2id.items()}

    def __len__(self):
        return len(self.token2id)

# ── Dataset ────────────────────────────────────────────
def load_sequences() -> list[list[str]]:
    sequences = []
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        print("   No sequence files found. Run tokenizer first.")
        return []
    for f in files:
        with open(f) as fp:
            for line in fp:
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    sequences.append(tokens)
    print(f"   Loaded {len(sequences)} sequences from {len(files)} files")
    return sequences

def make_windows(sequences, vocab, seq_len):
    """Sliding window: predict next token from context."""
    X, Y = [], []
    for seq in sequences:
        ids = [vocab.token2id.get("<BOS>", 2)] + vocab.encode(seq)
        for i in range(len(ids) - 1):
            ctx_start = max(0, i - seq_len + 1)
            ctx = ids[ctx_start:i+1]
            # Pad si nécessaire
            pad = seq_len - len(ctx)
            ctx = [0] * pad + ctx
            X.append(ctx)
            Y.append(ids[i+1])
    return np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)

# ── Transformer (MLX) ──────────────────────────────────
if BACKEND == "mlx":
    class PhantomTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len):
            super().__init__()
            self.embed    = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(seq_len, d_model)
            self.layers   = [
                nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
                for _ in range(n_layers)
            ]
            self.norm     = nn.LayerNorm(d_model)
            self.head     = nn.Linear(d_model, vocab_size)
            self.seq_len  = seq_len

        def __call__(self, x):
            T = x.shape[1]
            positions = mx.arange(T)
            h = self.embed(x) + self.pos_embed(positions)
            for layer in self.layers:
                h = layer(h)
            h = self.norm(h)
            return self.head(h[:, -1, :])  # predict next from last position

    def train_mlx(X, Y, vocab_size, config):
        model  = PhantomTransformer(
            vocab_size, config["d_model"], config["n_heads"],
            config["n_layers"], config["seq_len"]
        )
        optimizer = optim.AdamW(learning_rate=config["lr"])

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y).mean()

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        X_mx = mx.array(X)
        Y_mx = mx.array(Y)
        n    = len(X)
        bs   = config["batch_size"]

        print(f"\n   Training on {n} windows | {config['epochs']} epochs | MLX backend")
        for epoch in range(config["epochs"]):
            idx    = np.random.permutation(n)
            losses = []
            for start in range(0, n, bs):
                batch_idx = idx[start:start+bs]
                xb = X_mx[batch_idx]
                yb = Y_mx[batch_idx]
                loss, grads = loss_and_grad(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                losses.append(float(loss))
            avg = np.mean(losses)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{config['epochs']} | loss {avg:.4f}")

        return model

# ── Numpy fallback ─────────────────────────────────────
def train_numpy(X, Y, vocab_size, config):
    """Bigram frequency model — fallback si pas MLX."""
    print("   [numpy mode] Training bigram frequency model")
    transitions = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for i in range(len(X)):
        ctx_last = X[i, -1]
        transitions[ctx_last, Y[i]] += 1
    # Normalise
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    transitions = transitions / row_sums
    return transitions

# ── Predict ────────────────────────────────────────────
def predict_next(model, vocab, context_tokens: list[str], top_k=3):
    seq_len = CONFIG["seq_len"]
    ids = vocab.encode(context_tokens)
    pad = seq_len - len(ids)
    ids = [0] * max(0, pad) + ids[-seq_len:]
    
    if BACKEND == "mlx":
        x = mx.array([ids])
        logits = model(x)
        probs  = mx.softmax(logits[0]).tolist()
        top    = sorted(enumerate(probs), key=lambda x: -x[1])[:top_k]
        return [(vocab.id2token.get(i, "<UNK>"), p) for i, p in top]
    else:
        last_id = ids[-1]
        row = model[last_id]
        top = sorted(enumerate(row), key=lambda x: -x[1])[:top_k]
        return [(vocab.id2token.get(i, "<UNK>"), float(p)) for i, p in top]

# ── Save / Load ────────────────────────────────────────
def save_model(model, path: Path):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if BACKEND == "mlx":
        model.save_weights(str(path))
    else:
        np.save(str(path), model)
    print(f"   Model saved → {path}")

# ── Main ───────────────────────────────────────────────
def main():
    print("🧠 Phantom Trainer")
    print(f"   Backend: {BACKEND.upper()}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # 1. Load sequences
    sequences = load_sequences()
    if not sequences:
        return

    # 2. Build vocab
    vocab = Vocabulary()
    vocab.build(sequences)
    vocab.save(VOCAB_FILE)

    # 3. Make training windows
    X, Y = make_windows(sequences, vocab, CONFIG["seq_len"])
    print(f"   Training windows: {len(X)}")

    if len(X) < 10:
        print("   ⚠️  Too few windows. Let the daemon run longer before training.")
        return

    # 4. Train
    t0 = time.time()
    if BACKEND == "mlx":
        model = train_mlx(X, Y, len(vocab), CONFIG)
    else:
        model = train_numpy(X, Y, len(vocab), CONFIG)
    print(f"   Training time: {time.time()-t0:.1f}s")

    # 5. Save
    suffix = ".npz" if BACKEND == "mlx" else ".npy"
    model_path = MODELS_DIR / f"phantom_latest{suffix}"
    save_model(model, model_path)

    # 6. Demo predictions
    print("\n   📊 Sample predictions from your patterns:")
    samples = [
        ["SESSION:NIGHT", "APP:BROWSER"],
        ["SESSION:NIGHT", "APP:TERMINAL"],
        ["SESSION:NIGHT", "DAY:WED", "APP:BROWSER", "AI:CLAUDE"],
    ]
    for ctx in samples:
        preds = predict_next(model, vocab, ctx)
        ctx_str = " ".join(ctx)
        pred_str = " | ".join(f"{t} ({p:.0%})" for t, p in preds)
        print(f"   [{ctx_str}] → {pred_str}")

    print(f"\n✓ Done. Run again tomorrow for better predictions.")

if __name__ == "__main__":
    main()
