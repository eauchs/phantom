#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = Path(__file__).parent.parent
FEATURES_DIR = ROOT / "data" / "features"
EVENTS_DIR = ROOT / "data" / "events"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
        # x: (batch, seq_len, input_dim)
        h = nn.relu(self.fc1(x))
        # LSTM in MLX returns (h_seq, (h_last, c_last))
        _, (h_last, _) = self.lstm(h)
        # h_last is (num_layers, batch, output_dim) -> take last layer
        h_last = h_last[-1]
        out = self.fc2(h_last)
        # L2 Normalize
        return out / mx.linalg.norm(out, axis=-1, keepdims=True)

class CandidateTower(nn.Module):
    def __init__(self, n_actions, output_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, output_dim)

    def __call__(self, action_indices):
        # action_indices: (batch,) or (n_actions,)
        out = self.embedding(action_indices)
        # L2 Normalize
        return out / mx.linalg.norm(out, axis=-1, keepdims=True)

def load_data(seq_len=512):
    X = []
    Y = []
    
    feature_files = sorted(FEATURES_DIR.glob("*.npy"))
    event_files = sorted(EVENTS_DIR.glob("*.jsonl"))
    
    for f_feat, f_event in zip(feature_files, event_files):
        feats = np.load(f_feat)
        with open(f_event, "r") as jf:
            events = [json.loads(line) for line in jf]
        
        # Ensure they match
        n = min(len(feats), len(events))
        feats = feats[:n]
        events = events[:n]
        
        for i in range(n):
            # Sequence ending at i
            start = max(0, i - seq_len + 1)
            window = feats[start:i+1]
            if len(window) < seq_len:
                pad = np.zeros((seq_len - len(window), 24), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)
            
            X.append(window)
            
            # Label from current event
            e = events[i]
            label = np.zeros(N_ACTIONS, dtype=np.float32)
            
            # 1. Direct actions from log
            actions_str = e.get("action")
            if actions_str:
                for a in actions_str.split(","):
                    if a in ACTION_TO_ID:
                        label[ACTION_TO_ID[a]] = 1.0
            
            # 2. Pseudo-labels
            focus = e.get("focus_score", 50)
            if focus > 70: label[ACTION_TO_ID["FOCUS:DEEP"]] = 1.0
            if focus < 30: label[ACTION_TO_ID["FOCUS:SHALLOW"]] = 1.0
            
            sw_rate = e.get("app_switch_rate", 0)
            if sw_rate > 5: label[ACTION_TO_ID["SWITCH:FAST"]] = 1.0
            
            url = e.get("url", "").lower()
            if "github.com" in url: label[ACTION_TO_ID["WEB:GITHUB"]] = 1.0
            if "claude.ai" in url: label[ACTION_TO_ID["WEB:CLAUDE"]] = 1.0
            if "youtube.com" in url: label[ACTION_TO_ID["WEB:YOUTUBE"]] = 1.0
            
            # Mapping common apps to actions if not explicitly in ACT:
            app = e.get("app", "").lower()
            if "chrome" in app or "safari" in app or "arc" in app:
                label[ACTION_TO_ID["ACT:OPEN_BROWSER"]] = 1.0
            if "terminal" in app or "iterm" in app:
                label[ACTION_TO_ID["ACT:OPEN_TERMINAL"]] = 1.0
            if "cursor" in app or "code" in app or "xcode" in app:
                label[ACTION_TO_ID["ACT:OPEN_EDITOR"]] = 1.0
            if "slack" in app or "discord" in app or "telegram" in app or "whatsapp" in app:
                label[ACTION_TO_ID["ACT:OPEN_COMM"]] = 1.0

            Y.append(label)

    return np.array(X), np.array(Y)

def train():
    print("🧠 Training Two-Tower Recommender...")
    X, Y = load_data()
    if len(X) == 0:
        print("❌ No data to train on.")
        return

    user_tower = UserTower()
    candidate_tower = CandidateTower(N_ACTIONS)
    
    # Optimizer
    params = list(user_tower.parameters()) + list(candidate_tower.parameters())
    optimizer = optim.AdamW(learning_rate=1e-3)

    def loss_fn(model_u, model_c, x, y):
        user_emb = model_u(x) # (batch, 256)
        all_actions = mx.arange(N_ACTIONS)
        action_embs = model_c(all_actions) # (N_ACTIONS, 256)
        
        # Scores: (batch, N_ACTIONS)
        scores = user_emb @ action_embs.T
        
        # Multi-label BCE
        # nn.losses.binary_cross_entropy(logits, targets)
        # Note: scores are already normalized, but BCE usually expects logits or probs.
        # Since we use dot product of normalized vectors, scores are in [-1, 1].
        # We can scale them (temperature) or just use them as is.
        # Let's use a small temperature or scale up.
        logits = scores * 10.0
        return nn.losses.binary_cross_entropy(logits, y).mean()

    loss_and_grad = nn.value_and_grad(user_tower, loss_fn)
    
    batch_size = 32
    epochs = 50
    n = len(X)
    
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            indices = perm[i:i+batch_size]
            xb = mx.array(X[indices])
            yb = mx.array(Y[indices])
            
            # We need to pass both models to grad
            # But value_and_grad only takes one model by default if we use it like this.
            # We can use a container model.
            
            def combined_loss(params_list, x, y):
                # This is more complex in MLX if we don't use a Module.
                # Let's wrap them.
                pass
            
            # Simple way: just update them together
            loss, grads = nn.value_and_grad(user_tower, 
                lambda m, x, y: loss_fn(m, candidate_tower, x, y))(user_tower, xb, yb)
            
            # We also need grads for candidate_tower
            loss_c, grads_c = nn.value_and_grad(candidate_tower,
                lambda m, x, y: loss_fn(user_tower, m, x, y))(candidate_tower, xb, yb)
            
            optimizer.update(user_tower, grads)
            optimizer.update(candidate_tower, grads_c)
            
            mx.eval(user_tower.parameters(), candidate_tower.parameters(), optimizer.state)
            losses.append(float(loss))
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}/{epochs} | Loss: {np.mean(losses):.4f}")

    # Save weights
    weights = {}
    for k, v in user_tower.parameters().items():
        weights[f"user_tower.{k}"] = v
    for k, v in candidate_tower.parameters().items():
        weights[f"candidate_tower.{k}"] = v
    
    mx.savez(str(MODELS_DIR / "two_tower.npz"), **weights)
    print(f"✅ Model saved to {MODELS_DIR / 'two_tower.npz'}")

    # Print predictions for last event
    last_x = mx.array([X[-1]])
    user_emb = user_tower(last_x)
    action_embs = candidate_tower(mx.arange(N_ACTIONS))
    scores = (user_emb @ action_embs.T)[0]
    probs = mx.sigmoid(scores * 10.0).tolist()
    top = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    print("\n   Top predictions for last event:")
    for idx, p in top:
        print(f"      {ACTION_VOCAB[idx]:<20} : {p:.4f}")

if __name__ == "__main__":
    train()
