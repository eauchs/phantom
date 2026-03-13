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
        # LSTM in MLX returns (h_seq, c_seq) for 3D inputs
        h_seq, _ = self.lstm(h)
        # take last hidden state
        h_last = h_seq[:, -1, :]
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

class TwoTower(nn.Module):
    def __init__(self, user_tower, candidate_tower):
        super().__init__()
        self.user_tower = user_tower
        self.candidate_tower = candidate_tower

    def __call__(self, x, y):
        user_emb = self.user_tower(x)
        all_actions = mx.arange(N_ACTIONS)
        action_embs = self.candidate_tower(all_actions)
        scores = user_emb @ action_embs.T
        logits = scores * 10.0
        return nn.losses.binary_cross_entropy(logits, y).mean()

def train():
    print("🧠 Training Two-Tower Recommender...")
    X, Y = load_data()
    if len(X) == 0:
        print("❌ No data to train on.")
        return

    model = TwoTower(UserTower(), CandidateTower(N_ACTIONS))
    
    # Optimizer
    optimizer = optim.AdamW(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m(x, y))
    
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
            
            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(float(loss))
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}/{epochs} | Loss: {np.mean(losses):.4f}")

    # Save weights
    model.save_weights(str(MODELS_DIR / "two_tower.npz"))
    print(f"✅ Model saved to {MODELS_DIR / 'two_tower.npz'}")

    # Print predictions for last event
    last_x = mx.array(X[-1:][0])
    # add batch dim
    last_x = mx.array(np.array([X[-1]]))
    user_emb = model.user_tower(last_x)
    action_embs = model.candidate_tower(mx.arange(N_ACTIONS))
    scores = (user_emb @ action_embs.T)[0]
    probs = mx.sigmoid(scores * 10.0).tolist()
    top = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    print("\n   Top predictions for last event:")
    for idx, p in top:
        print(f"      {ACTION_VOCAB[idx]:<20} : {p:.4f}")

if __name__ == "__main__":
    train()
