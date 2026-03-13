# 👻 Phantom

A personal AI OS that watches what you do, learns your patterns, and acts before you ask.

> No cloud. No subscriptions. Runs entirely on your machine.

## Vision

Most AI assistants are reactive — you prompt, they respond.  
Phantom is different: it observes your behavior continuously, builds a behavioral model of you, and proactively executes actions on your behalf — with your confirmation.

## Architecture
```
[Daemon] → [Tokenizer] → [Transformer] → [Agent]
 observe     sequences     MLX train      act + feedback
```

The daemon captures ~15 behavioral signals every 5 seconds. The tokenizer converts them into a compact sequence language. A small transformer (128d, 4 layers, MLX) trains on these sequences and predicts your next action. The agent executes — or notifies — with a macOS confirmation dialog.

## Signals captured

| Layer | Signals |
|---|---|
| Context | App, URL, session time, day |
| Keyboard | WPM, backspace rate, modifier usage |
| Mouse | Distance, clicks, idle time |
| System | CPU, RAM, battery, network |
| Focus | Composite focus score (0–100) |
| Actions | App launches, git ops, file changes |

## Token vocabulary (v2)
```
SESSION:* DAY:* APP:* WEB:* DUR:*
TYPING:* ERROR:* MOUSE:* TABS:* WIN:*
CPU:* FOCUS:* SWITCH:* NET:*
ACT:OPEN_* ACT:GIT_* ACT:FILE_*
CLIP:* FILE:* MODE:* BAT:*
```

## Action layer

When the model predicts an `ACT:*` token with >70% confidence, Phantom asks for confirmation via a native macOS dialog before executing. Accept/reject signals are logged for future reward-based fine-tuning.

## Stack

- **Observation** — Python daemon, NSWorkspace, pynput, psutil, Quartz
- **Learning** — MLX transformer, Apple Silicon (M3 Max 128GB)
- **Inference** — on-device, <3s retrain cycle
- **Action** — osascript, subprocess, macOS native dialogs
- **Feedback** — JSONL reward log for future RL loop

## Structure
```
phantom/
├── daemon/       # collector_v2.py — behavioral event logger
├── tokenizer/    # tokenizer_v2.py — event → token sequences  
├── trainer/      # train.py — MLX transformer
├── agent/        # agent.py + feedback_logger.py
└── data/         # events/, sequences_v2/, feedback/ (gitignored)
```

## Status

Currently in active self-training. Retrains daily as behavioral data accumulates.  
Loss: 0.45 → target <0.30 with 5000+ sequences.

---

*Built on an M3 Max, in Paris, at night.*
