# 👻 Phantom

A personal AI OS that watches what you do, learns your patterns, and acts before you ask.

> No cloud. No subscriptions. Runs entirely on your machine.

## Vision

Most AI assistants are reactive — you prompt, they respond.  
Phantom is different: it observes your behavior continuously, builds a behavioral model of you, and proactively executes actions on your behalf — with your confirmation.

## Architecture
```
[Daemon] → [Feature Extractor] → [Two-Tower Model] → [Agent]
 observe     raw normalized         LSTM user tower     act + feedback
             features (24d)       + action embeddings   + interviewer RLHF loop
                                    (inspired by X/Phoenix)
```

The daemon captures ~15 behavioral signals every 5 seconds, including window titles for rich contextual grounding. The feature extractor normalizes them into 24-dimensional vectors. A two-tower model (LSTM user tower + action embeddings) scores all candidate actions simultaneously. The agent maintains session continuity, executes the top prediction — or notifies — with a macOS confirmation dialog, and triggers proactive RLHF interview cycles.

## RLHF Loop
Phantom asks you questions during idle moments. Your answers are parsed by Qwen into structured tokens and profile updates. Accept/reject feedback is scored by Qwen as a reward model (-1.0 to +1.0) and used to weight the next training pass.

## Signals captured

| Layer | Signals |
|---|---|
| Context | App, URL, session time, day, Window Title |
| Keyboard | WPM, backspace rate, modifier usage |
| Mouse | Distance, clicks, idle time |
| System | CPU, RAM, battery, network |
| Focus | Composite focus score (0–100) |
| Actions | App launches, git ops, file changes, RLHF feedback |

## Token vocabulary (v2)
```
SESSION:* DAY:* APP:* WEB:* DUR:*
TYPING:* ERROR:* MOUSE:* TABS:* WIN:*
CPU:* FOCUS:* SWITCH:* NET:*
ACT:OPEN_* ACT:GIT_* ACT:FILE_*
CLIP:* FILE:* MODE:* BAT:*
RLHF:* PROJECT:* CONTEXT:*
```

## Action layer

When the model predicts an `ACT:*` token with >70% confidence, Phantom asks for confirmation via a native macOS dialog before executing. Accept/reject signals are logged for future reward-based fine-tuning.

## Stack

- **Architecture** — Two-Tower Behavioral Recommender (inspired by xai-org/x-algorithm)
- **Observation** — Python daemon, NSWorkspace, pynput, psutil, Quartz
- **Learning** — MLX Two-Tower (LSTM + Embeddings), Apple Silicon (M3 Max 128GB)
- **Reward Model** — Qwen3.5-122B local (llama-server, 4 slots)
- **Inference** — on-device, <3s retrain cycle
- **Action** — osascript, subprocess, macOS native dialogs
- **Feedback** — JSONL reward log for future RL loop

## 📊 Status

- **Events captured**: 1,251+ behavior snapshots
- **Vocabulary**: 85 unique behavioral tokens
- **Architecture**: V2.2 live — RLHF loop, Qwen reward model, 4 slots
- **RLHF**: Active Interviewer loop with Qwen-122B Reward Model (v2.2)

## 🧠 Advanced Features (v2.2)

### 👻 Proactive Interviewer
Phantom detects "Focus: Shallow" or "Switch: Fast" moments to trigger a local dialogue via `osascript`. It asks targeted questions to refine your profile.

### 🏆 Qwen Reward Model
Instead of binary feedback, all user interactions are scored by a local **Qwen-122B** instance. 
- **Thinking tags**: Automatically strips `<think>` blocks for clean JSON parsing.
- **Weighted Training**: Rewards (-1.0 to 1.0) are used as sample weights in the BCE loss for the Two-Tower model.
- **NLP Context**: Interviewer answers are parsed into structured tokens (`PROJECT:X`, `CONTEXT:Y`) and injected into the Transformer's context window.

### ⚙️ Auto-Retrain & Hot Reload
A background thread in `agent.py` monitors feedback. Every 2 hours, it triggers a full retraining of the model suite and reloads weights into memory without interrupting the agent.

## Structure
```
phantom/
├── daemon/       # collector_v2.py — behavioral event logger
├── trainer/      
│   ├── feature_extractor.py  # raw events → normalized features
│   ├── two_tower.py          # MLX Two-Tower model definition
│   └── train_v2.py           # V2.2 orchestrator + RLHF Reward Model
├── agent/        
│   ├── agent.py          # Two-Tower + transformer + auto-retrain
│   ├── interviewer.py    # Proactive RLHF dialogue loop
│   └── context_builder.py # Centralized Qwen context manager
└── data/         # events/, features/, feedback/ (gitignored)
```

---

*Built on an M3 Max, in Paris, at night.*
