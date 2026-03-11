# 👻 Phantom

> A personal AI OS that watches what you do, learns your patterns, and acts before you ask.

## Vision

Most AI assistants are reactive — you prompt, they respond.
Phantom is different: it observes your behavior continuously, builds a model of *you*, and proactively acts on your behalf.

No cloud. No subscriptions. Runs entirely on your machine.

## Architecture
```
┌─────────────────────────────────────────────────────┐
│                    PHANTOM                          │
│                                                     │
│  [Daemon] → [Tokenizer] → [Transformer] → [Agent]  │
│     ↑                          ↓              ↓     │
│  observe                    predict          act    │
│  everything                 next action   pre-fetch │
│                                           telegram  │
│                                           pre-draft │
└─────────────────────────────────────────────────────┘
```

## Stack

- **Observation** — macOS daemon (Python, NSWorkspace) — apps, URLs, files, clipboard
- **Learning** — MLX fine-tuning on Apple Silicon (M3 Max 128GB)
- **Inference** — llama-server local, quantized models
- **Action** — MCP tools (mail, browser, Telegram), Codex CLI

## Roadmap

- [ ] `daemon/` — behavioral event collector
- [ ] `tokenizer/` — event → sequence representation
- [ ] `trainer/` — online fine-tuning pipeline (MLX)
- [ ] `agent/` — proactive action layer
- [ ] `self_model/` — dynamic user profile

## Author

[eauchs](https://github.com/eauchs) — built on a M3 Max, in Paris, at night.
