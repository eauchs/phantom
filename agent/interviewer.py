#!/usr/bin/env python3
import json
import time
import subprocess
import sys
import requests
import random
from datetime import datetime
from pathlib import Path
from agent.context_builder import build_context

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"
FEEDBACK_DIR = DATA_DIR / "feedback"
PROFILE_DIR = DATA_DIR / "profile"
PROFILE_FILE = PROFILE_DIR / "profile.json"
ANSWERS_FILE = PROFILE_DIR / "answers.jsonl"

LLAMA_BASE_URL = "http://127.0.0.1:8080"
LLAMA_URL = f"{LLAMA_BASE_URL}/v1/chat/completions"
HEALTH_URL = f"{LLAMA_BASE_URL}/health"

FALLBACK_QUESTIONS = [
    "Sur quoi travailles-tu en ce moment ?",
    "Comment puis-je t'aider à être plus productif ?",
    "Y a-t-il une tâche répétitive que je devrais automatiser ?",
    "Quel est ton objectif principal pour cette session ?",
    "Comment te sens-tu par rapport à ta charge de travail actuelle ?"
]

def get_last_events(n=10) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    f = EVENTS_DIR / f"{today}.jsonl"
    if not f.exists(): return []
    
    try:
        lines = f.read_text().strip().split("\n")
        events = []
        for l in lines[-n:]:
            if l.strip():
                try: events.append(json.loads(l))
                except: pass
        return events
    except:
        return []

def get_profile_summary():
    if not PROFILE_FILE.exists():
        return "New user, no profile data."
    try:
        profile = json.loads(PROFILE_FILE.read_text())
        summary = []
        if profile.get("preferences"):
            summary.append(f"Preferences: {json.dumps(profile['preferences'])}")
        if profile.get("avoid"):
            summary.append(f"Avoid: {json.dumps(profile['avoid'])}")
        if profile.get("context"):
            summary.append(f"Context: {json.dumps(profile['context'])}")
        return " | ".join(summary) if summary else "Empty profile."
    except:
        return "Error reading profile."

def ask_llm(last_5_tokens, profile_summary):

    if not check_health():
        print("[INTERVIEWER] llama-server not healthy, using fallback.")
        return random.choice(FALLBACK_QUESTIONS)

    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
    context = build_context()
    user_content = json.dumps(context)
    
    if not user_content or not user_content.strip() or user_content == "{}":
        print("[INTERVIEWER] Empty context, using fallback.")
        return random.choice(FALLBACK_QUESTIONS)

    system_prompt = (
        f"You are Phantom, a personal AI OS observing {user}'s behavior.\n"
        f"Current window: {context.get('window_title', 'Unknown')}\n"
        "Generate ONE short, specific question in French to better understand\n"
        "this person's preferences, goals, or current state.\n"
        "Max 15 words. No preamble. Just the question."
    )
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7,
        "max_tokens": -1,

    }
    
    # ── Guard: verify user query is not empty ──
    user_msg = next((m["content"] for m in payload["messages"] if m["role"] == "user"), None)
    if not user_msg or not user_msg.strip():
        print(f"[INTERVIEWER] Error: No user query found. Payload: {payload}")
        return random.choice(FALLBACK_QUESTIONS)

    print(f"[LLM DEBUG] ask_llm sending {len(payload['messages'])} messages")
    for m in payload["messages"]:
        print(f"  role={m['role']} content_len={len(str(m.get('content','')))}")

    try:
        res = call_qwen(payload)
        if res:
            content = res["choices"][0]["message"]["content"]
            # Extract between <think> tags if present
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()
            elif "</think>" in content:
                content = content.split("</think>")[-1].strip()
            
            return content.strip().strip('"')
    except Exception as e:
        print(f"[INTERVIEWER] LLM Error: {e}")
            
    return random.choice(FALLBACK_QUESTIONS)

def show_dialog(question):
    script = f'display dialog "{question}" default answer "" with title "👻 Phantom te pose une question"'
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if "button returned:OK" in r.stdout:
            # Extract text input
            # Output format: button returned:OK, text returned:some answer
            parts = r.stdout.strip().split("text returned:")
            if len(parts) > 1:
                return parts[1]
    except:
        pass
    return None

def save_answer(question, answer, context_tokens):
    entry = {
        "ts": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "context_tokens": context_tokens
    }
    with open(ANSWERS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    # Parse answer for structured context
    parse_answer_with_qwen(question, answer)

def parse_answer_with_qwen(question, answer):
    if not check_health():
        return
    
    system_prompt = (
        "You are a behavioral context extractor for a personal AI OS. "
        "Given a user's answer to a question, extract structured context. "
        "Return ONLY JSON (no preamble): "
        "{ "
        "  'tokens': ['PROJECT:X', 'CONTEXT:Y'], "
        "  'profile_update': {'key': 'value'}, "
        "  'reward_modifier': float between -1.0 and 1.0 "
        "} "
        "reward_modifier = how much this context should amplify/dampen "
        "interruptions right now. -1.0 = never interrupt, +1.0 = open. "
        "Strip <think> tags before parsing."
    )
    user_content = f"question={question}\nanswer={answer}"
    if not answer or not answer.strip():
        return # Nothing to parse
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": -1,

        "temperature": 0.0
    }
    
    # ── Guard: verify user query is not empty ──
    user_msg = next((m["content"] for m in payload["messages"] if m["role"] == "user"), None)
    if not user_msg or not user_msg.strip():
        print(f"[INTERVIEWER] NLP Error: No user query found. Payload: {payload}")
        return

    print(f"[LLM DEBUG] parse_answer_with_qwen sending {len(payload['messages'])} messages")
    for m in payload["messages"]:
        print(f"  role={m['role']} content_len={len(str(m.get('content','')))}")

    try:
        res = call_qwen(payload)
        if res:
            content = res["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            
            # Find the JSON part
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start:end+1].replace("'", "\"")
                data = json.loads(json_str)
                
                tokens = data.get("tokens", [])
                profile_update = data.get("profile_update", {})
                reward_modifier = data.get("reward_modifier", 0.0)
                
                apply_nlp_insights(tokens, profile_update, reward_modifier)
    except Exception as e:
        print(f"[INTERVIEWER] NLP Parsing error: {e}")

def apply_nlp_insights(tokens, profile_update, reward_modifier):
    try:
        if not PROFILE_FILE.exists():
            profile = {"preferences": {}, "avoid": {}, "context": {}, "action_feedback": {}}
        else:
            profile = json.loads(PROFILE_FILE.read_text())
        
        # 1. Store tokens for injection (temporary storage)
        TOKEN_INJECTION_FILE = PROFILE_DIR / "injected_tokens.json"
        TOKEN_INJECTION_FILE.write_text(json.dumps({"tokens": tokens, "ts": time.time()}))
        
        # 2. Merge profile_update
        if "preferences" not in profile: profile["preferences"] = {}
        profile["preferences"].update(profile_update)
        
        # 3. Store reward_modifier with timestamp
        profile["current_session_modifier"] = {
            "value": reward_modifier,
            "ts": time.time()
        }
        
        PROFILE_FILE.write_text(json.dumps(profile, indent=2))
        print(f"[INTERVIEWER] NLP Insights applied: {len(tokens)} tokens, reward_mod={reward_modifier:+.2f}")
    except Exception as e:
        print(f"[INTERVIEWER] Error applying NLP insights: {e}")

def update_profile_with_insight(question, answer):
    if not PROFILE_FILE.exists():
        profile = {"preferences": {}, "avoid": {}, "context": {}, "action_feedback": {}}
    else:
        profile = json.loads(PROFILE_FILE.read_text())
    
    # Simple heuristic: if question mentions "préfères" or similar, add to preferences
    # In a real scenario, we'd use the LLM to extract the insight.
    # For now, let's just store it in context as "last_interaction"
    profile["context"]["last_interaction"] = {"q": question, "a": answer, "ts": datetime.now().isoformat()}
    
    PROFILE_FILE.write_text(json.dumps(profile, indent=2))

def check_rejected_actions():
    # Read data/feedback/*.jsonl
    rejected_counts = {}
    for f in FEEDBACK_DIR.glob("*.jsonl"):
        try:
            for line in f.read_text().strip().split("\n"):
                if not line: continue
                entry = json.loads(line)
                action = entry.get("action")
                accepted = entry.get("accepted")
                if not accepted:
                    rejected_counts[action] = rejected_counts.get(action, 0) + 1
        except:
            continue
    
    if not rejected_counts:
        return

    # Check profile to see if we already asked today
    try:
        profile = json.loads(PROFILE_FILE.read_text())
    except:
        profile = {"preferences": {}, "avoid": {}, "context": {}, "action_feedback": {}}

    action_feedback = profile.get("action_feedback", {})
    today = datetime.now().strftime("%Y-%m-%d")

    for action, count in rejected_counts.items():
        last_asked = action_feedback.get(action, {}).get("last_asked")
        if last_asked == today:
            continue
        
        # Ask why
        question = f"Tu as refusé {action} {count} fois. Pourquoi ?"
        answer = show_dialog(question)
        
        if answer:
            save_answer(question, answer, [])
            if action not in action_feedback:
                action_feedback[action] = {}
            action_feedback[action].update({
                "rejected": count,
                "reason": answer,
                "last_asked": today
            })
            profile["action_feedback"] = action_feedback
            PROFILE_FILE.write_text(json.dumps(profile, indent=2))
            break # Only ask one per poll

def interviewer_loop():
    print("👻 Phantom Interviewer started.")
    last_question_time = 0
    
    while True:
        try:
            now = datetime.now()
            hour = now.hour
            
            # Condition: Hour between 10 and 2 (10 AM to 2 AM)
            # 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1
            is_active_hours = (hour >= 10 or hour < 2)
            
            # Condition: No question in last 30 minutes
            time_since_last = time.time() - last_question_time
            is_cooldown_over = time_since_last > 1800 # 30 mins
            
            if is_active_hours and is_cooldown_over:
                events = get_last_events(10)
                from tokenizer.tokenizer_v2 import tokenize_event
                
                last_5_tokens = []
                for ev in events[-5:]:
                    last_5_tokens.extend(tokenize_event(ev).split())
                
                trigger = any(t in ["FOCUS:SHALLOW", "SWITCH:FAST"] for t in last_5_tokens)
                
                if trigger:
                    profile_summary = get_profile_summary()
                    question = ask_llm(last_5_tokens, profile_summary)
                    if question:
                        answer = show_dialog(question)
                        if answer:
                            save_answer(question, answer, last_5_tokens)
                            update_profile_with_insight(question, answer)
                            last_question_time = time.time()
            
            # Also check rejected actions
            check_rejected_actions()
            
            time.sleep(60)
        except Exception as e:
            print(f"[INTERVIEWER] Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    interviewer_loop()
