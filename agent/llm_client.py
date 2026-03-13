import requests
import threading
from typing import Optional, Dict

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8080/health"
_semaphore = threading.Semaphore(4)  # match -np 4

def call_qwen(payload: Dict, timeout=120) -> Optional[Dict]:
    """
    Thread-safe Qwen call. Waits for a free slot (max 4 concurrent).
    Returns parsed response dict or None on failure.
    """
    with _semaphore:
        try:
            r = requests.post(LLAMA_URL, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            print(f"[LLM] Error {r.status_code}: {r.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[LLM] Timeout after {timeout}s")
        except Exception as e:
            print(f"[LLM] Error: {e}")
    return None

def check_health(timeout=10) -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("status") == "ok"
    except:
        pass
    return False
