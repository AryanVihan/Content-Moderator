"""
LLM connectivity diagnostic — run BEFORE inference.py.
Usage: py test_llm.py

Set these env vars first (PowerShell):
  $env:OPENAI_API_KEY = "hf_xxxx"
  $env:API_BASE_URL   = "https://router.huggingface.co/v1"
  $env:MODEL_NAME     = "Qwen/Qwen2.5-72B-Instruct"
"""
import os
import sys
import json
import httpx
from openai import OpenAI

API_KEY      = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

if not API_KEY and HF_TOKEN:
    API_KEY = HF_TOKEN

print("=" * 60)
print("LLM Connectivity Diagnostic")
print("=" * 60)
print(f"  API_BASE_URL : {API_BASE_URL}")
print(f"  MODEL_NAME   : {MODEL_NAME}")
print(f"  API_KEY set  : {'YES (' + API_KEY[:8] + '...)' if API_KEY else 'NO — this is the problem!'}")
print()

if not API_KEY:
    print("ERROR: No API key set.")
    print()
    print("Fix (PowerShell):")
    print('  $env:OPENAI_API_KEY = "hf_your_token_here"')
    print('  $env:API_BASE_URL   = "https://router.huggingface.co/v1"')
    print('  $env:MODEL_NAME     = "Qwen/Qwen2.5-72B-Instruct"')
    print()
    print("Get a free HuggingFace token at: https://huggingface.co/settings/tokens")
    sys.exit(1)

# ── Test 1: Raw HTTP call to the endpoint ────────────────
print("[1] Raw HTTP POST to LLM endpoint ...")
try:
    r = httpx.post(
        f"{API_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with only the word: PONG"}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
        timeout=30,
    )
    print(f"  HTTP status : {r.status_code}")
    if r.status_code == 200:
        print(f"  Response    : {r.text[:300]}")
        print("  [PASS] Raw HTTP call succeeded")
    elif r.status_code == 401:
        print("  [FAIL] 401 Unauthorized — API key is wrong or expired")
        print(f"  Body: {r.text[:300]}")
        sys.exit(1)
    elif r.status_code == 429:
        print("  [FAIL] 429 Rate Limited — wait a minute and try again, or use a different model")
        print(f"  Body: {r.text[:300]}")
        sys.exit(1)
    elif r.status_code == 404:
        print("  [FAIL] 404 — Model not found at this URL")
        print(f"  Body: {r.text[:300]}")
        print()
        print("  Try one of these API_BASE_URL values:")
        print('    https://router.huggingface.co/v1')
        print('    https://router.huggingface.co/v1')
        sys.exit(1)
    else:
        print(f"  [FAIL] Unexpected status {r.status_code}")
        print(f"  Body: {r.text[:500]}")
        sys.exit(1)
except httpx.ConnectError as e:
    print(f"  [FAIL] Cannot connect: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
    sys.exit(1)

# ── Test 2: OpenAI client (same as inference.py uses) ────
print()
print("[2] OpenAI client call (exact same path as inference.py) ...")
try:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a JSON API. Always respond with valid JSON only."},
            {"role": "user", "content": (
                'Respond with exactly this JSON and nothing else:\n'
                '{"action_type": "REMOVE", "target_item_id": "test_001", '
                '"policy_violated": "HATE_SPEECH", "reasoning": "test", "confidence": 0.9}'
            )},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content or ""
    print(f"  Raw response: {raw[:300]}")
    # Try to parse it
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    parsed = json.loads(cleaned)
    print(f"  Parsed action_type: {parsed.get('action_type')}")
    print("  [PASS] OpenAI client + JSON parse successful")
except json.JSONDecodeError as e:
    print(f"  [FAIL] JSON parse error: {e}")
    print(f"  Raw was: {raw[:300]}")
    print()
    print("  The model responded but not with valid JSON.")
    print("  Try a different model — some free models ignore the JSON instruction.")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

# ── Test 3: Model alternatives if Qwen fails ─────────────
print()
print("[3] Alternative free models to try (if Qwen is rate-limited):")
alternatives = [
    ("meta-llama/Llama-3.1-8B-Instruct",  "Fast, good JSON compliance"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Good for structured output"),
    ("microsoft/Phi-3-mini-4k-instruct",   "Very fast, small"),
    ("Qwen/Qwen2.5-72B-Instruct",          "Best quality, may be slow"),
]
for model, note in alternatives:
    print(f"  $env:MODEL_NAME = \"{model}\"   # {note}")

print()
print("=" * 60)
print("If Test 2 passed: run inference.py normally.")
print("If Test 1 or 2 failed: fix the env var shown above first.")
print("=" * 60)
