"""One real call to the ScaleDown compress API, to see the raw response and the token cost."""
import os
import requests

key = os.environ.get("SCALEDOWN_API_KEY")
if not key:
    raise SystemExit("SCALEDOWN_API_KEY is not set in this terminal. Run the 'set' command first.")

context = (
    "Hi, I wanted to check on the order I placed last week. The invoice number is 4821 "
    "and it was supposed to be delivered to the address on Park Street by Friday. "
    "I haven't received any tracking update, and the meeting with Rahul about the "
    "delivery schedule is tomorrow at 4 PM, so I really need to know where it is."
)
body = {
    "context": context,
    "prompt": "Where is my order and when will it arrive?",
    "scaledown": {"rate": 0.5},        # ScaleDown's rate = fraction REMOVED
}

r = requests.post("https://api.scaledown.xyz/compress/raw/", json=body,
                  headers={"x-api-key": key}, timeout=60)
print("status:", r.status_code)
try:
    data = r.json()
except Exception:
    print(r.text[:500])
    raise SystemExit
res = data.get("results", {})
print("compressed:", res.get("compressed_prompt"))
print("original tokens:", res.get("original_prompt_tokens"),
      "-> compressed tokens:", res.get("compressed_prompt_tokens"))
print("compression ratio:", res.get("compression_ratio"))
print("input_tokens (billed?):", data.get("input_tokens"))
if r.status_code != 200:
    print(data)