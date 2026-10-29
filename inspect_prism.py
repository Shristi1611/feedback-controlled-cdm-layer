"""Look at the real structure of the PRISM dataset before building the test.
Downloads two small files from Hugging Face (public data) and prints their columns.

  python inspect_prism.py
"""
import json

from huggingface_hub import hf_hub_download

REPO = "HannahRoseKirk/prism-alignment"


def peek(filename, n_rows=2):
    path = hf_hub_download(REPO, filename, repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"\n===== {filename}: {len(rows)} rows =====")
    print("columns:", list(rows[0].keys()))
    for r in rows[:n_rows]:
        print("--- example row ---")
        for k, v in r.items():
            s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            print(f"  {k}: {s[:160]}")
    return rows


survey = peek("survey.jsonl", 1)
utt = peek("utterances.jsonl", 2)

users = {r.get("user_id") for r in utt}
print(f"\nusers with rated utterances: {len(users)}")
per_user = {}
for r in utt:
    per_user[r.get("user_id")] = per_user.get(r.get("user_id"), 0) + 1
counts = sorted(per_user.values())
print("rated responses per user: min", counts[0], "median", counts[len(counts) // 2], "max", counts[-1])