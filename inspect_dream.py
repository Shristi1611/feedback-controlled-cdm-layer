"""Look at the real structure of the DREAM dialogue dataset (multiple-choice questions about conversations).

  python inspect_dream.py
"""
import statistics

from datasets import load_dataset

REPO = "dataset-org/dream"
ds = None
# The original dataset uses a loading script, which new `datasets` versions refuse.
# Hugging Face keeps an automatic parquet copy on a separate branch, with config "default".
attempts = [
    ("parquet copy", dict(path=REPO, name="default", revision="refs/convert/parquet")),
    ("original", dict(path=REPO, name="plain_text")),
]
for label, kwargs in attempts:
    try:
        ds = load_dataset(split="validation", **kwargs)
        print(f"loaded: {label}")
        break
    except Exception as e:
        print(f"load failed ({label}): {type(e).__name__}: {str(e)[:300]}")
if ds is None:
    raise SystemExit("Could not load DREAM. Paste me the messages above.")

print("\ncolumns:", ds.column_names, "| rows:", len(ds))
ex = ds[0]
print("\n--- example ---")
for k, v in ex.items():
    print(f"  {k}: {str(v)[:300]}")

words = [len(" ".join(r["dialogue"]).split()) for r in ds]
turns = [len(r["dialogue"]) for r in ds]
print(f"\ndialogue length in words: min {min(words)}, median {statistics.median(words)}, "
      f"mean {statistics.mean(words):.0f}, max {max(words)}")
print(f"turns per dialogue: min {min(turns)}, median {statistics.median(turns)}, max {max(turns)}")
print(f"unique dialogues: {len(set(r['dialogue_id'] for r in ds))}, questions: {len(ds)}")
print("choices per question:", sorted({len(r['choice']) for r in ds}))
in_choices = sum(r["answer"] in r["choice"] for r in ds)
print(f"answer text is exactly one of the choices: {in_choices}/{len(ds)}")