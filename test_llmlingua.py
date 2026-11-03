"""Retry: does LLMLingua-2 run now, and does force_tokens protect our IDs?"""
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",          # our PyTorch is the CPU-only build
)

text = (
    "Hi, I wanted to check on the order I placed last week. The invoice number is 4821 "
    "and it was supposed to be delivered to the address on Park Street by Friday. "
    "I haven't received any tracking update, and the meeting with Rahul about the "
    "delivery schedule is tomorrow at 4 PM. The package contains a replacement part for my "
    "laptop charger, model number X540, and without it I cannot work from home."
)
protected = ["4821", "Rahul", "Park", "Street", "X540", "PM"]

for rate in (1.0, 0.7, 0.5, 0.3):
    out = compressor.compress_prompt_llmlingua2(text, rate=rate, force_tokens=protected)
    comp = out["compressed_prompt"]
    lost = [w for w in protected if w.lower() not in comp.lower()]
    print(f"\n--- keep {rate} ---  {out['compressed_tokens']} of {out['origin_tokens']} tokens   lost: {lost or '-'}")
    print(comp)