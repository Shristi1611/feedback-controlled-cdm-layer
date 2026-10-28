"""Keyword rule vs Laya on 20 harder messages, each with a label I wrote by hand."""
import os
os.environ["USE_TF"] = "0"   # avoids a known hang if TensorFlow is installed

import laya
from fccl import FCCL

# (message, true label). Many deliberately avoid the keyword lists, and a few contain
# a keyword inside another word ("border" has "order", "feeling" has "feel").
DATA = [
    ("I can't believe my boss did that again, I'm done with this place", "chatty"),
    ("Nothing is going right lately and I just want to talk", "chatty"),
    ("My mom is in the hospital and I don't know what to do", "chatty"),
    ("Honestly it's been a rough year, I miss my old friends", "chatty"),
    ("lol that movie was wild, did you see the ending", "chatty"),
    ("The border crossing was hell and I'm completely exhausted", "chatty"),
    ("I'd order a pizza but I'm too sad to even move", "chatty"),
    ("Can you remind me to pick up the parcel on Thursday", "logistics"),
    ("What time does the train to Pune leave tomorrow", "logistics"),
    ("My flight leaves at 6 AM from Terminal 2, I need a cab", "logistics"),
    ("Where do I send the form and by when", "logistics"),
    ("Reschedule my dentist appointment to next week", "logistics"),
    ("Move the call with the client to 3 PM and tell the team", "logistics"),
    ("How much is the electricity bill due on the 15th", "logistics"),
    ("Tell me about the history of Rome", "other"),
    ("Write a short poem about the sea", "other"),
    ("Explain how photosynthesis works", "other"),
    ("What's a good name for a pet turtle", "other"),
    ("Translate 'good morning' into Spanish", "other"),
    ("Is Python or JavaScript easier to learn", "other"),
]

LABEL_OF_WEIGHT = {0.3: "chatty", 0.8: "logistics", 0.5: "other"}   # keyword rule outputs
questions = {
    "mode": {
        "type": "choice",
        "instructions": "What is the user mainly doing in this message?",
        "criteria": {
            "chatty": "venting, sharing feelings, casual conversation, ranting",
            "logistics": "scheduling, orders, deadlines, addresses, practical tasks",
            "other": "anything else",
        },
    }
}

print("loading model...", flush=True)
agent = laya.load("convaiinnovations/laya")
print("model loaded\n", flush=True)

keyword_rule = FCCL()
kw_right = laya_right = 0
lines = []
print(f"{'true':<10}{'keywords':<11}{'laya':<10}{'conf':<8}message")
for text, truth in DATA:
    kw = LABEL_OF_WEIGHT[keyword_rule.context_weight(text)]
    ans = agent.predict(text, questions)["answers"]["mode"]
    ly, conf = ans.get("choice"), ans.get("confidence")
    kw_right += kw == truth
    laya_right += ly == truth
    line = f"{truth:<10}{kw + (' ok' if kw == truth else ' X'):<11}{str(ly) + (' ok' if ly == truth else ' X'):<10}{conf:<8.3f}{text}"
    print(line, flush=True)
    lines.append(line)

summary = (f"\nkeyword rule correct: {kw_right}/{len(DATA)}"
           f"\nlaya correct:         {laya_right}/{len(DATA)}")
print(summary)
with open("compare_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + summary)