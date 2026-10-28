"""Try Laya as the conversation-type detector for FCCL."""
import os
os.environ["USE_TF"] = "0"   # avoids a known hang if TensorFlow is installed

import laya

print("loading model...", flush=True)
agent = laya.load("convaiinnovations/laya")   # English checkpoint
print("model loaded", flush=True)

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

texts = [
    "Ugh I feel so stressed about work today",
    "I'm so tired and annoyed with everything lately",
    "Honestly I just need to rant about my week",
    "Book the 4 PM meeting with Rahul and send the invoice",
    "What is the deadline for the order delivery",
    "Schedule the ticket pickup at the address",
    "I'm so stressed, can you move my 4 PM meeting to Friday?",
    "What's the capital of France?",
]

lines = []
for t in texts:
    result = agent.predict(t, questions)
    ans = result["answers"]["mode"]
    line = f"{str(ans.get('choice')):<10} conf={ans.get('confidence')}   {t}"
    print(line, flush=True)
    lines.append(line)

with open("laya_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("done - results also saved in laya_results.txt", flush=True)