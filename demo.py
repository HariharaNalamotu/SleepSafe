"""Run locally with: python demo.py"""
import json

from sleepsafe.agent import answer_question
from sleepsafe.classifier import classify_sleep_session

SESSION = "a3f9c2e1-lapel-0926"
PROMPTS = [
    "What happened around 12:18 AM?",
    "What was the longest apnea event?",
    "How much of the night had snoring?",
    "Were there any low-quality recording periods?",
    "Why was this session classified this way?",
    "What events were detected?",
    "Which detections are zero-shot?",
    "What was the most concerning period?",
    "How was this night different from the previous night?",
]

print(json.dumps(classify_sleep_session(SESSION), indent=2))
for prompt in PROMPTS:
    print(f"\nQ: {prompt}\nA: {answer_question(SESSION, prompt)}")
