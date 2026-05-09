import csv
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback_log.csv"

def ensure_feedback_file():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "timestamp",
                "question",
                "answer",
                "evidence",
                "sources",
                "feedback"
            ])
            writer.writeheader()

def format_sources_for_csv(sources):
    formatted = []
    for item in sources:
        formatted.append(
            f"{item['source']} | Page {item['page']} | Score {item['score']} | {item['snippet']}"
        )
    return " || ".join(formatted)

def save_interaction(question, answer, evidence, sources, feedback="pending"):
    ensure_feedback_file()

    row = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "sources": format_sources_for_csv(sources),
        "feedback": feedback
    }

    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "timestamp",
            "question",
            "answer",
            "evidence",
            "sources",
            "feedback"
        ])
        writer.writerow(row)