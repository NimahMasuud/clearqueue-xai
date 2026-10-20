"""Builds the synthetic complaint set. Labels follow the severity rubric in the README."""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (what the complaint says, department/queue, base priority: 0 Low, 1 Medium, 2 High)
ISSUES = [
    ("a burst water pipe is flooding the road", "Water & Sanitation", 1),
    ("we have had no running water", "Water & Sanitation", 1),
    ("the tap water is brown and people are falling sick", "Water & Sanitation", 2),
    ("a blocked drain is overflowing", "Water & Sanitation", 1),
    ("sewage is flowing into the street", "Water & Sanitation", 2),
    ("the water bill looks wrong", "Water & Sanitation", 0),
    ("the public tap handle is broken", "Water & Sanitation", 0),
    ("a live wire is hanging low after the storm", "Power Distribution", 2),
    ("the transformer is sparking", "Power Distribution", 2),
    ("we have had no electricity", "Power Distribution", 1),
    ("the electricity meter reading is wrong", "Power Distribution", 0),
    ("an electric pole is leaning to one side", "Power Distribution", 1),
    ("there is a large pothole", "Roads & Transport", 0),
    ("the traffic light is not working", "Roads & Transport", 1),
    ("part of the road has collapsed", "Roads & Transport", 2),
    ("the road is flooded and cars are stuck", "Roads & Transport", 1),
    ("an abandoned vehicle is blocking the road", "Roads & Transport", 0),
    ("the bridge railing is broken", "Roads & Transport", 1),
    ("refuse has not been collected", "Waste Management", 0),
    ("someone is dumping waste illegally", "Waste Management", 0),
    ("waste is being burned and smoke is filling the area", "Waste Management", 1),
    ("the public bins are overflowing", "Waste Management", 0),
    ("there is a strong smell of gas", "Emergency & Public Safety", 2),
    ("a building is on fire", "Emergency & Public Safety", 2),
    ("armed robbers are operating", "Emergency & Public Safety", 2),
    ("a cracked building looks ready to collapse", "Emergency & Public Safety", 2),
    ("the street lights are off", "Emergency & Public Safety", 0),
    ("the clinic has run out of drugs", "Health Services", 1),
    ("many people have the same fever and vomiting", "Health Services", 2),
    ("the ambulance has not arrived", "Health Services", 2),
    ("the clinic is dirty and unsafe", "Health Services", 1),
    ("the clinic opening hours are not displayed", "Health Services", 0),
    ("I have not received my voter card", "Electoral Services Desk", 0),
    ("I cannot find my name on the voter list", "Electoral Services Desk", 1),
    ("the registration centre has no staff", "Electoral Services Desk", 1),
]
OPENERS = ["Please help,", "Good morning,", "I want to report that", "Hello,", "We have a problem:", ""]
VULNERABLE = ["near the primary school", "next to the health centre", "beside the busy market",
              "and elderly neighbours are affected", "and young children live around here"]
PLAIN = ["on our street", "in our estate", "along the main road", "behind the bus stop", "in our area"]
PLACES = ["Ikeja", "Surulere", "Gwarinpa", "Trans-Amadi", "Bodija", "Nsukka", "Rumuola", "Garki"]
DURATIONS = ["for three days", "since morning", "for two weeks", "since last night", ""]
FILLERS = ["Please act quickly.", "This is urgent.", "Thank you.", "I have reported this before.", "Kindly assist.", ""]
LEVELS = ["Low", "Medium", "High"]


def main(n=3000, noise=0.06):
    random.seed(7)
    rows = []
    for _ in range(n):
        issue, dept, base = random.choice(ISSUES)
        vulnerable = random.random() < 0.4
        where = random.choice(VULNERABLE if vulnerable else PLAIN)
        text = f"{random.choice(OPENERS)} {issue} {where} in {random.choice(PLACES)} {random.choice(DURATIONS)}. {random.choice(FILLERS)}"
        text = " ".join(text.split())
        level = min(2, base + (1 if vulnerable else 0))
        if random.random() < noise:  # label noise: reviewers disagree sometimes
            level = max(0, min(2, level + random.choice([-1, 1])))
        rows.append((text, dept, LEVELS[level]))
    path = ROOT / "data" / "complaints.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "department", "priority"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
