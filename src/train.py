"""Trains the models, evaluates them, and writes metrics and a results table."""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import explain
import make_data

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "data" / "complaints.csv").exists():
    make_data.main()
df = pd.read_csv(ROOT / "data" / "complaints.csv")
tr, te = train_test_split(df, test_size=0.2, stratify=df.priority, random_state=42)

# keep negations like "no" and "not": they change what a complaint means
STOP = list(ENGLISH_STOP_WORDS - {"no", "not", "nor", "none", "never"})
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words=STOP)
Xtr, Xte = vec.fit_transform(tr.text), vec.transform(te.text)

candidates = {
    "Logistic regression": LogisticRegression(max_iter=1000, C=3, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
    "Random forest": RandomForestClassifier(300, random_state=0, class_weight="balanced"),
}
models = {}
for name, m in candidates.items():
    m.fit(Xtr, tr.priority)
    rep = classification_report(te.priority, m.predict(Xte), output_dict=True, zero_division=0)
    models[name] = {"macro_f1": rep["macro avg"]["f1-score"], "high_precision": rep["High"]["precision"],
                    "high_recall": rep["High"]["recall"], "high_f1": rep["High"]["f1-score"]}

prio = candidates["Logistic regression"]          # the readable model is the one we serve
dept = LogisticRegression(max_iter=1000, C=3).fit(Xtr, tr.department)
bundle = {"vec": vec, "prio": prio, "dept": dept}
joblib.dump(bundle, ROOT / "models" / "clearqueue.joblib")

pred, dpred = prio.predict(Xte), dept.predict(Xte)
is_high, pred_high = (te.priority == "High").values, pred == "High"
errors = {
    "missed_high_rate": float((is_high & ~pred_high).sum() / is_high.sum()),
    "false_high_rate": float((~is_high & pred_high).sum() / max(1, pred_high.sum())),
}

def tweak(t):  # same complaint, different politeness and filler
    t = re.sub(r"\s*(Please act quickly\.|This is urgent\.|Thank you\.|Kindly assist\.)", "", t)
    return t.replace("Please help, ", "").replace("Hello, ", "")
stability = float(np.mean(prio.predict(vec.transform([tweak(t) for t in te.text])) == pred))

wc = te.text.str.split().str.len()
length_group = np.where(wc > wc.median(), "longer", "shorter")
by_length = {g: float(f1_score(te.priority[length_group == g], pred[length_group == g], average="macro"))
             for g in ["shorter", "longer"]}
high_recall_by_dept = {}
for d in sorted(te.department.unique()):
    mask = ((te.department == d) & (te.priority == "High")).values
    if mask.sum():
        high_recall_by_dept[d] = float((pred[mask] == "High").mean())

overlaps = []                                     # coefficient view vs LIME, top-5 words
for t in te.text.sample(40, random_state=3):
    out = explain.predict(bundle, t)
    a = {w for f in explain.top(explain.contrib(bundle, "prio", t, out["priority"]), 5) for w in f.split()}
    l = {w.lower() for w, wt in explain.lime_explain(bundle, t, out["priority"], n=5, samples=300) if wt > 0}
    if a | l:
        overlaps.append(len(a & l) / len(a | l))

flagged = float(np.mean([explain.predict(bundle, t)["review"] for t in te.text]))
metrics = {
    "priority_models": models, "priority_errors": errors,
    "department_accuracy": float((dpred == te.department).mean()),
    "department_macro_f1": float(f1_score(te.department, dpred, average="macro")),
    "stability_under_rewording": stability, "macro_f1_by_length": by_length,
    "high_recall_by_department": high_recall_by_dept,
    "explanation_overlap_coef_vs_lime": float(np.mean(overlaps)),
    "share_sent_to_human_review": flagged, "test_size": int(len(te)),
}
text = json.dumps(metrics, indent=2)
(ROOT / "models" / "metrics.json").write_text(text)
(ROOT / "documentation" / "results.md").write_text("## Results on the held-out test set\n\n```json\n" + text + "\n```\n")
print(text)
