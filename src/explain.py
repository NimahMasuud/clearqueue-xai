"""Prediction and explanation helpers used by train.py and the dashboard."""
import re
from pathlib import Path

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

ROOT = Path(__file__).resolve().parents[1]
PRIO_REVIEW_BELOW = 0.75   # below this priority confidence, a person should look
DEPT_REVIEW_BELOW = 0.60


def load(path=ROOT / "models" / "clearqueue.joblib"):
    return joblib.load(path)


def predict(b, text):
    X = b["vec"].transform([text])
    pp = b["prio"].predict_proba(X)[0]
    dp = b["dept"].predict_proba(X)[0]
    pc, dc = b["prio"].classes_, b["dept"].classes_
    order = np.argsort(pp)[::-1]
    return {
        "priority": pc[order[0]], "runner_up": pc[order[1]], "prio_conf": float(pp[order[0]]),
        "probs": {c: round(float(p), 3) for c, p in zip(pc, pp)},
        "department": dc[dp.argmax()], "dept_conf": float(dp.max()),
        "review": bool(pp.max() < PRIO_REVIEW_BELOW or dp.max() < DEPT_REVIEW_BELOW),
    }


def contrib(b, key, text, cls):
    """Each feature's share of the model score for one class: tf-idf value times coefficient."""
    m = b[key]
    X = b["vec"].transform([text]).tocoo()
    names = b["vec"].get_feature_names_out()
    w = m.coef_[list(m.classes_).index(cls)]
    return {names[j]: float(v * w[j]) for j, v in zip(X.col, X.data)}


def top(d, n=3):
    return [k for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n] if v > 0]


def contrast(b, text, out):
    """Features that favour the predicted priority over the runner-up."""
    a = contrib(b, "prio", text, out["priority"])
    r = contrib(b, "prio", text, out["runner_up"])
    return {k: a.get(k, 0) - r.get(k, 0) for k in set(a) | set(r)}


def _quote(words):
    return ", ".join(f'"{w}"' for w in words)


def reason(b, text, out):
    pw = top(contrast(b, text, out))
    dw = top(contrib(b, "dept", text, out["department"]))
    s = f'Marked {out["priority"]} rather than {out["runner_up"]}'
    s += f", mainly because of {_quote(pw)}" if pw else " (no single phrase stands out)"
    s += f'. Routed to {out["department"]}'
    s += f" because of {_quote(dw)}." if dw else "."
    return s, pw, dw


def lime_explain(b, text, cls, n=6, samples=500):
    ex = LimeTextExplainer(class_names=list(b["prio"].classes_), random_state=1)
    fn = lambda ts: b["prio"].predict_proba(b["vec"].transform(ts))
    idx = list(b["prio"].classes_).index(cls)
    e = ex.explain_instance(text, fn, num_features=n, labels=[idx], num_samples=samples)
    return [(str(w), float(x)) for w, x in e.as_list(label=idx)]


def counterfactual(b, text, max_remove=8):
    """Remove the most influential words one at a time until the priority changes."""
    out = predict(b, text)
    words = []
    for f in top(contrast(b, text, out), 10):
        words += [w for w in f.split() if w not in words]
    cur, removed = text, []
    for w in words[:max_remove]:
        cur = re.sub(rf"\b{re.escape(w)}\b", "", cur, flags=re.I)
        cur = re.sub(r"\s+", " ", cur).strip()
        removed.append(w)
        new = predict(b, cur)
        if new["priority"] != out["priority"]:
            return {"removed": removed, "text": cur, "new_priority": new["priority"]}
    return None
