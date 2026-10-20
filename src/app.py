"""ClearQueue dashboard. Run from the repo root: streamlit run src/app.py"""
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import explain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "Gas smell near a school": "There is a strong smell of gas next to the primary school and children are coughing.",
    "Wrong water bill": "Hello, the water bill looks wrong this month in Ikeja. Kindly assist.",
    "Voter card not received": "I have not received my voter card since last night. Thank you.",
    "Registration centre closed": "The registration centre has no staff and many people are waiting on our street.",
    "No electricity": "We have had no electricity for two weeks on our street.",
}

st.set_page_config(page_title="ClearQueue", layout="wide")


@st.cache_resource
def bundle():
    return explain.load()


def highlight(text, words):
    safe = html.escape(text)
    if not words:
        return safe
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")\b", re.I)
    return pat.sub(r"<mark>\1</mark>", safe)


b = bundle()
st.title("ClearQueue")
st.caption("Explainable complaint triage for public services. Demo software using synthetic data. "
           "It supports staff decisions and does not replace them.")
tab1, tab2, tab3 = st.tabs(["Triage a complaint", "Global view", "Data, limits and ethics"])

with tab1:
    pick = st.selectbox("Try an example or write your own", ["(write my own)"] + list(EXAMPLES))
    text = st.text_area("Complaint text", EXAMPLES.get(pick, ""), height=110,
                        help="Do not enter real names, phone numbers or addresses.")
    if st.button("Triage", type="primary") and text.strip():
        out = explain.predict(b, text)
        sentence, pw, dw = explain.reason(b, text, out)
        c1, c2, c3 = st.columns(3)
        c1.metric("Priority", out["priority"], f'{out["prio_conf"]:.0%} confidence')
        c2.metric("Department / queue", out["department"], f'{out["dept_conf"]:.0%} confidence')
        c3.metric("Human review", "Needed" if out["review"] else "Not flagged")
        if out["review"]:
            st.warning("The model is unsure. A person should read this complaint before it is routed.")
        st.markdown("**Why**  \n" + sentence)
        st.markdown(highlight(text, set(w for f in pw + dw for w in f.split())), unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**LIME: words pushing toward " + out["priority"] + "**")
            lime = explain.lime_explain(b, text, out["priority"])
            st.bar_chart(pd.Series({w: wt for w, wt in lime}, name="weight"))
        with right:
            st.markdown("**Smallest wording change that flips the priority**")
            cf = explain.counterfactual(b, text)
            if cf:
                st.write(f'Removing {", ".join(repr(w) for w in cf["removed"])} changes the result '
                         f'from {out["priority"]} to {cf["new_priority"]}.')
                st.caption(cf["text"])
            else:
                st.write("No small wording change flips this result.")
            st.markdown("**Class probabilities**")
            st.bar_chart(pd.Series(out["probs"], name="probability"))

with tab2:
    names = b["vec"].get_feature_names_out()
    which = st.radio("Show strongest terms for", ["Priority", "Department"], horizontal=True)
    model = b["prio"] if which == "Priority" else b["dept"]
    cls = st.selectbox("Class", list(model.classes_))
    w = model.coef_[list(model.classes_).index(cls)]
    idx = w.argsort()[::-1][:12]
    st.bar_chart(pd.Series(w[idx], index=names[idx], name="coefficient"))
    mp = ROOT / "models" / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        st.markdown("**Priority models compared (held-out test set)**")
        st.dataframe(pd.DataFrame(m["priority_models"]).T.round(3))
        st.json({k: v for k, v in m.items() if k != "priority_models"})

with tab3:
    st.markdown("""
**Data.** Synthetic complaints written in a 311 style for Nigerian public services, including an election-period
services desk (voter card, voter list, registration centre). Labels follow a written severity rubric, so
they are a proxy, not ground truth. Scores on synthetic data look higher than real-world performance would.

**Limits.** Explanations describe how the model behaves, not what causes a problem. Wording outside the training
phrases lowers accuracy. Short or vague complaints are sent to human review.

**Ethics.** No personal data. The tool never decides on its own: an accountable officer reviews every
priority and routing call, and low-confidence cases are flagged. The election-period category is limited to
routine service requests, with no political content.
""")
