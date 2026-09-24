"""
Step 6: Dashboard. Run with:  streamlit run dashboard.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LabelQA", page_icon="🏷️", layout="wide")

@st.cache_data
def load():
    tasks = pd.read_csv("tasks_ds.csv")
    people = pd.read_csv("fraud_report.csv")
    routing = pd.read_csv("routing_results.csv")
    return tasks, people, routing

try:
    tasks, people, routing = load()
except FileNotFoundError as e:
    st.error(f"Missing {e.filename}. Run simulate.py, dawid_skene.py, "
             "fraud.py and routing.py first.")
    st.stop()

st.title("🏷️ LabelQA")
st.caption("Quality estimation, fraud detection and smart routing "
           "for a simulated crowdsourced labeling platform")

# ---- Headline numbers ----
mv_acc = (tasks.mv_label == tasks.true_label).mean()
ds_acc = (tasks.ds_label == tasks.true_label).mean()
is_bad = people.kind != "honest"
caught = (people.flagged & is_bad).sum()
false_flags = (people.flagged & ~is_bad).sum()
base = routing.iloc[0]
best = routing.loc[routing.strategy.str.contains("95%")].iloc[0]
saved = 1 - best.labels_per_task / base.labels_per_task

c1, c2, c3, c4 = st.columns(4)
c1.metric("Label accuracy (Dawid-Skene)", f"{ds_acc:.1%}",
          f"{ds_acc - mv_acc:+.1%} vs majority vote")
c2.metric("Bad actors caught", f"{caught} / {is_bad.sum()}",
          f"{false_flags} false flags", delta_color="off")
c3.metric("Routed accuracy", f"{best.accuracy:.1%}",
          f"{best.accuracy - base.accuracy:+.1%} vs random")
c4.metric("Labeling cost saved", f"{saved:.0%}",
          f"{best.labels_per_task:.1f} vs {base.labels_per_task:.0f} labels/task",
          delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(
    ["👥 Contributors", "🚩 Flagged accounts", "💰 Cost vs accuracy", "🔍 Task explorer"])

# ---- Contributors ----
with tab1:
    st.subheader("Contributor quality")
    st.write("Each dot is a contributor. Fraudsters cluster at fast speed "
             "and coin-flip accuracy.")
    people["status"] = "OK"
    people.loc[people.low_quality, "status"] = "Low quality"
    people.loc[people.flagged, "status"] = "Flagged"
    st.scatter_chart(people, x="median_seconds", y="est_accuracy",
                     color="status", height=420)

    show_truth = st.toggle("Reveal hidden contributor type (for grading only)")
    cols = ["contributor_id", "status", "est_accuracy", "consensus_agreement",
            "median_seconds", "n_tasks"] + (["kind"] if show_truth else [])
    st.dataframe(people.sort_values("est_accuracy", ascending=False)[cols],
                 hide_index=True)

# ---- Flagged accounts ----
with tab2:
    st.subheader("Flagged accounts and why")
    flagged = people[people.flagged].copy()

    def reasons(r):
        out = []
        if r.flag_too_fast: out.append(f"too fast ({r.median_seconds:.1f}s median)")
        if r.flag_one_answer: out.append("gives the same answer every time")
        if r.flag_anomaly: out.append(f"anomalous behavior (score {r.anomaly_score:.2f})")
        return "; ".join(out)

    flagged["reasons"] = flagged.apply(reasons, axis=1)
    cols = ["contributor_id", "reasons", "est_accuracy", "median_seconds"]
    if st.toggle("Reveal hidden type", key="truth2"):
        cols.append("kind")
    st.dataframe(flagged.sort_values("anomaly_score", ascending=False)[cols],
                 hide_index=True)
    st.info(f"{people.low_quality.sum()} honest-looking contributors have low quality "
            "but are not flagged as fraud. On a real platform they'd get training, "
            "not a ban.")

# ---- Cost vs accuracy ----
with tab3:
    st.subheader("Routing strategy: accuracy vs cost")
    st.scatter_chart(routing, x="labels_per_task", y="accuracy",
                     color="strategy", size=200, height=420)
    st.dataframe(routing.style.format({"accuracy": "{:.1%}",
                                       "labels_per_task": "{:.2f}"}),
                 hide_index=True)
    st.write("Up and to the left is better: higher accuracy for fewer paid labels.")

# ---- Task explorer ----
with tab4:
    st.subheader("Least confident tasks")
    st.write("Tasks where Dawid-Skene is least sure. These are the ones a real "
             "platform would send for expert review.")
    min_conf = st.slider("Show tasks with confidence below", 0.5, 1.0, 0.8, 0.01)
    low = tasks[tasks.ds_confidence < min_conf].sort_values("ds_confidence")
    st.write(f"{len(low)} tasks "
             f"({(low.ds_label == low.true_label).mean():.0%} correct)" if len(low) else "No tasks")
    st.dataframe(low[["task_id", "ds_label", "ds_confidence", "mv_label",
                      "true_label", "difficulty"]],
                 hide_index=True)
