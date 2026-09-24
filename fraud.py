"""
Step 4: Fraud detection. Builds behavioral features for every contributor and
flags the suspicious ones WITHOUT using the hidden 'kind' column.
The hidden column is only used at the end to grade how well we did.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

labels = pd.read_csv("labels.csv")
tasks = pd.read_csv("tasks_ds.csv")          # from dawid_skene.py
contributors = pd.read_csv("contributors_ds.csv")

df = labels.merge(tasks[["task_id", "ds_label"]], on="task_id")
df["agrees_with_consensus"] = df.label == df.ds_label

# ---- 1. Behavioral features per contributor ----
def label_entropy(s):
    p = s.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

feats = df.groupby("contributor_id").agg(
    n_tasks=("task_id", "count"),
    median_seconds=("seconds", "median"),
    std_seconds=("seconds", "std"),
    consensus_agreement=("agrees_with_consensus", "mean"),
    label_entropy=("label", label_entropy),          # lazy workers -> ~0
)
feats = feats.join(contributors.set_index("contributor_id")[["est_accuracy", "kind"]])

FEATURES = ["median_seconds", "std_seconds", "consensus_agreement",
            "label_entropy", "est_accuracy"]

# ---- 2. Rule-based flags (simple, explainable) ----
# "Too fast" is relative to how long the typical contributor takes
platform_median = feats.median_seconds.median()
feats["flag_too_fast"] = feats.median_seconds < 0.5 * platform_median
feats["flag_one_answer"] = feats.label_entropy < 0.3
# Low quality is NOT fraud: some honest people are just less skilled.
# We track it separately (e.g. send them to training), not ban them.
feats["low_quality"] = feats.est_accuracy < 0.6

# ---- 3. Anomaly detection (catches things rules miss) ----
X = StandardScaler().fit_transform(feats[FEATURES])
iso = IsolationForest(contamination=0.1, random_state=0).fit(X)
feats["anomaly_score"] = -iso.score_samples(X)     # higher = weirder
feats["flag_anomaly"] = iso.predict(X) == -1

feats["flagged"] = feats[["flag_too_fast", "flag_one_answer",
                          "flag_anomaly"]].any(axis=1)

# ---- 4. Grade against the hidden truth ----
is_bad = feats.kind != "honest"
tp = (feats.flagged & is_bad).sum()
fp = (feats.flagged & ~is_bad).sum()
fn = (~feats.flagged & is_bad).sum()
print(f"Flagged {feats.flagged.sum()} contributors")
print(f"Caught {tp} of {is_bad.sum()} bad actors, {fp} honest people wrongly flagged")
print(f"Precision {tp / (tp + fp):.2f} | Recall {tp / (tp + fn):.2f}")

print("\nCaught by type:")
print(feats[is_bad].groupby("kind")["flagged"].agg(["sum", "count"])
      .rename(columns={"sum": "caught", "count": "total"}))

# ---- 5. Does removing flagged people improve labels? ----
clean = df[~df.contributor_id.isin(feats.index[feats.flagged])]
mv = clean.groupby("task_id")["label"].agg(lambda s: s.mode().iloc[0])
t = tasks.set_index("task_id")
covered = t.index.isin(mv.index)
acc = (mv == t.loc[mv.index, "true_label"]).mean()
print(f"\nMajority vote after removing flagged: {acc:.3f} "
      f"({covered.mean():.0%} of tasks still have labels)")

print(f"\nHonest-but-low-quality contributors (not flagged as fraud): "
      f"{(feats.low_quality & ~feats.flagged & ~is_bad).sum()}")

feats.to_csv("fraud_report.csv")
