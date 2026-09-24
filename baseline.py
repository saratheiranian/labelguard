"""
Step 2: Majority-vote baseline. This is the number every later method must beat.
"""
import pandas as pd

tasks = pd.read_csv("tasks.csv")
labels = pd.read_csv("labels.csv")
contributors = pd.read_csv("contributors.csv")

# Majority vote per task
mv = labels.groupby("task_id")["label"].agg(lambda s: s.mode().iloc[0])
tasks["mv_label"] = tasks.task_id.map(mv)
acc = (tasks.mv_label == tasks.true_label).mean()
print(f"Majority-vote accuracy: {acc:.3f}")

# How accurate is each contributor type really? (uses hidden truth, for analysis only)
merged = labels.merge(tasks[["task_id", "true_label"]], on="task_id") \
               .merge(contributors[["contributor_id", "kind"]], on="contributor_id")
merged["correct"] = merged.label == merged.true_label
print("\nTrue accuracy by contributor type:")
print(merged.groupby("kind")["correct"].mean().round(3))
print("\nMedian seconds per task by type:")
print(merged.groupby("kind")["seconds"].median().round(1))
