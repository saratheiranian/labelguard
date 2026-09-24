"""
Step 1: Simulate a crowdsourced labeling platform.
Creates tasks with hidden true labels, a pool of contributors with different
behaviors, and a stream of labels. Saves everything to CSV.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_TASKS = 2000
N_CLASSES = 2          # binary for now (e.g. positive/negative sentiment)
LABELS_PER_TASK = 5

# ---- 1. Tasks with hidden ground truth ----
tasks = pd.DataFrame({
    "task_id": range(N_TASKS),
    "true_label": rng.integers(0, N_CLASSES, N_TASKS),
    "difficulty": rng.beta(2, 5, N_TASKS),   # most tasks easy, some hard
})
# 5% of tasks are "gold": we know the answer and can check contributors on them
tasks["is_gold"] = rng.random(N_TASKS) < 0.05

# ---- 2. Contributors with different behaviors ----
contributors = []
cid = 0
def add(kind, n, **kw):
    global cid
    for _ in range(n):
        contributors.append({"contributor_id": cid, "kind": kind, **kw})
        cid += 1

for _ in range(160):  # honest, varying skill
    contributors.append({"contributor_id": cid, "kind": "honest",
                         "skill": rng.uniform(0.70, 0.97)}); cid += 1
add("lazy", 15, skill=None)      # always picks label 0
add("bot", 15, skill=None)       # random answers, very fast
add("colluder", 10, skill=None)  # copy a shared answer key
contributors = pd.DataFrame(contributors)
bad_ids = set(contributors.loc[contributors.kind != "honest", "contributor_id"])

# colluders share one (mostly wrong-ish) answer key
collusion_key = rng.integers(0, N_CLASSES, N_TASKS)

# ---- 3. Generate labels ----
rows = []
for t in tasks.itertuples():
    workers = rng.choice(contributors.contributor_id, LABELS_PER_TASK, replace=False)
    for w in workers:
        c = contributors.loc[w]
        if c.kind == "honest":
            p_correct = c.skill - 0.3 * t.difficulty
            if rng.random() < p_correct:
                label = t.true_label
            else:
                label = rng.choice([l for l in range(N_CLASSES) if l != t.true_label])
            seconds = rng.normal(20 + 30 * t.difficulty, 5)
        elif c.kind == "lazy":
            label, seconds = 0, rng.normal(6, 2)
        elif c.kind == "bot":
            label, seconds = rng.integers(0, N_CLASSES), rng.normal(1.5, 0.5)
        else:  # colluder
            label, seconds = collusion_key[t.task_id], rng.normal(12, 3)
        rows.append({"task_id": t.task_id, "contributor_id": int(w),
                     "label": int(label), "seconds": max(0.3, round(seconds, 2))})

labels = pd.DataFrame(rows)

tasks.to_csv("tasks.csv", index=False)
contributors.to_csv("contributors.csv", index=False)
labels.to_csv("labels.csv", index=False)
print(f"{len(tasks)} tasks, {len(contributors)} contributors "
      f"({len(bad_ids)} bad actors), {len(labels)} labels saved.")
