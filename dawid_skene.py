"""
Step 3: Dawid-Skene. Learns how reliable each contributor is and uses that
to weight their votes, instead of treating every vote equally.
"""
import numpy as np
import pandas as pd

tasks = pd.read_csv("tasks.csv")
labels = pd.read_csv("labels.csv")
contributors = pd.read_csv("contributors.csv")

n_tasks = tasks.task_id.max() + 1
n_workers = contributors.contributor_id.max() + 1
K = labels.label.max() + 1  # number of classes

t = labels.task_id.values
w = labels.contributor_id.values
l = labels.label.values

# ---- Initialize: soft majority vote ----
# T[i, k] = probability that task i's true label is k
T = np.zeros((n_tasks, K))
np.add.at(T, (t, l), 1)
T /= T.sum(axis=1, keepdims=True)

for it in range(50):
    # ---- M-step: estimate each contributor's confusion matrix ----
    # conf[j, true, given] = P(contributor j answers `given` | truth is `true`)
    conf = np.full((n_workers, K, K), 0.01)  # small smoothing
    for k in range(K):
        np.add.at(conf[:, k, :], (w, l), T[t, k])
    conf /= conf.sum(axis=2, keepdims=True)
    prior = T.mean(axis=0)

    # ---- E-step: re-estimate each task's true label ----
    logT = np.tile(np.log(prior), (n_tasks, 1))
    for k in range(K):
        np.add.at(logT[:, k], t, np.log(conf[w, k, l]))
    newT = np.exp(logT - logT.max(axis=1, keepdims=True))
    newT /= newT.sum(axis=1, keepdims=True)

    change = np.abs(newT - T).max()
    T = newT
    if change < 1e-6:
        print(f"Converged after {it + 1} iterations")
        break

tasks["ds_label"] = T.argmax(axis=1)
tasks["ds_confidence"] = T.max(axis=1)

# ---- Compare with majority vote ----
mv = labels.groupby("task_id")["label"].agg(lambda s: s.mode().iloc[0])
tasks["mv_label"] = tasks.task_id.map(mv)
print(f"Majority vote accuracy: {(tasks.mv_label == tasks.true_label).mean():.3f}")
print(f"Dawid-Skene accuracy:   {(tasks.ds_label == tasks.true_label).mean():.3f}")

# ---- Estimated contributor quality ----
# Estimated accuracy = weighted diagonal of the confusion matrix
contributors["est_accuracy"] = (conf[:, range(K), range(K)] * prior).sum(axis=1)
print("\nAverage ESTIMATED accuracy by (hidden) contributor type:")
print(contributors.groupby("kind")["est_accuracy"].mean().round(3))

print("\n10 lowest-rated contributors:")
print(contributors.nsmallest(10, "est_accuracy")[["contributor_id", "kind", "est_accuracy"]]
      .to_string(index=False))

tasks.to_csv("tasks_ds.csv", index=False)
contributors.to_csv("contributors_ds.csv", index=False)
