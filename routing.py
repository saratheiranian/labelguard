"""
Step 5: Smart routing + adaptive stopping.
Replays labeling on a NEW batch of tasks and compares strategies on
accuracy vs. cost (number of labels paid for).

Uses what we learned in earlier steps:
  - est_accuracy from dawid_skene.py (how good each person is)
  - flagged from fraud.py (who to keep off the platform)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

people = pd.read_csv("fraud_report.csv").set_index("contributor_id")
skill = pd.read_csv("contributors.csv").set_index("contributor_id")["skill"]
people["skill"] = skill  # hidden, only used to SIMULATE answers

# ---- A new batch of tasks (the routing policy has never seen these) ----
N = 2000
true_label = rng.integers(0, 2, N)
difficulty = rng.beta(2, 5, N)
collusion_key = rng.integers(0, 2, N)
CAPACITY = 100 # max tasks one person can do in this batch

def answer(pid, i):
    """Simulate contributor pid answering task i (same behavior as simulate.py)."""
    kind = people.at[pid, "kind"]
    if kind == "honest":
        p = people.at[pid, "skill"] - 0.3 * difficulty[i]
        return true_label[i] if rng.random() < p else 1 - true_label[i]
    if kind == "lazy":
        return 0
    if kind == "bot":
        return rng.integers(0, 2)
    return collusion_key[i]  # colluder

def log_odds(pid):
    """How much one vote from this person should move our belief."""
    a = np.clip(people.at[pid, "est_accuracy"], 0.55, 0.99)
    return np.log(a / (1 - a))

def run(strategy, threshold=0.95, fixed_k=5, max_labels=7):
    used = pd.Series(0, index=people.index)
    if strategy == "random":
        pool = people.index.values
    else:  # exclude flagged fraudsters
        pool = people.index[~people.flagged].values

    correct, total_labels = 0, 0
    for i in rng.permutation(N):
        available = [p for p in pool if used[p] < CAPACITY]
        if strategy == "smart":
            # best available people first
            ranked = sorted(available, key=lambda p: -people.at[p, "est_accuracy"])
        else:
            ranked = list(rng.permutation(available))

        score, n = 0.0, 0   # score > 0 means "leaning label 1"
        for pid in ranked:
            vote = answer(pid, i)
            used[pid] += 1
            n += 1
            weight = log_odds(pid) if strategy == "smart" else 1.0
            score += weight if vote == 1 else -weight

            if strategy == "smart":
                confidence = 1 / (1 + np.exp(-abs(score)))
                if (n >= 2 and confidence >= threshold) or n >= max_labels:
                    break
            elif n >= fixed_k:
                break

        guess = 1 if score > 0 else 0 if score < 0 else rng.integers(0, 2)
        correct += guess == true_label[i]
        total_labels += n
    return correct / N, total_labels / N

results = []
print(f"{'Strategy':<42}{'Accuracy':>9}{'Labels/task':>13}")
for name, strat, kw in [
    ("Random, 5 labels (what we had)", "random", {}),
    ("Remove fraudsters, random, 5 labels", "filtered", {}),
    ("Smart routing, stop at 90% confidence", "smart", {"threshold": 0.90}),
    ("Smart routing, stop at 95% confidence", "smart", {"threshold": 0.95}),
    ("Smart routing, stop at 99% confidence", "smart", {"threshold": 0.99}),
]:
    acc, cost = run(strat, **kw)
    print(f"{name:<42}{acc:>9.3f}{cost:>13.2f}")
    results.append({"strategy": name, "accuracy": acc, "labels_per_task": cost})

pd.DataFrame(results).to_csv("routing_results.csv", index=False)
