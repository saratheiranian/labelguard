# 🐍 Python & AI Training

A hands-on learning repository covering Python data science fundamentals, exploratory data analysis, and AI/ML concepts.

## 📁 Repository Structure

```
python-ai-training/
├── titanic-eda/
│   ├── titanic_eda.py          # Full EDA script
│   ├── titanic.csv             # Dataset
│   └── titanic_eda_dashboard.png  # Generated visualisation
└── README.md
```

## 🔍 Project 1: Titanic EDA

**Goal:** Perform detailed exploratory data analysis on the Titanic dataset to uncover survival patterns.

### Key Findings

| Factor | Insight |
|--------|---------|
| Overall survival | **38.4%** of passengers survived |
| Gender | Women: **74%** survival · Men: **19%** survival |
| Passenger class | 1st: **63%** · 2nd: **47%** · 3rd: **24%** |
| Age | Children (≤12) had the **highest** survival rate (58%) |
| Family size | Small families (2–4) survived better than solo or large groups |
| Embarkation | Cherbourg passengers had highest survival (many 1st-class) |

### How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/python-ai-training.git
cd python-ai-training

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Run EDA
cd titanic-eda
python titanic_eda.py
```

### Dashboard Preview

Running the script produces an 11-panel visual dashboard covering:
- Survival counts and rates
- Survival by sex, class, age group
- Fare and embarkation distributions
- Family size trends
- Correlation matrix

## 🛠 Tech Stack

- Python 3.x
- pandas · numpy · matplotlib · seaborn

## 📚 Dataset

[Titanic dataset](https://www.kaggle.com/c/titanic) via seaborn's built-in datasets (mirrors the Kaggle Titanic competition data).

---
*Part of a structured Python & AI learning curriculum.*
