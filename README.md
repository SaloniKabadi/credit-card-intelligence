# 💳 Credit Card Customer Intelligence Platform

> End-to-end banking analytics: ETL pipeline · SQL segmentation · Churn prediction at **0.9932 AUC** · Interactive dashboard

---

## 🔍 Problem Statement

Predict which credit card customers are likely to churn and identify the key behavioural and financial drivers — enabling targeted retention campaigns before customers leave.

---

## 🏗️ Project Architecture

```
Raw CSV → ETL Pipeline (Python) → Clean Data → SQL Analysis (DuckDB)
       → ML Model (XGBoost 0.9932 AUC) → Interactive HTML Dashboard
```

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Dataset | 10,127 customers · 23 features |
| Overall Churn Rate | **16.07%** |
| XGBoost ROC-AUC | **0.9932** |
| 5-Fold CV Mean AUC | **0.9929** (σ = 0.002) |
| Top Churn Driver | Total Transaction Count |
| Highest Risk Segment | Platinum cardholders — **25% churn** |
| Key Behavioural Signal | 6 bank contacts → **100% churn rate** |

---

## 🔑 Key Insights

- Customers with **0 months of inactivity** show 51.7% churn — they are already disengaging
- **Platinum cardholders** churn at 25%, nearly 3× the Silver card rate (14.77%)
- **Low spenders** (< $2,000 transactions) churn at 21.35% vs 9.32% for top spenders
- Every additional bank contact increases churn risk — 6 contacts = 100% churn
- **Doctorate holders** churn most (21.06%) — likely higher financial mobility
- Low utilization customers churn more than high utilization — disengagement signal

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Engineering | Python, Pandas, NumPy |
| SQL Analysis | DuckDB, SQL |
| Machine Learning | Scikit-learn, XGBoost, SHAP |
| Visualisation | Matplotlib, Seaborn, Chart.js |
| Versioning | Git, GitHub |

---

## 📂 Project Structure

```
credit-card-intelligence/
├── data/
│   ├── raw/                  ← BankChurners.csv (Kaggle)
│   └── processed/            ← clean_customers.csv (engineered features)
├── notebooks/
│   ├── 01_EDA.ipynb          ← 8 exploratory charts
│   └── 02_churn_model.ipynb  ← 3 models, ROC curves, feature importance
├── src/
│   ├── pipeline.py           ← automated ETL with logging
│   ├── run_queries.py        ← DuckDB SQL runner
│   └── utils.py
├── sql/
│   └── analysis.sql          ← 6 business queries
├── dashboard/
│   ├── credit_card_dashboard.html  ← interactive Chart.js dashboard
│   └── build_dashboard.py
├── reports/                  ← exported charts + query CSVs
└── requirements.txt
```

---

## 🚀 How to Run

```bash
git clone https://github.com/SaloniKabadi/credit-card-intelligence.git
cd credit-card-intelligence

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run ETL pipeline
python3 src/pipeline.py

# Run SQL analysis
python3 src/run_queries.py

# Open dashboard
open dashboard/credit_card_dashboard.html

# Launch notebooks
jupyter notebook
```

---

## 📊 Model Comparison

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.9287 |
| Random Forest | 0.9879 |
| **XGBoost** | **0.9932** ✅ |

XGBoost 5-Fold CV: **0.9929 mean AUC** (σ = 0.002)

**Top 3 feature importances:**
1. `Total_Trans_Ct` — 26.9%
2. `Total_Revolving_Bal` — 15.6%
3. `Total_Relationship_Count` — 12.5%

---

## 📬 About

Built by **Saloni Kabadi** — Data Analyst at HDFC Bank.

[linkedin.com/in/saloni-kabadi](https://linkedin.com/in/saloni-kabadi) · [github.com/SaloniKabadi](https://github.com/SaloniKabadi)
