# 💳 Credit Card Customer Intelligence Platform

> End-to-end banking analytics: ETL pipeline · SQL segmentation · Churn prediction at **0.9932 AUC** · Interactive editorial dashboard

🔗 **Live dashboard:** **[salonikabadi.github.io/credit-card-intelligence](https://salonikabadi.github.io/credit-card-intelligence/)**

---

## 🖥️ The Dashboard

A four-page editorial-style dashboard (think data magazine, not corporate BI) covering:

| Page | What's inside |
|---|---|
| **Front Page** | 8 KPI tiles · churn donut · card-tier, income & spend-gap charts |
| **Segmentation** | Age band · gender · education · tenure · spend × utilization · products held |
| **Risk Signals** | Risk segments · spend tiers · inactivity arc · service contacts · utilization shape |
| **Model Lab** | Model bake-off · 5-fold CV stability · top-10 feature importance |

Built with Plotly + handwritten HTML/CSS. Includes count-up animations, scroll-triggered reveals, an animated stat ticker, a cursor spotlight and a live IST clock. Deployed via **GitHub Pages** straight from `main` &mdash; every push rebuilds the site automatically.

```
Run locally:
  python3 dashboard/build_dashboard.py
  open dashboard/credit_card_dashboard.html
```

---

## 🔍 Problem Statement

Predict which credit card customers are likely to churn and identify the key behavioural and financial drivers &mdash; enabling targeted retention campaigns before customers leave.

---

## 🏗️ Project Architecture

```
Raw CSV → ETL Pipeline (Python) → Clean Data → SQL Analysis (DuckDB)
       → ML Model (XGBoost 0.9932 AUC) → Editorial HTML Dashboard (Plotly)
       → GitHub Pages
```

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Dataset | 10,127 customers · 23 features |
| Overall Churn Rate | **16.07%** |
| XGBoost ROC-AUC | **0.9932** |
| 5-Fold CV Mean AUC | **0.9929** (σ = 0.002) |
| Top Churn Driver | Total Transaction Count (26.9%) |
| Highest Risk Segment | Platinum cardholders — **25% churn** |
| Key Behavioural Signal | 6 bank contacts → **100% churn rate** |

---

## 🔑 Key Insights

- Customers with **0 months of inactivity** show 51.7% churn &mdash; already disengaging
- **Platinum cardholders** churn at 25%, nearly 3× the Silver card rate (14.77%)
- **Low spenders** (< $2,000 transactions) churn at 21.35% vs 9.32% for top spenders
- Every additional bank contact increases churn risk &mdash; 6 contacts = 100% churn
- **Doctorate holders** churn most (21.06%) &mdash; likely higher financial mobility
- Low utilization customers churn more than high utilization &mdash; disengagement signal

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Engineering | Python, Pandas, NumPy |
| SQL Analysis | DuckDB, SQL |
| Machine Learning | Scikit-learn, XGBoost, SHAP |
| Visualisation | Matplotlib, Seaborn, **Plotly** |
| Dashboard | Plotly + handwritten HTML/CSS/JS, served via GitHub Pages |
| Versioning | Git, GitHub |

---

## 📂 Project Structure

```
credit-card-intelligence/
├── data/
│   ├── raw/                          ← BankChurners.csv (Kaggle)
│   └── processed/                    ← clean_customers.csv (engineered features)
├── notebooks/
│   ├── 01_EDA.ipynb                  ← 8 exploratory charts
│   └── 02_churn_model.ipynb          ← 3 models, ROC curves, feature importance
├── src/
│   ├── pipeline.py                   ← automated ETL with logging
│   ├── run_queries.py                ← DuckDB SQL runner
│   └── utils.py
├── sql/
│   └── analysis.sql                  ← 6 business queries
├── dashboard/
│   ├── build_dashboard.py            ← regenerates the dashboard from data
│   └── credit_card_dashboard.html    ← rendered editorial dashboard
├── reports/                          ← exported charts + query CSVs
├── index.html                        ← root redirect to the live dashboard
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

# Rebuild + open the dashboard
python3 dashboard/build_dashboard.py
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

## 🌐 Deployment

The dashboard is deployed automatically via **GitHub Pages** from the `main` branch root. Each push to `main` triggers a rebuild within ~1 minute.

- Production URL: <https://salonikabadi.github.io/credit-card-intelligence/>
- Source: `dashboard/credit_card_dashboard.html`
- Root redirect: `index.html` sends visitors straight to the dashboard

To regenerate after a data refresh:

```bash
python3 dashboard/build_dashboard.py
git add dashboard/credit_card_dashboard.html
git commit -m "Refresh dashboard"
git push
```

---

## 📬 About

Built by **Saloni Kabadi** — Data Analyst at HDFC Bank.

[linkedin.com/in/saloni-kabadi-69325036a](https://www.linkedin.com/in/saloni-kabadi-69325036a) · [github.com/SaloniKabadi](https://github.com/SaloniKabadi)
