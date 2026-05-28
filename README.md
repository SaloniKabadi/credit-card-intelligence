# 💳 Credit Card Customer Intelligence Platform

> End-to-end banking analytics project: ETL pipeline, SQL analysis,
> churn prediction ML model, and Power BI dashboard.

**Tech Stack:** Python | SQL | Scikit-learn | XGBoost | Power BI | DuckDB

**Status:** 🚧 In Progress

## Project Structure

- `data/` — raw and processed datasets
- `notebooks/` — EDA and ML notebooks
- `src/` — ETL pipeline and model scripts
- `sql/` — analytical queries
- `dashboard/` — Power BI report

## How to Run

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place `BankChurners.csv` in `data/raw/`
4. Run the pipeline: `python src/pipeline.py`
5. Train the model: `python src/model.py`

## Dataset

Source: [Kaggle — Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

10,000+ bank customers with demographic and transaction features.
Target variable: `Attrition_Flag` (Existing vs Attrited Customer)
