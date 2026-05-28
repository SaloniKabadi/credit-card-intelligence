import duckdb
import pandas as pd
import os

# Paths
CLEAN_DATA = "data/processed/clean_customers.csv"
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connect DuckDB and load CSV as a table
conn = duckdb.connect()
conn.execute(f"CREATE TABLE customers AS SELECT * FROM read_csv_auto('{CLEAN_DATA}')")

print("✅ Data loaded into DuckDB\n")

# Define all queries
queries = {
    "01_overall_churn_rate": """
        SELECT
            COUNT(*) AS total_customers,
            SUM(Churn) AS churned_customers,
            ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
        FROM customers
    """,

    "02_churn_by_card_category": """
        SELECT
            Card_Category,
            COUNT(*) AS total,
            SUM(Churn) AS churned,
            ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY Card_Category
        ORDER BY churn_rate_pct DESC
    """,

    "03_churn_by_income": """
        SELECT
            Income_Category,
            COUNT(*) AS total_customers,
            SUM(Churn) AS churned,
            ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
            ROUND(AVG(Credit_Limit), 0) AS avg_credit_limit
        FROM customers
        GROUP BY Income_Category
        ORDER BY churn_rate_pct DESC
    """,

    "04_risk_segments": """
        SELECT
            CASE
                WHEN Utilization_Ratio >= 0.7 THEN 'High Risk (>70%)'
                WHEN Utilization_Ratio >= 0.4 THEN 'Medium Risk (40-70%)'
                ELSE 'Low Risk (<40%)'
            END AS risk_segment,
            COUNT(*) AS total_customers,
            SUM(Churn) AS churned,
            ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
            ROUND(AVG(Total_Trans_Amt), 0) AS avg_transaction_amt
        FROM customers
        GROUP BY risk_segment
        ORDER BY churn_rate_pct DESC
    """,

    "05_spending_segments": """
        SELECT
            CASE
                WHEN Total_Trans_Amt >= 4500 THEN 'Top Spenders'
                WHEN Total_Trans_Amt >= 2000 THEN 'Mid Spenders'
                ELSE 'Low Spenders'
            END AS spending_segment,
            COUNT(*) AS total,
            SUM(Churn) AS churned,
            ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
            ROUND(AVG(Months_Inactive_12_mon), 2) AS avg_inactive_months
        FROM customers
        GROUP BY spending_segment
        ORDER BY churn_rate_pct DESC
    """,

    "06_inactivity_vs_churn": """
        SELECT
            Months_Inactive_12_mon,
            COUNT(*) AS total_customers,
            SUM(Churn) AS churned,
            ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY Months_Inactive_12_mon
        ORDER BY Months_Inactive_12_mon
    """
}

# Run each query, print result, save to CSV
for name, query in queries.items():
    print(f"{'='*50}")
    print(f"📊 {name.replace('_', ' ').upper()}")
    print(f"{'='*50}")
    result = conn.execute(query).df()
    print(result.to_string(index=False))
    print()

    # Save each result as CSV
    result.to_csv(f"{OUTPUT_DIR}/{name}.csv", index=False)

print("✅ All query results saved to /reports folder")
conn.close()
