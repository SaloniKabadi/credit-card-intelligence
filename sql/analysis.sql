-- ================================================
-- Credit Card Customer Intelligence — SQL Analysis
-- ================================================

-- QUERY 1: Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(Churn) AS churned_customers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers;


-- QUERY 2: Churn rate by card category
SELECT
    Card_Category,
    COUNT(*) AS total,
    SUM(Churn) AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Card_Category
ORDER BY churn_rate_pct DESC;


-- QUERY 3: Churn rate by income bracket
SELECT
    Income_Category,
    COUNT(*) AS total_customers,
    SUM(Churn) AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(Credit_Limit), 0) AS avg_credit_limit
FROM customers
GROUP BY Income_Category
ORDER BY churn_rate_pct DESC;


-- QUERY 4: High risk vs low risk segments by utilization
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
ORDER BY churn_rate_pct DESC;


-- QUERY 5: Top spending vs bottom spending customers
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
ORDER BY churn_rate_pct DESC;


-- QUERY 6: Month-on-month inactivity vs churn
SELECT
    Months_Inactive_12_mon,
    COUNT(*) AS total_customers,
    SUM(Churn) AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Months_Inactive_12_mon
ORDER BY Months_Inactive_12_mon;
