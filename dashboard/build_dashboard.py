import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Load Data ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "../data/processed/clean_customers.csv"))

total_customers    = len(df)
churned_customers  = int(df["Churn"].sum())
retained_customers = total_customers - churned_customers
churn_rate         = round(df["Churn"].mean() * 100, 2)
avg_credit_limit   = round(df["Credit_Limit"].mean(), 0)
avg_utilization    = round(df["Utilization_Ratio"].mean() * 100, 2)

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#0F172A"
CARD_BG  = "#1E293B"
BORDER   = "#334155"
CYAN     = "#22D3EE"
GREEN    = "#10B981"
RED      = "#F43F5E"
AMBER    = "#FBBF24"
PURPLE   = "#A78BFA"
BLUE     = "#60A5FA"
TEXT     = "#F1F5F9"
SUBTEXT  = "#94A3B8"

CHART_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Fira Sans, Inter, Arial, sans-serif", color=TEXT),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=SUBTEXT)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=SUBTEXT)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
)

def styled_bar(x, y, color, name="", text=None):
    return go.Bar(
        x=x, y=y, name=name,
        marker=dict(
            color=color,
            opacity=0.9,
            line=dict(color=color, width=0),
        ),
        text=[f"{v:.1f}%" for v in text] if text is not None else None,
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
    )

# ══════════════════════════════════════════════════════════════════════════════
# Compute segment data
# ══════════════════════════════════════════════════════════════════════════════
card_churn = (df.groupby("Card_Category")["Churn"].mean() * 100).sort_values(ascending=False).reset_index()
card_churn.columns = ["Card_Category", "Churn_Rate"]

income_churn = (df.groupby("Income_Category")["Churn"].mean() * 100).sort_values(ascending=False).reset_index()
income_churn.columns = ["Income_Category", "Churn_Rate"]

edu_churn = (df.groupby("Education_Level")["Churn"].mean() * 100).sort_values(ascending=False).reset_index()
edu_churn.columns = ["Education_Level", "Churn_Rate"]

gender_churn = (df.groupby("Gender")["Churn"].mean() * 100).sort_values(ascending=False).reset_index()
gender_churn.columns = ["Gender", "Churn_Rate"]

inactive_churn = (df.groupby("Months_Inactive_12_mon")["Churn"].mean() * 100).reset_index()
inactive_churn.columns = ["Months_Inactive", "Churn_Rate"]

contacts_churn = (df.groupby("Contacts_Count_12_mon")["Churn"].mean() * 100).reset_index()
contacts_churn.columns = ["Contacts", "Churn_Rate"]

def risk_segment(u):
    if u >= 0.7:    return "High Risk"
    elif u >= 0.4:  return "Medium Risk"
    else:           return "Low Risk"

def spend_segment(t):
    if t >= 4500:   return "Top Spenders"
    elif t >= 2000: return "Mid Spenders"
    else:           return "Low Spenders"

df["Risk_Segment"]  = df["Utilization_Ratio"].apply(risk_segment)
df["Spend_Segment"] = df["Total_Trans_Amt"].apply(spend_segment)

risk_stats = df.groupby("Risk_Segment").agg(Total=("Churn","count"), Churned=("Churn","sum")).reset_index()
risk_stats["Churn_Rate"] = (risk_stats["Churned"] / risk_stats["Total"] * 100).round(2)
risk_order = ["High Risk", "Medium Risk", "Low Risk"]
risk_stats = risk_stats.set_index("Risk_Segment").reindex(risk_order).reset_index()

spend_stats = df.groupby("Spend_Segment").agg(Total=("Churn","count"), Churned=("Churn","sum")).reset_index()
spend_stats["Churn_Rate"] = (spend_stats["Churned"] / spend_stats["Total"] * 100).round(2)

trans_churn = df.groupby("Churn")["Total_Trans_Amt"].mean().reset_index()
trans_churn["Label"] = trans_churn["Churn"].map({0: "Retained", 1: "Churned"})

# ══════════════════════════════════════════════════════════════════════════════
# Build individual chart figures
# ══════════════════════════════════════════════════════════════════════════════

# Chart 1 — Donut
fig_donut = go.Figure(go.Pie(
    labels=["Retained", "Churned"],
    values=[retained_customers, churned_customers],
    hole=0.65,
    marker=dict(colors=[GREEN, RED], line=dict(color=BG, width=3)),
    textinfo="label+percent",
    textfont=dict(color=TEXT, size=13),
))
fig_donut.update_layout(
    **CHART_LAYOUT,
    title=dict(text="Churn vs Retained", font=dict(size=15, color=TEXT), x=0.5),
    showlegend=False,
    annotations=[dict(text=f"<b>{churn_rate}%</b><br>Churn", x=0.5, y=0.5,
                      font=dict(size=16, color=AMBER), showarrow=False)],
    height=320,
)

# Chart 2 — Churn by card category
fig_card = go.Figure(styled_bar(
    card_churn["Card_Category"], card_churn["Churn_Rate"], CYAN,
    text=card_churn["Churn_Rate"]
))
fig_card.update_layout(**CHART_LAYOUT,
    title=dict(text="Churn Rate by Card Category", font=dict(size=15, color=TEXT), x=0.5),
    height=320, yaxis_title="Churn Rate %")

# Chart 3 — Churn by income
fig_income = go.Figure(styled_bar(
    income_churn["Income_Category"], income_churn["Churn_Rate"], PURPLE,
    text=income_churn["Churn_Rate"]
))
fig_income.update_layout(**CHART_LAYOUT,
    title=dict(text="Churn Rate by Income Bracket", font=dict(size=15, color=TEXT), x=0.5),
    height=320, yaxis_title="Churn Rate %")
fig_income.update_xaxes(tickangle=-20)

# Chart 4 — Avg transaction by churn
fig_trans = go.Figure(go.Bar(
    x=trans_churn["Label"],
    y=trans_churn["Total_Trans_Amt"].round(0),
    marker=dict(color=[GREEN, RED]),
    text=["$" + str(int(v)) for v in trans_churn["Total_Trans_Amt"].round(0)],
    textposition="outside",
    textfont=dict(color=TEXT, size=12),
))
fig_trans.update_layout(**CHART_LAYOUT,
    title=dict(text="Avg Transaction Amount by Churn Status", font=dict(size=15, color=TEXT), x=0.5),
    height=320, yaxis_title="Avg Transaction $")

# Chart 5 — Churn by education
fig_edu = go.Figure(styled_bar(
    edu_churn["Education_Level"], edu_churn["Churn_Rate"], BLUE,
    text=edu_churn["Churn_Rate"]
))
fig_edu.update_layout(**CHART_LAYOUT,
    title=dict(text="Churn Rate by Education Level", font=dict(size=15, color=TEXT), x=0.5),
    height=320, yaxis_title="Churn Rate %")

# Chart 6 — Scatter
sample = df.sample(min(2000, len(df)), random_state=42)
fig_scatter = go.Figure()
for churn_val, label, color in [(0, "Retained", GREEN), (1, "Churned", RED)]:
    mask = sample["Churn"] == churn_val
    fig_scatter.add_trace(go.Scatter(
        x=sample.loc[mask, "Total_Trans_Amt"],
        y=sample.loc[mask, "Utilization_Ratio"],
        mode="markers", name=label,
        marker=dict(color=color, size=4, opacity=0.5),
    ))
fig_scatter.update_layout(**CHART_LAYOUT,
    title=dict(text="Transaction Amount vs Utilization Ratio", font=dict(size=15, color=TEXT), x=0.5),
    height=320, xaxis_title="Total Transaction Amount", yaxis_title="Utilization Ratio")

# Chart 7 — Churn by gender
fig_gender = go.Figure(styled_bar(
    gender_churn["Gender"], gender_churn["Churn_Rate"], AMBER,
    text=gender_churn["Churn_Rate"]
))
fig_gender.update_layout(**CHART_LAYOUT,
    title=dict(text="Churn Rate by Gender", font=dict(size=15, color=TEXT), x=0.5),
    height=320, yaxis_title="Churn Rate %")

# Chart 8 — Inactivity line
fig_inactive = go.Figure(go.Scatter(
    x=inactive_churn["Months_Inactive"], y=inactive_churn["Churn_Rate"],
    mode="lines+markers",
    line=dict(color=RED, width=3),
    marker=dict(size=9, color=RED, line=dict(color=BG, width=2)),
    fill="tozeroy", fillcolor="rgba(244,63,94,0.12)",
))
fig_inactive.update_layout(**CHART_LAYOUT,
    title=dict(text="Months Inactive vs Churn Rate", font=dict(size=15, color=TEXT), x=0.5),
    height=320, xaxis_title="Months Inactive", yaxis_title="Churn Rate %")

# Chart 9 — Risk segments
fig_risk = go.Figure(go.Bar(
    x=risk_stats["Risk_Segment"], y=risk_stats["Churn_Rate"],
    marker=dict(color=[RED, AMBER, GREEN]),
    text=[f"{v:.1f}%" for v in risk_stats["Churn_Rate"]],
    textposition="outside", textfont=dict(color=TEXT, size=12),
))
fig_risk.update_layout(**CHART_LAYOUT,
    title=dict(text="Churn Rate by Risk Segment", font=dict(size=15, color=TEXT), x=0.5),
    height=320, yaxis_title="Churn Rate %")

# Chart 10 — Spending segments
fig_spend = go.Figure(styled_bar(
    spend_stats["Spend_Segment"], spend_stats["Churn_Rate"], PURPLE,
    text=spend_stats["Churn_Rate"]
))
fig_spend.update_layout(**CHART_LAYOUT,
    title=dict(text="Churn Rate by Spending Segment", font=dict(size=15, color=TEXT), x=0.5),
    height=320, yaxis_title="Churn Rate %")

# Chart 11 — Utilization histogram
fig_util = go.Figure()
for churn_val, label, color in [(0, "Retained", GREEN), (1, "Churned", RED)]:
    mask = df["Churn"] == churn_val
    fig_util.add_trace(go.Histogram(
        x=df.loc[mask, "Utilization_Ratio"], name=label,
        opacity=0.7, marker_color=color, nbinsx=40,
    ))
fig_util.update_layout(**CHART_LAYOUT,
    title=dict(text="Utilization Ratio Distribution", font=dict(size=15, color=TEXT), x=0.5),
    barmode="overlay", height=320,
    xaxis_title="Utilization Ratio", yaxis_title="Count")

# Chart 12 — Contacts vs churn
fig_contacts = go.Figure(go.Bar(
    x=contacts_churn["Contacts"], y=contacts_churn["Churn_Rate"],
    marker=dict(
        color=contacts_churn["Churn_Rate"],
        colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
        showscale=False,
    ),
    text=[f"{v:.1f}%" for v in contacts_churn["Churn_Rate"]],
    textposition="outside", textfont=dict(color=TEXT, size=11),
))
fig_contacts.update_layout(**CHART_LAYOUT,
    title=dict(text="Contacts Count vs Churn Rate", font=dict(size=15, color=TEXT), x=0.5),
    height=320, xaxis_title="Contacts in Last 12 Months", yaxis_title="Churn Rate %")

# ══════════════════════════════════════════════════════════════════════════════
# Convert all charts to HTML snippets
# ══════════════════════════════════════════════════════════════════════════════
charts = [fig_donut, fig_card, fig_income, fig_trans,
          fig_edu, fig_scatter, fig_gender, fig_inactive,
          fig_risk, fig_spend, fig_util, fig_contacts]

chart_html = []
for i, fig in enumerate(charts):
    include_js = True if i == 0 else False
    chart_html.append(fig.to_html(full_html=False, include_plotlyjs=include_js,
                                   config={"displayModeBar": False}))

# ══════════════════════════════════════════════════════════════════════════════
# Assemble full HTML
# ══════════════════════════════════════════════════════════════════════════════
def kpi(value, label, color, prefix="", suffix=""):
    return f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{color}">{prefix}{value}{suffix}</div>
      <div class="kpi-label">{label}</div>
    </div>"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Credit Card Customer Intelligence Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Fira Sans', Inter, Arial, sans-serif;
      background: {BG};
      color: {TEXT};
      min-height: 100vh;
      padding: 0 0 60px 0;
    }}

    /* ── Header ── */
    .header {{
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
      border-bottom: 1px solid {BORDER};
      padding: 28px 40px 24px;
      text-align: center;
    }}
    .header h1 {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, {CYAN}, {BLUE});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 6px;
    }}
    .header p {{
      color: {SUBTEXT};
      font-size: 14px;
      font-weight: 400;
    }}
    .badge {{
      display: inline-block;
      background: rgba(34,211,238,0.12);
      color: {CYAN};
      border: 1px solid rgba(34,211,238,0.3);
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 12px;
      font-weight: 600;
      margin: 0 4px;
    }}

    /* ── Nav tabs ── */
    .nav-tabs {{
      display: flex;
      gap: 4px;
      padding: 16px 40px 0;
      border-bottom: 1px solid {BORDER};
      background: {BG};
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .tab-btn {{
      background: none;
      border: none;
      color: {SUBTEXT};
      font-family: inherit;
      font-size: 13px;
      font-weight: 500;
      padding: 10px 20px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      border-radius: 6px 6px 0 0;
    }}
    .tab-btn:hover {{ color: {TEXT}; background: rgba(255,255,255,0.04); }}
    .tab-btn.active {{
      color: {CYAN};
      border-bottom-color: {CYAN};
      background: rgba(34,211,238,0.06);
    }}

    /* ── Page sections ── */
    .page {{ display: none; padding: 32px 40px; }}
    .page.active {{ display: block; }}

    /* ── KPI cards ── */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 20px 24px;
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .kpi-card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}
    .kpi-value {{
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -1px;
      line-height: 1;
      margin-bottom: 6px;
    }}
    .kpi-label {{
      font-size: 12px;
      font-weight: 500;
      color: {SUBTEXT};
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }}

    /* ── Chart grid ── */
    .chart-grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .chart-grid-3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .chart-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 12px;
      overflow: hidden;
      transition: box-shadow 0.2s;
    }}
    .chart-card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.25); }}

    /* ── Section label ── */
    .section-label {{
      font-size: 11px;
      font-weight: 600;
      color: {SUBTEXT};
      text-transform: uppercase;
      letter-spacing: 1.2px;
      margin-bottom: 16px;
    }}

    /* ── Back to top ── */
    .back-to-top {{
      position: fixed; bottom: 24px; right: 24px;
      width: 44px; height: 44px; border-radius: 50%;
      background: {CYAN}; color: {BG};
      border: none; cursor: pointer;
      opacity: 0; visibility: hidden;
      transition: all 0.2s; font-size: 20px;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 12px rgba(34,211,238,0.4);
    }}
    .back-to-top.visible {{ opacity: 1; visibility: visible; }}
    .back-to-top:hover {{ transform: scale(1.1); }}
  </style>
</head>
<body>

  <!-- Header -->
  <div class="header">
    <h1>💳 Credit Card Customer Intelligence Platform</h1>
    <p>
      <span class="badge">10,127 Customers</span>
      <span class="badge">XGBoost AUC 0.9932</span>
      <span class="badge">Churn Rate {churn_rate}%</span>
    </p>
  </div>

  <!-- Nav Tabs -->
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="showPage('p1', this)">📊 Executive Overview</button>
    <button class="tab-btn" onclick="showPage('p2', this)">👥 Customer Segmentation</button>
    <button class="tab-btn" onclick="showPage('p3', this)">⚠️ Risk Analysis</button>
  </div>

  <!-- Page 1 — Executive Overview -->
  <div id="p1" class="page active">
    <div class="section-label">Key Performance Indicators</div>
    <div class="kpi-row">
      {kpi(f"{total_customers:,}", "Total Customers", CYAN)}
      {kpi(f"{churned_customers:,}", "Churned Customers", RED)}
      {kpi(churn_rate, "Churn Rate", AMBER, suffix="%")}
      {kpi(f"{int(avg_credit_limit):,}", "Avg Credit Limit", GREEN, prefix="$")}
    </div>
    <div class="chart-grid-2">
      <div class="chart-card">{chart_html[0]}</div>
      <div class="chart-card">{chart_html[1]}</div>
    </div>
    <div class="chart-grid-2">
      <div class="chart-card">{chart_html[2]}</div>
      <div class="chart-card">{chart_html[3]}</div>
    </div>
  </div>

  <!-- Page 2 — Customer Segmentation -->
  <div id="p2" class="page">
    <div class="section-label">Customer Segmentation Analysis</div>
    <div class="chart-grid-2">
      <div class="chart-card">{chart_html[4]}</div>
      <div class="chart-card">{chart_html[5]}</div>
    </div>
    <div class="chart-grid-2">
      <div class="chart-card">{chart_html[6]}</div>
      <div class="chart-card">{chart_html[7]}</div>
    </div>
  </div>

  <!-- Page 3 — Risk Analysis -->
  <div id="p3" class="page">
    <div class="section-label">Risk & Behavioural Analysis</div>
    <div class="chart-grid-2">
      <div class="chart-card">{chart_html[8]}</div>
      <div class="chart-card">{chart_html[9]}</div>
    </div>
    <div class="chart-grid-2">
      <div class="chart-card">{chart_html[10]}</div>
      <div class="chart-card">{chart_html[11]}</div>
    </div>
  </div>

  <!-- Back to top -->
  <button class="back-to-top" aria-label="Back to top" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑</button>

  <script>
    function showPage(id, btn) {{
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(id).classList.add('active');
      btn.classList.add('active');
      window.scrollTo({{top: 0, behavior: 'smooth'}});
    }}
    const btt = document.querySelector('.back-to-top');
    window.addEventListener('scroll', () => btt.classList.toggle('visible', window.scrollY > 300));
  </script>
</body>
</html>"""

output_path = os.path.join(BASE_DIR, "credit_card_dashboard.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"✅ Dashboard saved → {output_path}")
print("   Open credit_card_dashboard.html in Chrome to view it.")
