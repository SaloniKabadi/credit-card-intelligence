"""
Credit Card Customer Intelligence — Editorial Dashboard
--------------------------------------------------------
A hand-crafted, magazine-style dashboard.  Heavy on serif type,
bento-grid layout, and tasteful motion.  Built to feel like
something a designer made on a Tuesday afternoon, not a boilerplate.
"""

import os
import pandas as pd
import plotly.graph_objects as go


# ── Load Data ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "../data/processed/clean_customers.csv"))

total_customers    = len(df)
churned_customers  = int(df["Churn"].sum())
retained_customers = total_customers - churned_customers
churn_rate         = round(df["Churn"].mean() * 100, 2)
retention_rate     = round(100 - churn_rate, 2)
avg_credit_limit   = int(round(df["Credit_Limit"].mean(), 0))
avg_utilization    = round(df["Utilization_Ratio"].mean() * 100, 2)
avg_trans_amt      = int(round(df["Total_Trans_Amt"].mean(), 0))
avg_age            = int(round(df["Customer_Age"].mean(), 0))


# ── Palette: warm-dark editorial ──────────────────────────────────────────────
INK        = "#0E0E10"   # near-black, slightly warm
PAPER      = "#F5F1E8"   # cream paper
PAPER_DIM  = "#E9E2D2"
CARD       = "#16161A"
CARD_HI    = "#1F1F25"
LINE       = "#2A2A33"
MUTED      = "#8A8A92"
BONE       = "#D8D2C2"

# Accent set — slightly off-spec, like print inks
LIME       = "#D4FF3A"   # signature accent
CORAL      = "#FF6B4A"   # warning / churn
SKY        = "#7AC7FF"   # cool
BUTTER     = "#FFD66B"   # caution
MOSS       = "#7BCB7B"   # retained
PLUM       = "#C8A8FF"   # neutral

CHART_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family="'Inter', system-ui, sans-serif", color=BONE, size=12),
    margin=dict(l=46, r=24, t=58, b=44),
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, tickfont=dict(color=MUTED, size=11)),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, tickfont=dict(color=MUTED, size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=BONE, size=11)),
    hoverlabel=dict(bgcolor=INK, bordercolor=LIME, font=dict(family="'JetBrains Mono', monospace", color=PAPER)),
)

def title_block(text):
    return dict(text=text, font=dict(size=14, color=PAPER, family="'Fraunces', serif"), x=0.02, xanchor="left")

def styled_bar(x, y, color, name="", text=None):
    return go.Bar(
        x=x, y=y, name=name,
        marker=dict(color=color, line=dict(color=color, width=0)),
        text=[f"{v:.1f}%" for v in text] if text is not None else None,
        textposition="outside",
        textfont=dict(color=BONE, size=11, family="'JetBrains Mono', monospace"),
        hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Aggregations
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
    if u >= 0.7:
        return "High Risk"
    elif u >= 0.4:
        return "Medium Risk"
    return "Low Risk"


def spend_segment(t):
    if t >= 4500:
        return "Top Spenders"
    elif t >= 2000:
        return "Mid Spenders"
    return "Low Spenders"


df["Risk_Segment"]  = df["Utilization_Ratio"].apply(risk_segment)
df["Spend_Segment"] = df["Total_Trans_Amt"].apply(spend_segment)

risk_stats = df.groupby("Risk_Segment").agg(Total=("Churn", "count"), Churned=("Churn", "sum")).reset_index()
risk_stats["Churn_Rate"] = (risk_stats["Churned"] / risk_stats["Total"] * 100).round(2)
risk_stats = risk_stats.set_index("Risk_Segment").reindex(["High Risk", "Medium Risk", "Low Risk"]).reset_index()

spend_stats = df.groupby("Spend_Segment").agg(Total=("Churn", "count"), Churned=("Churn", "sum")).reset_index()
spend_stats["Churn_Rate"] = (spend_stats["Churned"] / spend_stats["Total"] * 100).round(2)

trans_churn = df.groupby("Churn")["Total_Trans_Amt"].mean().reset_index()
trans_churn["Label"] = trans_churn["Churn"].map({0: "Retained", 1: "Churned"})


# ══════════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════════

# 1 — Donut
fig_donut = go.Figure(go.Pie(
    labels=["Retained", "Churned"],
    values=[retained_customers, churned_customers],
    hole=0.72,
    marker=dict(colors=[MOSS, CORAL], line=dict(color=CARD, width=4)),
    textinfo="label+percent",
    textfont=dict(color=PAPER, size=12, family="'JetBrains Mono', monospace"),
    hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
    rotation=110,
    sort=False,
))
fig_donut.update_layout(
    **CHART_LAYOUT,
    title=title_block("Retention Pulse"),
    showlegend=False,
    height=340,
    annotations=[
        dict(text=f"<b>{churn_rate}%</b>", x=0.5, y=0.56,
             font=dict(size=34, color=LIME, family="'Fraunces', serif"), showarrow=False),
        dict(text="left us", x=0.5, y=0.42,
             font=dict(size=11, color=MUTED, family="'JetBrains Mono', monospace"), showarrow=False),
    ],
)


# 2 — Card category
fig_card = go.Figure(styled_bar(
    card_churn["Card_Category"], card_churn["Churn_Rate"], SKY,
    text=card_churn["Churn_Rate"]
))
fig_card.update_layout(**CHART_LAYOUT,
    title=title_block("Churn by Card Tier"),
    height=340, yaxis_title=None, xaxis_title=None)

# 3 — Income
fig_income = go.Figure(styled_bar(
    income_churn["Income_Category"], income_churn["Churn_Rate"], PLUM,
    text=income_churn["Churn_Rate"]
))
fig_income.update_layout(**CHART_LAYOUT,
    title=title_block("Churn by Income Bracket"),
    height=340, yaxis_title=None, xaxis_title=None)
fig_income.update_xaxes(tickangle=-25)

# 4 — Avg transaction by churn
fig_trans = go.Figure(go.Bar(
    x=trans_churn["Label"],
    y=trans_churn["Total_Trans_Amt"].round(0),
    marker=dict(color=[MOSS, CORAL]),
    text=["$" + f"{int(v):,}" for v in trans_churn["Total_Trans_Amt"].round(0)],
    textposition="outside",
    textfont=dict(color=BONE, size=13, family="'JetBrains Mono', monospace"),
    hovertemplate="<b>%{x}</b><br>$%{y:,.0f} avg<extra></extra>",
    width=[0.45, 0.45],
))
fig_trans.update_layout(**CHART_LAYOUT,
    title=title_block("Spend Gap: Stayers vs Leavers"),
    height=340, yaxis_title=None, xaxis_title=None)

# 5 — Education
fig_edu = go.Figure(styled_bar(
    edu_churn["Education_Level"], edu_churn["Churn_Rate"], BUTTER,
    text=edu_churn["Churn_Rate"]
))
fig_edu.update_layout(**CHART_LAYOUT,
    title=title_block("Churn by Education"),
    height=340, yaxis_title=None, xaxis_title=None)
fig_edu.update_xaxes(tickangle=-20)

# 6 — Scatter (Trans vs Utilization)
sample = df.sample(min(2000, len(df)), random_state=42)
fig_scatter = go.Figure()
for churn_val, label, color in [(0, "Retained", MOSS), (1, "Churned", CORAL)]:
    mask = sample["Churn"] == churn_val
    fig_scatter.add_trace(go.Scatter(
        x=sample.loc[mask, "Total_Trans_Amt"],
        y=sample.loc[mask, "Utilization_Ratio"],
        mode="markers", name=label,
        marker=dict(color=color, size=5, opacity=0.55, line=dict(width=0)),
        hovertemplate="$%{x:,.0f} • util %{y:.2f}<extra></extra>",
    ))
fig_scatter.update_layout(**CHART_LAYOUT,
    title=title_block("Spend × Utilization — the behavioural map"),
    height=340, xaxis_title="Total Transaction $", yaxis_title="Utilization Ratio")

# 7 — Gender
fig_gender = go.Figure(styled_bar(
    gender_churn["Gender"], gender_churn["Churn_Rate"], BUTTER,
    text=gender_churn["Churn_Rate"]
))
fig_gender.update_layout(**CHART_LAYOUT,
    title=title_block("Churn by Gender"),
    height=340, yaxis_title=None, xaxis_title=None)

# 8 — Inactivity line
fig_inactive = go.Figure(go.Scatter(
    x=inactive_churn["Months_Inactive"], y=inactive_churn["Churn_Rate"],
    mode="lines+markers",
    line=dict(color=CORAL, width=3, shape="spline"),
    marker=dict(size=10, color=CORAL, line=dict(color=CARD, width=2)),
    fill="tozeroy", fillcolor="rgba(255,107,74,0.14)",
    hovertemplate="%{x} months idle → %{y:.1f}%<extra></extra>",
))
fig_inactive.update_layout(**CHART_LAYOUT,
    title=title_block("The silence curve — inactivity vs churn"),
    height=340, xaxis_title="Months Inactive", yaxis_title="Churn Rate %")

# 9 — Risk segments
fig_risk = go.Figure(go.Bar(
    x=risk_stats["Risk_Segment"], y=risk_stats["Churn_Rate"],
    marker=dict(color=[CORAL, BUTTER, MOSS]),
    text=[f"{v:.1f}%" for v in risk_stats["Churn_Rate"]],
    textposition="outside",
    textfont=dict(color=BONE, size=12, family="'JetBrains Mono', monospace"),
    hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
    width=[0.5, 0.5, 0.5],
))
fig_risk.update_layout(**CHART_LAYOUT,
    title=title_block("Risk Segment × Churn"),
    height=340, yaxis_title=None, xaxis_title=None)

# 10 — Spending segments
fig_spend = go.Figure(styled_bar(
    spend_stats["Spend_Segment"], spend_stats["Churn_Rate"], PLUM,
    text=spend_stats["Churn_Rate"]
))
fig_spend.update_layout(**CHART_LAYOUT,
    title=title_block("Churn by Spending Tier"),
    height=340, yaxis_title=None, xaxis_title=None)

# 11 — Utilization histogram
fig_util = go.Figure()
for churn_val, label, color in [(0, "Retained", MOSS), (1, "Churned", CORAL)]:
    mask = df["Churn"] == churn_val
    fig_util.add_trace(go.Histogram(
        x=df.loc[mask, "Utilization_Ratio"], name=label,
        opacity=0.75, marker_color=color, nbinsx=40,
    ))
fig_util.update_layout(**CHART_LAYOUT,
    title=title_block("Utilization shape, side-by-side"),
    barmode="overlay", height=340,
    xaxis_title="Utilization Ratio", yaxis_title="Customers")

# 12 — Contacts
fig_contacts = go.Figure(go.Bar(
    x=contacts_churn["Contacts"], y=contacts_churn["Churn_Rate"],
    marker=dict(
        color=contacts_churn["Churn_Rate"],
        colorscale=[[0, MOSS], [0.5, BUTTER], [1, CORAL]],
        showscale=False,
    ),
    text=[f"{v:.1f}%" for v in contacts_churn["Churn_Rate"]],
    textposition="outside",
    textfont=dict(color=BONE, size=11, family="'JetBrains Mono', monospace"),
    hovertemplate="%{x} contacts → %{y:.1f}%<extra></extra>",
))
fig_contacts.update_layout(**CHART_LAYOUT,
    title=title_block("Service contacts vs churn"),
    height=340, xaxis_title="Contacts in last 12 mo.", yaxis_title="Churn %")


# ══════════════════════════════════════════════════════════════════════════════
# Render to HTML
# ══════════════════════════════════════════════════════════════════════════════
charts = [fig_donut, fig_card, fig_income, fig_trans,
          fig_edu, fig_scatter, fig_gender, fig_inactive,
          fig_risk, fig_spend, fig_util, fig_contacts]

chart_html = []
for i, fig in enumerate(charts):
    chart_html.append(fig.to_html(
        full_html=False,
        include_plotlyjs=(i == 0),
        config={"displayModeBar": False, "responsive": True},
    ))


# ══════════════════════════════════════════════════════════════════════════════
# HTML / CSS / JS — editorial layout
# ══════════════════════════════════════════════════════════════════════════════
top_card = card_churn.iloc[0]
top_income = income_churn.iloc[0]
peak_inactive = inactive_churn.sort_values("Churn_Rate", ascending=False).iloc[0]

CSS = """
:root{
  --ink:#0E0E10; --paper:#F5F1E8; --paper2:#E9E2D2;
  --card:#16161A; --card2:#1F1F25; --line:#2A2A33;
  --muted:#8A8A92; --bone:#D8D2C2;
  --lime:#D4FF3A; --coral:#FF6B4A; --sky:#7AC7FF;
  --butter:#FFD66B; --moss:#7BCB7B; --plum:#C8A8FF;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:var(--ink); color:var(--paper);
  font-family:'Inter',system-ui,sans-serif;
  font-size:15px; line-height:1.5;
  min-height:100vh; overflow-x:hidden;
  background-image:
    radial-gradient(ellipse 1200px 600px at 10% -5%, rgba(212,255,58,0.07), transparent 60%),
    radial-gradient(ellipse 900px 500px at 100% 30%, rgba(255,107,74,0.06), transparent 60%);
}
::selection{background:var(--lime); color:var(--ink)}

/* Grain overlay — subtle film noise */
body::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:1;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.06 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  opacity:.5; mix-blend-mode:overlay;
}

/* ── Top bar ── */
.topbar{
  position:sticky; top:0; z-index:50;
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 36px; background:rgba(14,14,16,0.85);
  backdrop-filter:blur(14px) saturate(140%);
  border-bottom:1px solid var(--line);
}
.brand{display:flex; align-items:center; gap:12px; font-family:'Fraunces',serif; font-weight:600; font-size:18px; letter-spacing:-0.3px}
.brand .dot{width:10px;height:10px;border-radius:50%;background:var(--lime); box-shadow:0 0 14px var(--lime); animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{50%{opacity:.4; transform:scale(.85)}}
.topbar nav{display:flex; gap:4px}
.tab{
  background:none; border:none; cursor:pointer;
  color:var(--muted); font:500 13px/1 'Inter',sans-serif; letter-spacing:.4px;
  padding:10px 16px; border-radius:999px; transition:all .25s ease;
}
.tab:hover{color:var(--paper); background:rgba(255,255,255,0.04)}
.tab.active{color:var(--ink); background:var(--lime)}
.meta{font:400 12px/1 'JetBrains Mono',monospace; color:var(--muted)}
.meta b{color:var(--paper)}

/* ── Ticker ── */
.ticker{
  position:relative; overflow:hidden;
  border-bottom:1px solid var(--line);
  background:linear-gradient(90deg, rgba(212,255,58,0.06), transparent 40%, transparent 60%, rgba(255,107,74,0.06));
  font:500 12px/1 'JetBrains Mono',monospace; color:var(--bone);
  height:38px; display:flex; align-items:center;
}
.ticker-track{display:flex; gap:48px; white-space:nowrap; animation:slide 60s linear infinite; padding-left:36px}
.ticker:hover .ticker-track{animation-play-state:paused}
.ticker span{display:inline-flex; align-items:center; gap:8px}
.ticker em{color:var(--lime); font-style:normal}
.ticker .sep{color:var(--line)}
@keyframes slide{to{transform:translateX(-50%)}}

/* ── Hero ── */
.hero{
  padding:64px 36px 28px; max-width:1480px; margin:0 auto;
  position:relative;
}
.eyebrow{
  display:inline-flex; align-items:center; gap:10px;
  font:500 11px/1 'JetBrains Mono',monospace; letter-spacing:.18em;
  color:var(--muted); text-transform:uppercase; margin-bottom:18px;
}
.eyebrow::before{content:""; width:34px; height:1px; background:var(--lime)}
.hero h1{
  font-family:'Fraunces',serif; font-weight:500;
  font-size: clamp(40px, 6.4vw, 92px);
  line-height:0.96; letter-spacing:-0.025em;
  color:var(--paper); margin-bottom:22px; max-width:14ch;
}
.hero h1 .ital{font-style:italic; font-weight:400; color:var(--lime)}
.hero h1 .strike{position:relative; display:inline-block}
.hero h1 .strike::after{
  content:""; position:absolute; left:-4%; right:-4%; top:55%; height:6px;
  background:var(--coral); transform:scaleX(0); transform-origin:left;
  animation:strike 1.4s .8s cubic-bezier(.2,.8,.2,1) forwards;
}
@keyframes strike{to{transform:scaleX(1)}}
.hero p.lede{
  max-width:62ch; color:var(--bone); font-size:17px; line-height:1.55;
}
.hero p.lede b{color:var(--paper); font-weight:600}

/* Decorative byline */
.byline{
  display:flex; gap:24px; align-items:center; margin-top:28px;
  font:500 12px/1 'JetBrains Mono',monospace; color:var(--muted);
}
.byline .pill{padding:6px 12px; border:1px solid var(--line); border-radius:999px}
.byline .pill b{color:var(--lime); font-weight:600}

/* ── Page sections ── */
main{max-width:1480px; margin:0 auto; padding:0 36px 96px; position:relative; z-index:2}
.page{display:none; animation:fadein .55s ease both}
.page.active{display:block}
@keyframes fadein{from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none}}

.section-rule{
  display:flex; align-items:baseline; gap:16px; margin:48px 0 22px;
}
.section-rule .num{
  font:500 12px/1 'JetBrains Mono',monospace; color:var(--lime); letter-spacing:.2em;
}
.section-rule h2{
  font-family:'Fraunces',serif; font-weight:500; font-size:30px; letter-spacing:-0.02em;
  color:var(--paper);
}
.section-rule h2 i{font-weight:400; color:var(--bone)}
.section-rule .line{flex:1; height:1px; background:var(--line)}
.section-rule .stamp{
  font:500 11px/1 'JetBrains Mono',monospace; color:var(--muted); letter-spacing:.2em;
}

/* ── KPI bento ── */
.bento{
  display:grid; gap:14px;
  grid-template-columns:repeat(12, 1fr);
  grid-auto-rows:minmax(0,auto);
}
.tile{
  position:relative; overflow:hidden;
  background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:24px; transition:transform .35s cubic-bezier(.2,.8,.2,1), border-color .25s, box-shadow .35s;
  opacity:0; transform:translateY(18px);
}
.tile.in{opacity:1; transform:none}
.tile:hover{transform:translateY(-4px); border-color:#3a3a44; box-shadow:0 18px 50px -12px rgba(0,0,0,.7)}
.tile .corner{
  position:absolute; top:14px; right:14px;
  font:500 10px/1 'JetBrains Mono',monospace; color:var(--muted); letter-spacing:.2em;
}
.tile .label{
  font:500 11px/1 'JetBrains Mono',monospace; color:var(--muted);
  text-transform:uppercase; letter-spacing:.2em; margin-bottom:14px;
}
.tile .num{
  font-family:'Fraunces',serif; font-weight:500;
  font-size:54px; line-height:1; letter-spacing:-.03em; color:var(--paper);
}
.tile .num small{font-size:26px; color:var(--muted); margin-left:2px}
.tile .delta{
  margin-top:10px; display:inline-flex; align-items:center; gap:8px;
  font:500 12px/1 'JetBrains Mono',monospace; color:var(--bone);
}
.tile .delta .chip{padding:3px 8px; border-radius:6px; background:rgba(212,255,58,0.12); color:var(--lime); font-size:11px}
.tile .delta .chip.warn{background:rgba(255,107,74,0.12); color:var(--coral)}
.tile.accent{background:linear-gradient(135deg, var(--lime) 0%, #B8E83A 100%); color:var(--ink); border-color:transparent}
.tile.accent .num,.tile.accent .label,.tile.accent .corner,.tile.accent .delta{color:var(--ink)}
.tile.accent .label{opacity:.7}

/* Grid spans */
.span-3{grid-column:span 3}
.span-4{grid-column:span 4}
.span-5{grid-column:span 5}
.span-6{grid-column:span 6}
.span-7{grid-column:span 7}
.span-8{grid-column:span 8}
.span-12{grid-column:span 12}

/* Chart cards */
.chart-card{
  background:var(--card); border:1px solid var(--line); border-radius:18px;
  overflow:hidden; transition:transform .35s, border-color .25s, box-shadow .35s;
  opacity:0; transform:translateY(18px); position:relative;
}
.chart-card.in{opacity:1; transform:none}
.chart-card:hover{transform:translateY(-3px); border-color:#3a3a44; box-shadow:0 22px 60px -16px rgba(0,0,0,.65)}
.chart-card .head{
  padding:16px 22px 0; display:flex; justify-content:space-between; align-items:center;
}
.chart-card .head .tag{
  font:500 10px/1 'JetBrains Mono',monospace; letter-spacing:.2em;
  color:var(--muted); text-transform:uppercase;
}
.chart-card .plotly-graph-div{margin-top:-8px}

/* Pull-quote tile */
.quote{
  background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:34px; font-family:'Fraunces',serif; font-size:24px; line-height:1.3;
  color:var(--bone); position:relative; overflow:hidden;
  opacity:0; transform:translateY(18px); transition:opacity .6s, transform .6s;
}
.quote.in{opacity:1; transform:none}
.quote::before{
  content:"“"; position:absolute; top:-22px; left:18px;
  font-family:'Fraunces',serif; font-size:160px; color:var(--lime); opacity:.18; line-height:1;
}
.quote em{color:var(--paper); font-style:italic}
.quote .sig{
  margin-top:18px; font:500 11px/1 'JetBrains Mono',monospace;
  color:var(--muted); letter-spacing:.2em; text-transform:uppercase;
}

/* Footer */
footer{
  border-top:1px solid var(--line); margin-top:64px; padding:28px 36px;
  display:flex; justify-content:space-between; align-items:center;
  font:500 12px/1 'JetBrains Mono',monospace; color:var(--muted);
}
footer b{color:var(--bone)}

/* Back-to-top */
.btt{
  position:fixed; bottom:26px; right:26px; z-index:60;
  width:48px; height:48px; border-radius:50%; border:none;
  background:var(--lime); color:var(--ink); font-size:20px; cursor:pointer;
  opacity:0; visibility:hidden; transition:all .25s;
  box-shadow:0 10px 30px -8px rgba(212,255,58,.6);
}
.btt.visible{opacity:1; visibility:visible}
.btt:hover{transform:translateY(-3px) rotate(-8deg)}

/* Cursor accent — a soft spotlight that follows */
.spotlight{
  position:fixed; top:0; left:0; width:480px; height:480px; border-radius:50%;
  background:radial-gradient(circle, rgba(212,255,58,0.08) 0%, transparent 60%);
  pointer-events:none; z-index:0; transform:translate(-50%,-50%); transition:transform .15s ease-out;
  mix-blend-mode:screen;
}

/* Responsive */
@media (max-width: 1100px){
  .span-3,.span-4,.span-5,.span-6,.span-7,.span-8{grid-column:span 6}
  .hero h1{font-size:54px}
}
@media (max-width: 720px){
  .topbar{padding:12px 18px}
  .topbar nav{display:none}
  .hero{padding:40px 18px 18px}
  main{padding:0 18px 60px}
  .span-3,.span-4,.span-5,.span-6,.span-7,.span-8{grid-column:span 12}
}
"""


JS = """
// ── Tabs ──
function showPage(id, btn){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  // re-run reveal for newly visible tiles
  requestAnimationFrame(()=>revealOnView(true));
  // give Plotly a kick so charts size correctly
  setTimeout(()=>{ if(window.Plotly){ document.querySelectorAll('.js-plotly-plot').forEach(el=>Plotly.Plots.resize(el)); } }, 50);
}

// ── Count-up animation for KPI numbers ──
function animateCount(el){
  const target = parseFloat(el.dataset.target);
  const decimals = parseInt(el.dataset.decimals || '0', 10);
  const prefix = el.dataset.prefix || '';
  const suffix = el.dataset.suffix || '';
  const duration = 1400;
  const start = performance.now();
  function frame(now){
    const t = Math.min(1, (now-start)/duration);
    const eased = 1 - Math.pow(1-t, 3);
    const v = target * eased;
    el.textContent = prefix + (decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString()) + suffix;
    if(t < 1) requestAnimationFrame(frame);
    else el.textContent = prefix + (decimals ? target.toFixed(decimals) : Math.round(target).toLocaleString()) + suffix;
  }
  requestAnimationFrame(frame);
}

// ── Scroll reveal ──
const seen = new WeakSet();
function revealOnView(force=false){
  const els = document.querySelectorAll('.tile, .chart-card, .quote');
  const vh = window.innerHeight;
  els.forEach((el, i)=>{
    if(seen.has(el) && !force) return;
    const r = el.getBoundingClientRect();
    if(r.top < vh - 60){
      setTimeout(()=>{
        el.classList.add('in');
        const num = el.querySelector('.num[data-target]');
        if(num && !num.dataset.done){ num.dataset.done='1'; animateCount(num); }
      }, (i % 6) * 60);
      seen.add(el);
    }
  });
}
window.addEventListener('scroll', ()=>revealOnView(), {passive:true});
window.addEventListener('load', ()=>revealOnView(true));

// ── Spotlight cursor ──
const spot = document.querySelector('.spotlight');
window.addEventListener('mousemove', e=>{
  if(!spot) return;
  spot.style.transform = `translate(${e.clientX}px, ${e.clientY}px) translate(-50%,-50%)`;
});

// ── Back to top ──
const btt = document.querySelector('.btt');
window.addEventListener('scroll', ()=>btt.classList.toggle('visible', window.scrollY > 400));

// ── Live clock ──
function tick(){
  const el = document.getElementById('live-clock');
  if(!el) return;
  const d = new Date();
  el.textContent = d.toLocaleTimeString('en-US', {hour12:false});
}
setInterval(tick, 1000); tick();

// ── Resize plotly on window resize ──
window.addEventListener('resize', ()=>{
  if(window.Plotly) document.querySelectorAll('.js-plotly-plot').forEach(el=>Plotly.Plots.resize(el));
});
"""


def kpi_tile(value, label, *, span=3, accent=False, decimals=0, prefix="", suffix="", note=None, note_kind="ok", code=""):
    cls = "tile accent" if accent else "tile"
    chip = ""
    if note:
        chip_cls = "chip warn" if note_kind == "warn" else "chip"
        chip = f'<div class="delta"><span class="{chip_cls}">{note}</span></div>'
    return f"""
    <div class="{cls} span-{span}">
      <div class="corner">{code}</div>
      <div class="label">{label}</div>
      <div class="num" data-target="{value}" data-decimals="{decimals}" data-prefix="{prefix}" data-suffix="{suffix}">{prefix}0{suffix}</div>
      {chip}
    </div>"""


def chart_tile(html_snippet, tag, *, span=6):
    return f"""
    <div class="chart-card span-{span}">
      <div class="head"><span class="tag">{tag}</span></div>
      {html_snippet}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — The Front Page (Executive overview)
# ══════════════════════════════════════════════════════════════════════════════
page1 = f"""
<div id="p1" class="page active">

  <section class="hero">
    <div class="eyebrow">Vol. 01 · Issue 04 · Customer Intelligence Quarterly</div>
    <h1>
      One in <span class="ital">six</span> walks away.<br>
      <span class="strike">Forever?</span> &nbsp;Not if we can help it.
    </h1>
    <p class="lede">
      A close look at <b>{total_customers:,}</b> credit-card holders, the ones who stayed,
      the ones who didn&rsquo;t, and the small signals that gave them away weeks before
      they cancelled. Built on top of an XGBoost model running at <b>AUC 0.9932</b>.
    </p>
    <div class="byline">
      <span class="pill">By <b>Risk &amp; Retention Lab</b></span>
      <span>Updated <b id="live-clock">--:--:--</b></span>
      <span>Refreshes hourly</span>
    </div>
  </section>

  <div class="section-rule">
    <span class="num">§ 01</span>
    <h2>The state of the book <i>— at a glance</i></h2>
    <span class="line"></span>
    <span class="stamp">KPI · LIVE</span>
  </div>

  <div class="bento">
    {kpi_tile(total_customers, "Total customers on file", span=3, code="01")}
    {kpi_tile(churn_rate, "Churn rate", span=3, accent=True, decimals=2, suffix="%", code="02", note=f"{churned_customers:,} lost", note_kind="warn")}
    {kpi_tile(avg_credit_limit, "Avg credit limit", span=3, prefix="$", code="03")}
    {kpi_tile(avg_utilization, "Avg utilization", span=3, decimals=2, suffix="%", code="04")}
    {kpi_tile(retention_rate, "Retention rate", span=3, decimals=2, suffix="%", code="05", note="healthy", note_kind="ok")}
    {kpi_tile(avg_trans_amt, "Avg yearly spend", span=3, prefix="$", code="06")}
    {kpi_tile(avg_age, "Avg customer age", span=3, suffix=" yrs", code="07")}
    {kpi_tile(churned_customers, "Walked away", span=3, code="08", note="this cohort", note_kind="warn")}
  </div>

  <div class="section-rule">
    <span class="num">§ 02</span>
    <h2>Who&rsquo;s leaving, who&rsquo;s staying</h2>
    <span class="line"></span>
    <span class="stamp">FIG · 01—04</span>
  </div>

  <div class="bento">
    {chart_tile(chart_html[0], "Fig. 01 — Donut", span=5)}
    {chart_tile(chart_html[1], "Fig. 02 — Card tier", span=7)}
    {chart_tile(chart_html[2], "Fig. 03 — Income brackets", span=7)}
    {chart_tile(chart_html[3], "Fig. 04 — The spend gap", span=5)}
  </div>

  <div class="section-rule">
    <span class="num">§ 03</span>
    <h2>A word from the data</h2>
    <span class="line"></span>
  </div>

  <div class="bento">
    <div class="quote span-12">
      Customers on the <em>{top_card['Card_Category']}</em> tier churn at
      <em>{top_card['Churn_Rate']:.1f}%</em>, while the <em>{top_income['Income_Category']}</em>
      bracket leads income-based churn at <em>{top_income['Churn_Rate']:.1f}%</em>.
      Inactivity peaks the alarm: <em>{int(peak_inactive['Months_Inactive'])} months</em> idle
      pushes churn to <em>{peak_inactive['Churn_Rate']:.1f}%</em>.
      <div class="sig">— Pattern, not coincidence.</div>
    </div>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Segmentation
# ══════════════════════════════════════════════════════════════════════════════
page2 = f"""
<div id="p2" class="page">
  <section class="hero" style="padding-top:48px;padding-bottom:8px">
    <div class="eyebrow">Chapter Two · Segmentation</div>
    <h1>The shape of the <span class="ital">customer base</span>.</h1>
    <p class="lede">Slice by education, gender, behaviour. The book isn&rsquo;t one block of people.
    It&rsquo;s a dozen smaller stories.</p>
  </section>

  <div class="section-rule">
    <span class="num">§ 04</span>
    <h2>Demographics <i>— who they are</i></h2>
    <span class="line"></span>
    <span class="stamp">FIG · 05—08</span>
  </div>

  <div class="bento">
    {chart_tile(chart_html[4], "Fig. 05 — Education", span=7)}
    {chart_tile(chart_html[6], "Fig. 06 — Gender", span=5)}
    {chart_tile(chart_html[5], "Fig. 07 — Behavioural map", span=7)}
    {chart_tile(chart_html[7], "Fig. 08 — Inactivity arc", span=5)}
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Risk
# ══════════════════════════════════════════════════════════════════════════════
page3 = f"""
<div id="p3" class="page">
  <section class="hero" style="padding-top:48px;padding-bottom:8px">
    <div class="eyebrow">Chapter Three · Risk Signals</div>
    <h1>The quiet ones <span class="ital">leave loudest</span>.</h1>
    <p class="lede">Four signals predict churn before the cancel button is clicked: utilization,
    inactivity, contact frequency and spend tier. They tell on each other.</p>
  </section>

  <div class="section-rule">
    <span class="num">§ 05</span>
    <h2>Risk &amp; behaviour <i>— the levers we can pull</i></h2>
    <span class="line"></span>
    <span class="stamp">FIG · 09—12</span>
  </div>

  <div class="bento">
    {chart_tile(chart_html[8], "Fig. 09 — Risk segments", span=5)}
    {chart_tile(chart_html[9], "Fig. 10 — Spend tiers", span=7)}
    {chart_tile(chart_html[10], "Fig. 11 — Utilization shape", span=7)}
    {chart_tile(chart_html[11], "Fig. 12 — Service contacts", span=5)}
  </div>

  <div class="bento" style="margin-top:14px">
    <div class="quote span-12">
      The customers who call us <em>four or more times</em> are the customers we&rsquo;re about to lose.
      The customers who go quiet for <em>three months</em> are the customers we&rsquo;ve already lost &mdash;
      they just haven&rsquo;t told us yet.
      <div class="sig">— Risk &amp; Retention Lab, 2026</div>
    </div>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Final HTML
# ══════════════════════════════════════════════════════════════════════════════
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Credit Card Intelligence · A Quarterly</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>{CSS}</style>
</head>
<body>
  <div class="spotlight"></div>

  <header class="topbar">
    <div class="brand"><span class="dot"></span> Card &middot; Intelligence <span style="color:var(--muted);font-weight:400">/ a quarterly</span></div>
    <nav>
      <button class="tab active" onclick="showPage('p1', this)">Front Page</button>
      <button class="tab" onclick="showPage('p2', this)">Segmentation</button>
      <button class="tab" onclick="showPage('p3', this)">Risk Signals</button>
    </nav>
    <div class="meta">vol·01 &nbsp; iss·04 &nbsp; <b>{total_customers:,}</b> rows</div>
  </header>

  <div class="ticker" aria-hidden="true">
    <div class="ticker-track">
      <span>customers <em>{total_customers:,}</em></span><span class="sep">·</span>
      <span>churn <em>{churn_rate}%</em></span><span class="sep">·</span>
      <span>retained <em>{retained_customers:,}</em></span><span class="sep">·</span>
      <span>avg credit <em>${avg_credit_limit:,}</em></span><span class="sep">·</span>
      <span>avg utilization <em>{avg_utilization}%</em></span><span class="sep">·</span>
      <span>avg spend <em>${avg_trans_amt:,}</em></span><span class="sep">·</span>
      <span>top churn tier <em>{top_card['Card_Category']}</em></span><span class="sep">·</span>
      <span>model AUC <em>0.9932</em></span><span class="sep">·</span>
      <!-- duplicate for seamless loop -->
      <span>customers <em>{total_customers:,}</em></span><span class="sep">·</span>
      <span>churn <em>{churn_rate}%</em></span><span class="sep">·</span>
      <span>retained <em>{retained_customers:,}</em></span><span class="sep">·</span>
      <span>avg credit <em>${avg_credit_limit:,}</em></span><span class="sep">·</span>
      <span>avg utilization <em>{avg_utilization}%</em></span><span class="sep">·</span>
      <span>avg spend <em>${avg_trans_amt:,}</em></span><span class="sep">·</span>
      <span>top churn tier <em>{top_card['Card_Category']}</em></span><span class="sep">·</span>
      <span>model AUC <em>0.9932</em></span><span class="sep">·</span>
    </div>
  </div>

  <main>
    {page1}
    {page2}
    {page3}
  </main>

  <footer>
    <span>© 2026 · Risk &amp; Retention Lab · <b>set in Fraunces &amp; Inter</b></span>
    <span>built by hand · pour over &amp; iterate</span>
  </footer>

  <button class="btt" aria-label="Back to top" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑</button>

  <script>{JS}</script>
</body>
</html>"""


output_path = os.path.join(BASE_DIR, "credit_card_dashboard.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"✅ Dashboard saved → {output_path}")
print("   Open credit_card_dashboard.html in a browser to view it.")
