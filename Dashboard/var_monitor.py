"""
VaR Monitor — All Commodities
Parametric 1-Day VaR at 99% confidence (rolling window: 20D / 60D / 120D)
Run: streamlit run var_monitor.py
"""

import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Rollex utils (Master Database — single source of truth) ──────────────────
_ROLLEX_DIR = Path(__file__).resolve().parents[1]
_DB_DIR     = _ROLLEX_DIR / "Database"
for _p in [_ROLLEX_DIR, Path(__file__).resolve().parent, Path.cwd(), Path.cwd().parent]:
    if (_p / "rollex_utils.py").exists():
        sys.path.insert(0, str(_p))
        break
from rollex_utils import load_rollex as _rx_load

# commodity code → parquet filename in repo Database/
_PARQUET_MAP = {
    "KC": "master_KC.parquet", "LRC": "master_RC.parquet", "CC": "master_CC.parquet",
    "LCC": "master_LCC.parquet", "SB": "master_SB.parquet", "CT": "master_CT.parquet", "LSU": "master_LSU.parquet",
}

def _load_front_price(comm: str) -> pd.DataFrame:
    """Return a Date-indexed DataFrame with settlement price and active contract name."""
    path = _DB_DIR / _PARQUET_MAP[comm]
    raw  = pd.read_parquet(path, columns=["Date", "FND", "settlement", "base_ric"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["FND"]  = pd.to_datetime(raw["FND"])
    raw = raw.dropna(subset=["settlement"])
    # Front contract on each date = earliest FND that is still >= that date
    active = (
        raw[raw["FND"] >= raw["Date"]]
        .sort_values(["Date", "FND"])
        .groupby("Date")[["settlement", "base_ric"]]
        .first()
    )
    return active  # DatetimeIndex → {settlement, base_ric}

st.set_page_config(page_title="VaR Monitor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
  [data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:#fafafa!important;color:#1d1d1f!important}
  [data-testid="stHeader"]{background:transparent!important}
  .block-container{padding-top:2rem!important;padding-bottom:1.5rem;max-width:1440px}
  hr{border:none!important;border-top:1px solid #e8e8ed!important;margin:.4rem 0!important}
  [data-testid="stRadio"] label,[data-testid="stRadio"] label p,[data-testid="stRadio"] label div{font-size:.78rem!important;color:#1d1d1f!important}
  [data-testid="stExpander"]{border:1px solid #e8e8ed!important;border-radius:8px!important;background:#fff!important}
  h1,h2,h3{color:#1d1d1f!important;font-weight:500!important}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NAVY  = "#0a2463"
BLACK = "#1d1d1f"

LOT_SIZES = {"KC": 375, "LRC": 10, "CC": 10, "LCC": 10, "SB": 1120, "CT": 500, "LSU": 50}
COLORS    = {"KC": "#0a2463", "LRC": "#8b1a00", "CC": "#e8a020", "LCC": "#4a7fb5",
             "SB": "#1a6b1a", "CT": "#7b2d8b", "LSU": "#c0392b"}
NAMES     = {"KC": "Arabica", "LRC": "Robusta", "CC": "NYC Cocoa", "LCC": "London Cocoa",
             "SB": "Sugar #11", "CT": "Cotton", "LSU": "White Sugar"}

COMBINED = {
    "Coffee (KC+LRC)": ["KC",  "LRC"],
    "Cocoa (CC+LCC)":  ["CC",  "LCC"],
    "Sugar (SB+LSU)":  ["SB",  "LSU"],
}
COMBINED_COLORS = {
    "Coffee (KC+LRC)": "#2a4a7a",
    "Cocoa (CC+LCC)":  "#c87010",
    "Sugar (SB+LSU)":  "#2a8a2a",
}

CONF_Z  = 2.5758
WINDOWS = {"20D": 20, "60D": 60, "120D": 120}
MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,Helvetica Neue,sans-serif", color=BLACK, size=10),
)

def lbl(text):
    return (f"<div style='background:{NAVY};padding:5px 13px;border-radius:5px;"
            f"margin-bottom:8px'><span style='font-size:.78rem;font-weight:500;"
            f"letter-spacing:.07em;text-transform:uppercase;color:#dde4f0'>{text}</span></div>")

# ── Data loading & VaR computation ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    data = {}
    for comm in LOT_SIZES:
        # ── Returns & vol: rollex continuous price ────────────────────────────
        rx = _rx_load(comm)[["rollex_px"]].reset_index()
        rx.columns = ["Date", "Close"]
        rx["Date"] = pd.to_datetime(rx["Date"])
        rx = rx.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        full_idx = pd.bdate_range(rx["Date"].min(), rx["Date"].max())
        rx = rx.set_index("Date").reindex(full_idx).ffill()
        rx["log_ret"] = np.log(rx["Close"] / rx["Close"].shift(1))
        for w_name, w in WINDOWS.items():
            rx[f"vol_{w_name}"] = rx["log_ret"].rolling(w).std()

        # ── Price for VaR: active front contract settlement ───────────────────
        front = _load_front_price(comm)
        df = rx.join(front, how="left")
        df["settlement"] = df["settlement"].ffill()
        df["base_ric"]   = df["base_ric"].ffill()

        for w_name in WINDOWS:
            df[f"VaR_{w_name}"] = df["settlement"] * LOT_SIZES[comm] * df[f"vol_{w_name}"] * CONF_Z

        data[comm] = df.reset_index().rename(columns={"index": "Date"})
    return data

data = load_all()

# ── Helpers ───────────────────────────────────────────────────────────────────
INDIV_OPTIONS = {f"{comm} — {NAMES[comm]}": comm for comm in LOT_SIZES}
ALL_OPTIONS   = list(INDIV_OPTIONS.keys()) + list(COMBINED.keys())

def _var_series(label: str, var_col: str) -> pd.DataFrame:
    if label in INDIV_OPTIONS:
        comm = INDIV_OPTIONS[label]
        return data[comm][["Date", var_col, "base_ric"]].copy().rename(columns={var_col: "VaR", "base_ric": "contract"})
    else:
        comms  = COMBINED[label]
        frames = [data[c].set_index("Date")[var_col] for c in comms]
        combined = pd.concat(frames, axis=1).ffill()
        s = combined.sum(axis=1, min_count=len(comms)).reset_index()
        s.columns = ["Date", "VaR"]
        # For combined, show both active contracts e.g. "KCH6 + RCH6"
        contracts = pd.concat([data[c].set_index("Date")["base_ric"] for c in comms], axis=1).ffill()
        contracts.columns = comms
        s["contract"] = contracts.apply(lambda r: " + ".join(r.dropna().values), axis=1)
        return s

def _label_meta(label: str):
    if label in INDIV_OPTIONS:
        comm = INDIV_OPTIONS[label]
        return NAMES[comm], COLORS[comm]
    return label, COMBINED_COLORS[label]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='font-family:\"Playfair Display\",Georgia,serif;color:#0a2463;"
    "font-weight:400;letter-spacing:-.01em;margin-bottom:2px'>VaR Monitor</h2>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Global date bounds ────────────────────────────────────────────────────────
all_dates     = pd.concat([data[c]["Date"] for c in data]).dropna()
min_d         = all_dates.min().date()
max_d         = all_dates.max().date()
default_start = (all_dates.max() - pd.DateOffset(years=5)).date()

st.markdown(
    f"<i style='font-size:.75rem;color:#888'>Data as of {max_d.strftime('%b %d, %Y')}</i>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Collapsible Filters + Line Chart
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Controls", expanded=True):
    c1, c2, c3 = st.columns([3, 4, 2])
    with c1:
        selected_labels = st.multiselect(
            "Commodities", ALL_OPTIONS,
            default=["KC — Arabica", "LRC — Robusta"],
            key="ms_main",
        )
    with c2:
        date_range = st.slider(
            "Date range", min_value=min_d, max_value=max_d,
            value=(default_start, max_d), key="sl_main",
        )
    with c3:
        window_label = st.radio(
            "VaR Window", list(WINDOWS.keys()), index=1,
            horizontal=True, key="radio_window",
        )

var_col        = f"VaR_{window_label}"
start_d, end_d = date_range

st.markdown(lbl(f"1-Day VaR · 99% Confidence · {window_label} Rolling Window · Per Lot"), unsafe_allow_html=True)

fig_line = go.Figure()
for label in selected_labels:
    name, color = _label_meta(label)
    s = _var_series(label, var_col)
    s = s[(s["Date"].dt.date >= start_d) & (s["Date"].dt.date <= end_d)]
    fig_line.add_trace(go.Scatter(
        x=s["Date"], y=s["VaR"].round(0),
        name=name, mode="lines",
        line=dict(color=color, width=1.8),
        customdata=s["contract"],
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>VaR: $%{y:,.0f}<br>Contract: %{customdata}<extra>" + name + "</extra>",
    ))

fig_line.update_layout(
    height=380,
    xaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
               tickfont=dict(size=9, color=BLACK), title="VaR (USD / lot)"),
    legend=dict(orientation="h", y=1.02, x=0,
                font=dict(size=8, color=BLACK), bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(t=10, b=10, l=4, r=4), **_D,
)
st.plotly_chart(fig_line, width="100%")

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Vol Percentile Bar
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(lbl(f"Current Volatility Percentile vs Full History · {window_label}"), unsafe_allow_html=True)

pct_rows = []
for comm in LOT_SIZES:
    vol_col = f"vol_{window_label}"
    hist    = data[comm][vol_col].dropna()
    if hist.empty:
        continue
    cur_vol = hist.iloc[-1]
    pct     = float((hist < cur_vol).mean() * 100)
    cur_var = data[comm][var_col].dropna().iloc[-1]
    pct_rows.append({
        "Commodity":   NAMES[comm],
        "Percentile":  round(pct, 1),
        "Current VaR": f"${cur_var:,.0f}",
        "color":       COLORS[comm],
    })

pct_df = pd.DataFrame(pct_rows).sort_values("Percentile", ascending=True)

fig_pct = go.Figure(go.Bar(
    x=pct_df["Percentile"], y=pct_df["Commodity"],
    orientation="h",
    marker_color=pct_df["color"],
    text=pct_df.apply(lambda r: f"{r['Percentile']:.0f}th  |  {r['Current VaR']}", axis=1),
    textposition="outside",
    textfont=dict(size=9, color=BLACK),
))
fig_pct.add_vline(x=50, line_dash="dot", line_color="#aaaaaa", line_width=1,
                  annotation_text="50th", annotation_font=dict(size=8, color="#aaaaaa"),
                  annotation_position="top")
fig_pct.add_vline(x=80, line_dash="dot", line_color="#e07b39", line_width=1,
                  annotation_text="80th", annotation_font=dict(size=8, color="#e07b39"),
                  annotation_position="top")
fig_pct.update_layout(
    height=300,
    xaxis=dict(range=[0, 120], showgrid=False,
               tickfont=dict(size=9, color=BLACK), title="Percentile"),
    yaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
    margin=dict(t=10, b=10, l=4, r=120), **_D,
)
st.plotly_chart(fig_pct, width="100%")

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Heatmap (Years × Months)
# ═══════════════════════════════════════════════════════════════════════════════
hm_label = st.selectbox("Commodity for Heatmap", ALL_OPTIONS, index=0, key="hm_sel")
st.markdown(lbl(f"Monthly Avg VaR Heatmap · {window_label} · {_label_meta(hm_label)[0]}"), unsafe_allow_html=True)

hm_s = _var_series(hm_label, var_col)
hm_s = hm_s[(hm_s["Date"].dt.date >= start_d) & (hm_s["Date"].dt.date <= end_d)].dropna(subset=["VaR"])
hm_s["Year"]  = hm_s["Date"].dt.year
hm_s["Month"] = hm_s["Date"].dt.month

pivot = (
    hm_s.groupby(["Year", "Month"])["VaR"]
    .mean()
    .reset_index()
    .pivot(index="Year", columns="Month", values="VaR")
)
pivot.columns = [MONTHS[m - 1] for m in pivot.columns]
pivot = pivot.sort_index(ascending=False)

z         = pivot.values
years     = [str(y) for y in pivot.index]
months    = list(pivot.columns)
text_vals = [[f"${v:,.0f}" if not np.isnan(v) else "" for v in row] for row in z]

fig_hm = go.Figure(go.Heatmap(
    z=z, x=months, y=years,
    text=text_vals,
    texttemplate="%{text}",
    textfont=dict(size=8, color=BLACK),
    colorscale=[
        [0.0, "#d4edda"],
        [0.4, "#fff3cd"],
        [0.7, "#f8d7a0"],
        [1.0, "#f5c6cb"],
    ],
    showscale=True,
    colorbar=dict(
        title=dict(text="VaR (USD)", font=dict(size=9, color=BLACK)),
        tickfont=dict(size=8, color=BLACK),
        thickness=12, len=0.8,
    ),
    hoverongaps=False,
    hovertemplate="<b>%{y} %{x}</b><br>Avg VaR: $%{z:,.0f}<extra></extra>",
))
fig_hm.update_layout(
    height=max(300, len(years) * 28),
    xaxis=dict(side="top", tickfont=dict(size=9, color=BLACK), showgrid=False),
    yaxis=dict(tickfont=dict(size=9, color=BLACK), showgrid=False),
    margin=dict(t=40, b=10, l=60, r=10), **_D,
)
st.plotly_chart(fig_hm, width="100%")
