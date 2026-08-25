"""
Pakistan Air Intelligence
AI-Powered Air Quality Monitoring & PM2.5 Prediction Platform

Built on top of a pre-trained Tuned Gradient Boosting model + ColumnTransformer
preprocessor. No retraining happens in this app — all ML artifacts are loaded
directly from assets/models/.
"""

import warnings
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ============================================================================
# PATHS  (all relative to this file, so the app runs from any location)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATASET_PATH = ASSETS_DIR / "dataset" / "pakistan_air_quality_final_clean_v2.csv"
DQ_REPORT_PATH = ASSETS_DIR / "dataset" / "DATA_QUALITY_REPORT.md"
MODEL_PATH = ASSETS_DIR / "models" / "tuned_gradient_boosting_pm25.joblib"
PREPROCESSOR_PATH = ASSETS_DIR / "models" / "pm25_preprocessor.joblib"
FEATURE_IMPORTANCE_PATH = ASSETS_DIR / "models" / "feature_importance.csv"
MODEL_COMPARISON_PATH = ASSETS_DIR / "models" / "final_model_comparison.csv"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Pakistan Air Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# AQI / PM2.5 CLASSIFICATION SCALE
# This is the signature visual system reused across KPI cards, predictions,
# rankings and charts — color always encodes the same real pollution severity.
# ============================================================================
AQI_SCALE = [
    {"name": "Good", "max": 12.0, "color": "#3DD68C",
     "desc": "Air quality is satisfactory and poses little or no risk."},
    {"name": "Moderate", "max": 35.4, "color": "#E8C34A",
     "desc": "Acceptable air quality. Unusually sensitive individuals should consider limiting prolonged outdoor exertion."},
    {"name": "Unhealthy for Sensitive Groups", "max": 55.4, "color": "#F0954A",
     "desc": "Sensitive groups (children, elderly, respiratory/heart conditions) may experience health effects."},
    {"name": "Unhealthy", "max": 150.4, "color": "#EA6262",
     "desc": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."},
    {"name": "Very Unhealthy", "max": 250.4, "color": "#B57BE0",
     "desc": "Health alert — everyone may experience more serious health effects."},
    {"name": "Hazardous", "max": float("inf"), "color": "#8B3A3A",
     "desc": "Health warning of emergency conditions. The entire population is likely to be affected."},
]


def classify_pm25(value: float) -> dict:
    """Map a PM2.5 concentration (µg/m³) onto the AQI_SCALE bands."""
    if value is None or pd.isna(value):
        return {"name": "Unknown", "max": None, "color": "#6B7686", "desc": "Insufficient data."}
    for band in AQI_SCALE:
        if value <= band["max"]:
            return band
    return AQI_SCALE[-1]


def aqi_color_for_category(category: str) -> str:
    """Map a dataset aqi_category string onto the same color system."""
    for band in AQI_SCALE:
        if band["name"] == category:
            return band["color"]
    return "#6B7686"


# ============================================================================
# GLOBAL CSS — dark "environmental command-center" theme
# ============================================================================
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root{
    --bg-0:#080b10;
    --bg-1:#0c1119;
    --bg-2:#111826;
    --panel:rgba(255,255,255,0.035);
    --panel-border:rgba(255,255,255,0.09);
    --panel-border-hover:rgba(255,255,255,0.18);
    --text-primary:#E9EDF4;
    --text-secondary:#8B96A8;
    --text-tertiary:#5B6576;
    --accent:#F2A93B;
    --accent-soft:rgba(242,169,59,0.14);
    --teal:#35D0BA;
    --teal-soft:rgba(53,208,186,0.14);
    --good:#3DD68C;
    --danger:#EA6262;
}

/* base app */
.stApp{
    background:
        radial-gradient(ellipse 1200px 600px at 15% -10%, rgba(53,208,186,0.07), transparent 60%),
        radial-gradient(ellipse 1000px 700px at 110% 10%, rgba(242,169,59,0.06), transparent 55%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 40%, var(--bg-0) 100%);
    color: var(--text-primary);
    font-family:'Inter', sans-serif;
}
/* faint radar grid overlay */
.stApp::before{
    content:"";
    position:fixed; inset:0;
    background-image:
        linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
    background-size: 42px 42px;
    pointer-events:none;
    z-index:0;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.6rem; padding-bottom:3rem; max-width:1400px;}

h1,h2,h3,h4,h5{
    font-family:'Space Grotesk', sans-serif !important;
    color:var(--text-primary) !important;
    letter-spacing:-0.01em;
}
p, span, div, label, li{ color:var(--text-primary); }
.stMarkdown p{ color:var(--text-secondary); }

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0a0e15 0%, #0d1320 100%);
    border-right:1px solid var(--panel-border);
}
section[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}

/* ---------------- HERO ---------------- */
.hero-wrap{
    position:relative;
    padding:2.6rem 2.4rem;
    border-radius:22px;
    background:
        radial-gradient(700px 260px at 12% 0%, rgba(53,208,186,0.14), transparent 65%),
        radial-gradient(700px 320px at 95% 30%, rgba(242,169,59,0.12), transparent 60%),
        linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border:1px solid var(--panel-border);
    overflow:hidden;
    margin-bottom:1.8rem;
    animation: fadeIn 0.6s ease;
}
.hero-eyebrow{
    display:inline-flex; align-items:center; gap:0.5rem;
    font-family:'JetBrains Mono', monospace;
    font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase;
    color:var(--teal);
    background:var(--teal-soft);
    border:1px solid rgba(53,208,186,0.35);
    padding:0.3rem 0.75rem; border-radius:100px;
    margin-bottom:1rem;
}
.hero-eyebrow .dot{ width:6px; height:6px; border-radius:50%; background:var(--teal); box-shadow:0 0 8px var(--teal); animation:pulse 2s infinite; }
.hero-title{ font-size:2.6rem; font-weight:700; margin:0 0 0.35rem 0; line-height:1.05;
    background:linear-gradient(90deg, #ffffff, #b9c3d4);
    -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero-sub{ font-size:1.08rem; color:var(--text-secondary); font-weight:500; margin-bottom:0.9rem;}
.hero-desc{ font-size:0.95rem; color:var(--text-tertiary); max-width:760px; line-height:1.6; }
.status-row{ display:flex; gap:0.9rem; margin-top:1.3rem; flex-wrap:wrap;}
.status-chip{
    display:flex; align-items:center; gap:0.5rem;
    background:rgba(255,255,255,0.035); border:1px solid var(--panel-border);
    padding:0.45rem 0.9rem; border-radius:10px;
    font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:var(--text-secondary);
}
.status-chip .led{ width:8px; height:8px; border-radius:50%; background:var(--good); box-shadow:0 0 8px var(--good); animation:pulse 2s infinite;}

@keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:0.35;} }
@keyframes fadeIn{ from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:translateY(0);} }

/* ---------------- GLASS CARD / KPI ---------------- */
.glass-card{
    background:var(--panel);
    border:1px solid var(--panel-border);
    border-radius:16px;
    padding:1.3rem 1.4rem;
    transition: all 0.25s ease;
    animation: fadeIn 0.5s ease;
}
.glass-card:hover{
    border-color:var(--panel-border-hover);
    transform:translateY(-3px);
    box-shadow:0 12px 28px rgba(0,0,0,0.35);
}
.kpi-icon{ font-size:1.3rem; opacity:0.9; margin-bottom:0.6rem; display:block;}
.kpi-value{ font-family:'JetBrains Mono', monospace; font-size:1.85rem; font-weight:700; color:var(--text-primary); line-height:1.1;}
.kpi-label{ font-size:0.8rem; color:var(--text-secondary); margin-top:0.35rem; font-weight:500;}
.kpi-sub{ font-size:0.72rem; color:var(--text-tertiary); margin-top:0.15rem; }

.section-title{
    font-size:1.35rem; font-weight:600; margin:0 0 0.2rem 0;
    display:flex; align-items:center; gap:0.55rem;
}
.section-caption{ color:var(--text-tertiary); font-size:0.87rem; margin-bottom:1.1rem;}

/* AQI pill */
.aqi-pill{
    display:inline-flex; align-items:center; gap:0.45rem;
    padding:0.32rem 0.85rem; border-radius:100px;
    font-size:0.78rem; font-weight:600; font-family:'JetBrains Mono', monospace;
}
.aqi-dot{ width:8px; height:8px; border-radius:50%; }

/* AQI gradient scale bar — the signature element, reused everywhere */
.aqi-scale-bar{ height:10px; border-radius:6px; width:100%;
    background:linear-gradient(90deg, #3DD68C 0%, #E8C34A 20%, #F0954A 40%, #EA6262 60%, #B57BE0 80%, #8B3A3A 100%);
    position:relative; margin:0.9rem 0 0.4rem 0;}
.aqi-scale-marker{ position:absolute; top:-5px; width:3px; height:20px; background:#fff; border-radius:2px;
    box-shadow:0 0 8px rgba(255,255,255,0.9); }
.aqi-scale-labels{ display:flex; justify-content:space-between; font-size:0.62rem; color:var(--text-tertiary); font-family:'JetBrains Mono', monospace;}

/* Prediction result card */
.predict-result{
    text-align:center; padding:2.2rem 1.5rem; border-radius:20px;
    border:1px solid var(--panel-border);
    background:linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    animation: fadeIn 0.5s ease;
}
.predict-label{ font-family:'JetBrains Mono', monospace; letter-spacing:0.16em; text-transform:uppercase;
    font-size:0.75rem; color:var(--text-tertiary); margin-bottom:0.6rem;}
.predict-value{ font-family:'JetBrains Mono', monospace; font-size:3.4rem; font-weight:700; line-height:1;}
.predict-unit{ font-size:1rem; color:var(--text-secondary); margin-left:0.3rem;}

/* footer */
.app-footer{
    margin-top:3rem; padding-top:1.8rem; border-top:1px solid var(--panel-border);
    text-align:center; color:var(--text-tertiary); font-size:0.82rem;
}
.footer-tags{ display:flex; justify-content:center; gap:0.6rem; flex-wrap:wrap; margin:0.8rem 0;}
.footer-tag{ background:rgba(255,255,255,0.04); border:1px solid var(--panel-border);
    padding:0.25rem 0.7rem; border-radius:100px; font-size:0.72rem; color:var(--text-secondary);}

/* streamlit widget overrides */
div[data-testid="stMetric"]{
    background:var(--panel); border:1px solid var(--panel-border); border-radius:14px; padding:1rem 1.1rem;
}
.stButton>button{
    background:linear-gradient(135deg, var(--accent), #d98c1f);
    color:#1a1206; border:none; border-radius:10px; font-weight:700;
    padding:0.6rem 1.4rem; transition:all 0.2s ease; font-family:'Inter', sans-serif;
}
.stButton>button:hover{ transform:translateY(-2px); box-shadow:0 10px 24px rgba(242,169,59,0.28); }
.stTabs [data-baseweb="tab-list"]{ gap:4px; }
.stTabs [data-baseweb="tab"]{ background:var(--panel); border-radius:8px 8px 0 0; color:var(--text-secondary); }
.stDataFrame{ border-radius:12px; overflow:hidden; border:1px solid var(--panel-border); }
hr{ border-color:var(--panel-border); }

/* selectbox / inputs */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input{
    background:var(--panel) !important; border-color:var(--panel-border) !important; color:var(--text-primary) !important;
}
</style>
"""


# ============================================================================
# DATA / MODEL LOADERS
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    fi["Feature_Clean"] = (
        fi["Feature"].str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ")
    )
    return fi


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_dq_report_text() -> str:
    return DQ_REPORT_PATH.read_text()


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the exact feature-engineering pipeline used to train the
    Tuned Gradient Boosting model (see assets/pakistan-air-intelligence.ipynb):
    cyclical time features, PM2.5 / pollutant lag features, rolling PM2.5
    statistics, PM2.5 trend/change features, and the next-hour target.
    """
    d = df.sort_values(["city", "timestamp"]).reset_index(drop=True).copy()

    d["day_of_week_num"] = d["timestamp"].dt.dayofweek
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["dayofweek_sin"] = np.sin(2 * np.pi * d["day_of_week_num"] / 7)
    d["dayofweek_cos"] = np.cos(2 * np.pi * d["day_of_week_num"] / 7)

    for lag in [1, 2, 3, 6, 12, 24]:
        d[f"pm2_5_lag_{lag}h"] = d.groupby("city")["pm2_5"].shift(lag)

    for feat in ["pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust"]:
        for lag in [1, 3, 6, 24]:
            d[f"{feat}_lag_{lag}h"] = d.groupby("city")[feat].shift(lag)

    for w in [3, 6, 12, 24]:
        d[f"pm2_5_rolling_mean_{w}h"] = d.groupby("city")["pm2_5"].transform(
            lambda x: x.rolling(w, min_periods=w).mean()
        )
        d[f"pm2_5_rolling_std_{w}h"] = d.groupby("city")["pm2_5"].transform(
            lambda x: x.rolling(w, min_periods=w).std()
        )

    d["pm2_5_change_1h"] = d["pm2_5"] - d["pm2_5_lag_1h"]
    d["pm2_5_change_3h"] = d["pm2_5"] - d["pm2_5_lag_3h"]
    d["pm2_5_change_6h"] = d["pm2_5"] - d["pm2_5_lag_6h"]
    d["pm2_5_change_24h"] = d["pm2_5"] - d["pm2_5_lag_24h"]

    d["target_pm2_5"] = d.groupby("city")["pm2_5"].shift(-1)

    return d


def get_pipeline_columns(preprocessor):
    """Pull the exact numerical / categorical column order the fitted
    ColumnTransformer expects, straight from the artifact — never hardcoded."""
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])
    return num_cols, cat_cols


# ============================================================================
# SMALL UI HELPERS
# ============================================================================
def render_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def kpi_card(icon, value, label, sub=""):
    st.markdown(
        f"""
        <div class="glass-card">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon, title, caption=""):
    st.markdown(
        f"""
        <div class="section-title">{icon} {title}</div>
        <div class="section-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def aqi_pill(category_name):
    color = aqi_color_for_category(category_name)
    st.markdown(
        f"""<span class="aqi-pill" style="background:{color}22; color:{color}; border:1px solid {color}55;">
        <span class="aqi-dot" style="background:{color};"></span>{category_name}</span>""",
        unsafe_allow_html=True,
    )
    return color


def aqi_scale_bar(value=None, vmax=300):
    marker_html = ""
    if value is not None and not pd.isna(value):
        pct = max(0, min(100, (value / vmax) * 100))
        marker_html = f'<div class="aqi-scale-marker" style="left:{pct}%;"></div>'
    st.markdown(
        f"""
        <div class="aqi-scale-bar">{marker_html}</div>
        <div class="aqi-scale-labels">
            <span>Good</span><span>Moderate</span><span>USG</span><span>Unhealthy</span><span>Very Unhealthy</span><span>Hazardous</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_dark_layout(fig, height=420, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C7D0DD", size=12),
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#151c28", font_size=12, font_family="Inter"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


POLLUTANT_LABELS = {
    "pm2_5": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "nitrogen_dioxide": "NO₂ (µg/m³)",
    "sulphur_dioxide": "SO₂ (µg/m³)",
    "carbon_monoxide": "CO (µg/m³)",
    "ozone": "O₃ (µg/m³)",
    "dust": "Dust (µg/m³)",
}


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0 1.2rem 0;">
                <div style="font-size:1.7rem;">🛰️</div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.05rem;line-height:1.1;">Pakistan Air<br>Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pages = {
            "🏠 Overview": "Overview",
            "🤖 AI Prediction": "AI Prediction",
            "🗺️ City Intelligence": "City Intelligence",
            "📊 Air Quality Analytics": "Air Quality Analytics",
            "📈 Trends & Patterns": "Trends & Patterns",
            "🧠 Model Intelligence": "Model Intelligence",
            "🔍 Data Explorer": "Data Explorer",
            "ℹ️ About": "About",
        }
        choice = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8B96A8;line-height:2;">
            <div>🟢 AI Model: <span style="color:#3DD68C;">Online</span></div>
            <div>🟢 Prediction Engine: <span style="color:#3DD68C;">Ready</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div style="font-size:0.7rem;color:#5B6576;margin-top:1rem;">
            Tuned Gradient Boosting · 10 Cities<br>Nov 2025 – Feb 2026
            </div>""",
            unsafe_allow_html=True,
        )
        return pages[choice]


# ============================================================================
# HERO HEADER
# ============================================================================
def render_hero():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow"><span class="dot"></span>LIVE ENVIRONMENTAL INTELLIGENCE</div>
            <div class="hero-title">Pakistan Air Intelligence</div>
            <div class="hero-sub">AI-Powered Air Quality Monitoring &amp; PM2.5 Prediction Platform</div>
            <div class="hero-desc">Explore Pakistan's air-quality patterns, analyze pollution trends, compare cities,
            and generate AI-powered PM2.5 predictions using a trained Gradient Boosting model.</div>
            <div class="status-row">
                <div class="status-chip"><span class="led"></span>AI Model: Online</div>
                <div class="status-chip"><span class="led"></span>Prediction Engine: Ready</div>
                <div class="status-chip">📡 10 Cities Monitored</div>
                <div class="status-chip">🕒 Nov 2025 – Feb 2026</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
def render_overview(df: pd.DataFrame):
    render_hero()

    n_records = len(df)
    n_cities = df["city"].nunique()
    avg_pm25 = df["pm2_5"].mean()
    max_pm25 = df["pm2_5"].max()
    avg_pm10 = df["pm10"].mean()
    date_min, date_max = df["date"].min().date(), df["date"].max().date()

    section_header("📡", "Dataset Overview", "Key indicators computed directly from the live dataset.")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("📄", f"{n_records:,}", "Total Records")
    with c2: kpi_card("🏙️", n_cities, "Cities Monitored")
    with c3: kpi_card("🌫️", f"{avg_pm25:.1f}", "Avg PM2.5", "µg/m³")
    with c4: kpi_card("🔺", f"{max_pm25:.1f}", "Max PM2.5", "µg/m³")
    with c5: kpi_card("💨", f"{avg_pm10:.1f}", "Avg PM10", "µg/m³")
    with c6: kpi_card("📅", f"{(date_max-date_min).days}d", "Date Range", f"{date_min} → {date_max}")

    st.write("")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        section_header("🧭", "AQI Category Distribution", "Share of hourly readings in each air-quality category.")
        cat_counts = df["aqi_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        color_map = {row["category"]: aqi_color_for_category(row["category"]) for _, row in cat_counts.iterrows()}
        fig = px.bar(
            cat_counts, x="count", y="category", orientation="h",
            color="category", color_discrete_map=color_map, text="count",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", showlegend=False)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
        st.plotly_chart(plotly_dark_layout(fig, height=380), use_container_width=True)

    with col2:
        section_header("🏆", "Pollution Ranking", "Cities ranked by average PM2.5.")
        rank = df.groupby("city")["pm2_5"].mean().sort_values(ascending=False).reset_index()
        rank["rank"] = range(1, len(rank) + 1)
        for _, row in rank.iterrows():
            band = classify_pm25(row["pm2_5"])
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:0.55rem 0.2rem;border-bottom:1px solid rgba(255,255,255,0.06);">
                    <div style="display:flex;align-items:center;gap:0.6rem;">
                        <span style="font-family:'JetBrains Mono',monospace;color:#5B6576;font-size:0.8rem;">#{row['rank']:02d}</span>
                        <span style="font-weight:600;">{row['city']}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{band['color']};">{row['pm2_5']:.1f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    section_header("📏", "PM2.5 Interpretation Scale", "How predicted / measured PM2.5 values map to health categories.")
    aqi_scale_bar()
    cols = st.columns(len(AQI_SCALE))
    for c, band in zip(cols, AQI_SCALE):
        with c:
            st.markdown(
                f"""<div style="text-align:center;">
                <div style="width:14px;height:14px;border-radius:50%;background:{band['color']};margin:0 auto 0.4rem auto;"></div>
                <div style="font-size:0.75rem;font-weight:600;">{band['name']}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ============================================================================
# PAGE: AI PREDICTION
# ============================================================================
def render_prediction(df: pd.DataFrame, df_eng: pd.DataFrame, model, preprocessor):
    section_header("🤖", "AI PM2.5 Prediction", "Forecast next-hour PM2.5 using the trained Tuned Gradient Boosting model.")

    num_cols, cat_cols = get_pipeline_columns(preprocessor)
    required_cols = num_cols + cat_cols

    valid = df_eng.dropna(subset=required_cols).copy()
    if valid.empty:
        st.error("No rows with complete lag/rolling history are available for prediction.")
        return

    st.markdown(
        """<div class="glass-card" style="margin-bottom:1.2rem;">
        <b>How this works:</b> pick a city and a reference hour from real historical data. The model needs up to
        24 hours of lag &amp; rolling context (previous PM2.5, PM10, weather, etc.) — that context is pulled
        automatically from the dataset for the hour you choose. You can then override the current-hour readings
        to run an AI-generated "what-if" prediction for the <b>next hour's PM2.5</b>.
        </div>""",
        unsafe_allow_html=True,
    )

    cities = sorted(valid["city"].unique())
    c1, c2 = st.columns([1, 2])
    with c1:
        city = st.selectbox("City", cities, key="pred_city")

    city_rows = valid[valid["city"] == city].sort_values("timestamp")
    ts_options = city_rows["timestamp"].dt.strftime("%Y-%m-%d %H:00").tolist()
    with c2:
        ts_choice = st.selectbox("Reference date & hour", ts_options, index=len(ts_options) - 1, key="pred_ts")

    ref_row = city_rows[city_rows["timestamp"].dt.strftime("%Y-%m-%d %H:00") == ts_choice].iloc[0]

    st.markdown("##### Current-hour readings (editable)")
    st.caption("Defaults are the actual measured values for this hour. Adjust them to explore an AI what-if scenario.")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        in_pm25 = st.number_input("PM2.5 (µg/m³)", value=float(ref_row["pm2_5"]), min_value=0.0, step=0.5)
        in_pm10 = st.number_input("PM10 (µg/m³)", value=float(ref_row["pm10"]), min_value=0.0, step=0.5)
    with f2:
        in_co = st.number_input("Carbon Monoxide", value=float(ref_row["carbon_monoxide"]), min_value=0.0, step=1.0)
        in_no2 = st.number_input("Nitrogen Dioxide", value=float(ref_row["nitrogen_dioxide"]), min_value=0.0, step=0.5)
    with f3:
        in_so2 = st.number_input("Sulphur Dioxide", value=float(ref_row["sulphur_dioxide"]), min_value=0.0, step=0.5)
        in_o3 = st.number_input("Ozone", value=float(ref_row["ozone"]), min_value=0.0, step=0.5)
    with f4:
        in_dust = st.number_input("Dust", value=float(ref_row["dust"]), min_value=0.0, step=0.5)
        in_temp = st.number_input("Temperature (°C)", value=float(ref_row["temperature"]), step=0.5)

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        in_humidity = st.number_input("Humidity (%)", value=float(ref_row["humidity"]), min_value=0.0, max_value=100.0, step=1.0)
    with w2:
        in_wind = st.number_input("Wind Speed", value=float(ref_row["wind_speed"]), min_value=0.0, step=0.5)
    with w3:
        in_pressure = st.number_input("Pressure (hPa)", value=float(ref_row["pressure"]), step=0.5)
    with w4:
        in_precip = st.number_input("Precipitation", value=float(ref_row["precipitation"]), min_value=0.0, step=0.1)

    st.write("")
    predict_clicked = st.button("🔮 Predict PM2.5", use_container_width=True)

    with st.expander("📉 Last 24 hours of PM2.5 for this city (model context)"):
        hist = city_rows[city_rows["timestamp"] <= ref_row["timestamp"]].tail(24)
        fig = px.line(hist, x="timestamp", y="pm2_5", markers=True)
        fig.update_traces(line_color="#35D0BA")
        st.plotly_chart(plotly_dark_layout(fig, height=260), use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing environmental conditions..."):
            row = ref_row.copy()
            row["pm2_5"] = in_pm25
            row["pm10"] = in_pm10
            row["carbon_monoxide"] = in_co
            row["nitrogen_dioxide"] = in_no2
            row["sulphur_dioxide"] = in_so2
            row["ozone"] = in_o3
            row["dust"] = in_dust
            row["temperature"] = in_temp
            row["humidity"] = in_humidity
            row["wind_speed"] = in_wind
            row["pressure"] = in_pressure
            row["precipitation"] = in_precip

            # Recompute the PM2.5 trend features against the (possibly overridden) current reading.
            row["pm2_5_change_1h"] = row["pm2_5"] - row["pm2_5_lag_1h"]
            row["pm2_5_change_3h"] = row["pm2_5"] - row["pm2_5_lag_3h"]
            row["pm2_5_change_6h"] = row["pm2_5"] - row["pm2_5_lag_6h"]
            row["pm2_5_change_24h"] = row["pm2_5"] - row["pm2_5_lag_24h"]

            try:
                X_row = pd.DataFrame([row[required_cols]])
                X_transformed = preprocessor.transform(X_row)
                prediction = float(model.predict(X_transformed)[0])
                prediction = max(0.0, prediction)
            except Exception:
                st.error("The prediction could not be completed because the input did not match the model's "
                          "expected schema. Please try a different reference hour.")
                return

        band = classify_pm25(prediction)
        st.write("")
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(
                f"""
                <div class="predict-result">
                    <div class="predict-label">Predicted PM2.5 · Next Hour</div>
                    <div class="predict-value" style="color:{band['color']};">{prediction:.1f}<span class="predict-unit">µg/m³</span></div>
                    <div style="margin-top:0.9rem;">
                        <span class="aqi-pill" style="background:{band['color']}22;color:{band['color']};border:1px solid {band['color']}55;">
                        <span class="aqi-dot" style="background:{band['color']};"></span>{band['name']}</span>
                    </div>
                    <div style="margin-top:1rem;color:#8B96A8;font-size:0.85rem;max-width:420px;margin-left:auto;margin-right:auto;">{band['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("⚠️ This prediction is an AI-generated estimate based on historical environmental patterns "
                       "and the supplied input conditions. It is not a certified air-quality reading.")

        with r2:
            st.markdown("##### Model context for this prediction")
            actual_next = ref_row.get("target_pm2_5", np.nan)
            if not pd.isna(actual_next):
                st.metric("Actual next-hour PM2.5 (historical ground truth)", f"{actual_next:.1f} µg/m³",
                          delta=f"{prediction - actual_next:+.1f} vs. prediction")
            st.metric("Rolling 24h mean PM2.5 used as context", f"{ref_row['pm2_5_rolling_mean_24h']:.1f} µg/m³")
            st.metric("1h PM2.5 change (input)", f"{row['pm2_5_change_1h']:+.1f} µg/m³")
            aqi_scale_bar(value=prediction)

            result_export = pd.DataFrame([{
                "city": city, "reference_hour": ts_choice, "predicted_pm2_5": round(prediction, 2),
                "pollution_level": band["name"], "input_pm2_5": in_pm25, "input_pm10": in_pm10,
            }])
            st.download_button("⬇️ Download Prediction Result (CSV)", result_export.to_csv(index=False),
                                file_name="pm25_prediction_result.csv", mime="text/csv")


# ============================================================================
# PAGE: CITY INTELLIGENCE
# ============================================================================
def render_city_intelligence(df: pd.DataFrame):
    section_header("🗺️", "City Intelligence", "Deep-dive into any monitored city, or compare pollutants across all ten.")

    cities = sorted(df["city"].unique())
    city = st.selectbox("Select a city", cities)
    cdf = df[df["city"] == city]

    band = classify_pm25(cdf["pm2_5"].mean())
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("🌫️", f"{cdf['pm2_5'].mean():.1f}", "Avg PM2.5", "µg/m³")
    with c2: kpi_card("🔺", f"{cdf['pm2_5'].max():.1f}", "Max PM2.5", "µg/m³")
    with c3: kpi_card("💨", f"{cdf['pm10'].mean():.1f}", "Avg PM10", "µg/m³")
    with c4: kpi_card("🌡️", f"{cdf['temperature'].mean():.1f}°C", "Avg Temperature")
    with c5: kpi_card("💧", f"{cdf['humidity'].mean():.1f}%", "Avg Humidity")

    c6, c7 = st.columns(2)
    with c6: kpi_card("📄", f"{len(cdf):,}", "Records for this city")
    with c7:
        top_cat = cdf["aqi_category"].mode().iloc[0]
        kpi_card("🏷️", top_cat, "Most Common AQI Category")

    st.write("")
    section_header("📊", "Pollution Ranking — Compare Cities", "Switch pollutants to re-rank all cities.")
    pollutant = st.selectbox("Pollutant", list(POLLUTANT_LABELS.keys()), format_func=lambda x: POLLUTANT_LABELS[x])
    rank = df.groupby("city")[pollutant].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(rank, x=pollutant, y="city", orientation="h", color=pollutant,
                 color_continuous_scale=["#3DD68C", "#E8C34A", "#F0954A", "#EA6262", "#8B3A3A"])
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
    st.plotly_chart(plotly_dark_layout(fig, height=420, title=f"Average {POLLUTANT_LABELS[pollutant]} by City"),
                     use_container_width=True)


# ============================================================================
# PAGE: AIR QUALITY ANALYTICS
# ============================================================================
def render_analytics(df: pd.DataFrame):
    section_header("📊", "Air Quality Analytics", "Historical trends, pollutant comparisons, and weather relationships.")

    cities = sorted(df["city"].unique())
    c1, c2 = st.columns([1, 2])
    with c1:
        sel_cities = st.multiselect("Cities", cities, default=[cities[0]])
    with c2:
        min_d, max_d = df["date"].min().date(), df["date"].max().date()
        date_range = st.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d))

    mask = (df["city"].isin(sel_cities)) & (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    fdf = df[mask]

    if fdf.empty or not sel_cities:
        st.info("Select at least one city to view analytics.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["PM2.5 Trend", "PM10 Trend", "Pollutant Comparison", "Weather vs Pollution"])

    with tab1:
        fig = px.line(fdf, x="timestamp", y="pm2_5", color="city")
        st.plotly_chart(plotly_dark_layout(fig, height=420, title="PM2.5 Over Time"), use_container_width=True)

    with tab2:
        fig = px.line(fdf, x="timestamp", y="pm10", color="city")
        st.plotly_chart(plotly_dark_layout(fig, height=420, title="PM10 Over Time"), use_container_width=True)

    with tab3:
        selected_pollutants = st.multiselect("Pollutants to compare", list(POLLUTANT_LABELS.keys()),
                                              default=["pm2_5", "pm10"], format_func=lambda x: POLLUTANT_LABELS[x])
        if selected_pollutants:
            daily = fdf.groupby([fdf["date"].dt.date, "city"])[selected_pollutants].mean().reset_index()
            for pol in selected_pollutants:
                fig = px.line(daily, x="date", y=pol, color="city", title=POLLUTANT_LABELS[pol])
                st.plotly_chart(plotly_dark_layout(fig, height=320), use_container_width=True)

    with tab4:
        weather_var = st.selectbox("Weather variable", ["temperature", "humidity", "wind_speed", "pressure"],
                                    format_func=lambda x: x.replace("_", " ").title())
        fig = px.scatter(fdf.sample(min(3000, len(fdf)), random_state=1), x=weather_var, y="pm2_5",
                          color="city", opacity=0.55, trendline=None)
        st.plotly_chart(plotly_dark_layout(fig, height=440,
                         title=f"{weather_var.replace('_',' ').title()} vs PM2.5"), use_container_width=True)

        st.markdown("###### Correlation heatmap — weather vs PM2.5")
        weather_cols = ["temperature", "humidity", "wind_speed", "pressure", "precipitation", "pm2_5"]
        corr = fdf[weather_cols].corr()
        fig2 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                          colorscale=[[0, "#8B3A3A"], [0.5, "#111826"], [1, "#35D0BA"]],
                                          zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}"))
        st.plotly_chart(plotly_dark_layout(fig2, height=380), use_container_width=True)


# ============================================================================
# PAGE: TRENDS & PATTERNS
# ============================================================================
def render_trends(df: pd.DataFrame):
    section_header("📈", "Trends & Patterns", "Time-based pollution patterns across hour, weekday, month, and season.")

    t1, t2, t3, t4, t5 = st.tabs(["Hourly", "Day of Week", "Monthly", "Seasonal", "City × Hour Heatmap"])

    with t1:
        hourly = df.groupby("hour")["pm2_5"].mean().reset_index()
        fig = px.bar(hourly, x="hour", y="pm2_5", color="pm2_5",
                     color_continuous_scale=["#3DD68C", "#E8C34A", "#F0954A", "#EA6262"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_dark_layout(fig, height=400, title="Average PM2.5 by Hour of Day"), use_container_width=True)

    with t2:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = df.groupby("day_of_week")["pm2_5"].mean().reindex(dow_order).reset_index()
        fig = px.bar(dow, x="day_of_week", y="pm2_5", color="pm2_5",
                     color_continuous_scale=["#3DD68C", "#E8C34A", "#F0954A", "#EA6262"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_dark_layout(fig, height=400, title="Average PM2.5 by Day of Week"), use_container_width=True)

    with t3:
        month_order = df.sort_values("month")["month_name"].unique().tolist()
        month_avg = df.groupby("month_name")["pm2_5"].mean().reindex(month_order).reset_index()
        fig = px.line(month_avg, x="month_name", y="pm2_5", markers=True)
        fig.update_traces(line_color="#F2A93B", line_width=3, marker_size=9)
        st.plotly_chart(plotly_dark_layout(fig, height=400, title="Average PM2.5 by Month"), use_container_width=True)

    with t4:
        seasons = sorted(df["season"].unique())
        season_avg = df.groupby("season")["pm2_5"].mean().reindex(seasons).reset_index()
        fig = px.bar(season_avg, x="season", y="pm2_5", color="season",
                     color_discrete_sequence=["#35D0BA", "#F2A93B", "#EA6262", "#B57BE0"])
        st.plotly_chart(plotly_dark_layout(fig, height=400, title="Average PM2.5 by Season"), use_container_width=True)

    with t5:
        pivot = df.pivot_table(index="city", columns="hour", values="pm2_5", aggfunc="mean")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#3DD68C"], [0.3, "#E8C34A"], [0.6, "#F0954A"], [1, "#8B3A3A"]],
        ))
        st.plotly_chart(plotly_dark_layout(fig, height=460, title="Average PM2.5 — City × Hour"), use_container_width=True)


# ============================================================================
# PAGE: MODEL INTELLIGENCE
# ============================================================================
def render_model_intelligence(comparison_df: pd.DataFrame, importance_df: pd.DataFrame):
    section_header("🧠", "Model Intelligence", "How the trained models perform, and which features drive predictions.")

    best_row = comparison_df.loc[comparison_df["R2 Score"].idxmax()]
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("🏆", best_row["Model"], "Best Model", f"R² = {best_row['R2 Score']:.4f}")
    with c2: kpi_card("📉", f"{best_row['MAE']:.2f}", "MAE (best model)", "µg/m³")
    with c3: kpi_card("📐", f"{best_row['RMSE']:.2f}", "RMSE (best model)", "µg/m³")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        section_header("📊", "Model Comparison", "")
        fig = px.bar(comparison_df, x="Model", y="R2 Score", color="Model",
                     color_discrete_sequence=["#35D0BA", "#F2A93B", "#EA6262"], text="R2 Score")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", showlegend=False)
        st.plotly_chart(plotly_dark_layout(fig, height=380, title="R² Score by Model"), use_container_width=True)
    with col2:
        section_header("📋", "Full Metrics Table", "")
        st.dataframe(comparison_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2 Score": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Model Comparison", comparison_df.to_csv(index=False),
                            file_name="model_comparison.csv", mime="text/csv")

    st.write("")
    section_header("🔬", "Feature Importance", "Which inputs most strongly influence the Gradient Boosting model's predictions.")
    top_n = st.select_slider("Show top", options=[5, 10, 15, 20], value=10)
    top_features = importance_df.sort_values("Importance", ascending=False).head(top_n).sort_values("Importance")
    fig = px.bar(top_features, x="Importance", y="Feature_Clean", orientation="h",
                 color="Importance", color_continuous_scale=["#111826", "#35D0BA", "#F2A93B"])
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(plotly_dark_layout(fig, height=max(360, top_n * 28)), use_container_width=True)
    st.caption("Feature importance indicates which input variables contributed most strongly to the model's "
               "predictions. It should not be interpreted as causal evidence.")


# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
def render_data_explorer(df: pd.DataFrame):
    section_header("🔍", "Data Explorer", "Filter, search, and export the underlying dataset.")

    with st.expander("🎛️ Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            cities = st.multiselect("City", sorted(df["city"].unique()))
        with f2:
            categories = st.multiselect("AQI Category", sorted(df["aqi_category"].unique()))
        with f3:
            seasons = st.multiselect("Season", sorted(df["season"].unique()))

        f4, f5 = st.columns(2)
        with f4:
            min_d, max_d = df["date"].min().date(), df["date"].max().date()
            date_range = st.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d), key="explorer_date")
        with f5:
            hour_range = st.slider("Hour range", 0, 23, (0, 23))

    fdf = df.copy()
    if cities:
        fdf = fdf[fdf["city"].isin(cities)]
    if categories:
        fdf = fdf[fdf["aqi_category"].isin(categories)]
    if seasons:
        fdf = fdf[fdf["season"].isin(seasons)]
    fdf = fdf[(fdf["date"].dt.date >= date_range[0]) & (fdf["date"].dt.date <= date_range[1])]
    fdf = fdf[(fdf["hour"] >= hour_range[0]) & (fdf["hour"] <= hour_range[1])]

    st.markdown(f"**{len(fdf):,}** records match your filters (of {len(df):,} total).")
    st.dataframe(fdf, use_container_width=True, height=420)
    st.download_button("⬇️ Download Filtered Data (CSV)", fdf.to_csv(index=False),
                        file_name="pakistan_air_quality_filtered.csv", mime="text/csv")

    st.write("")
    section_header("📐", "Dataset Statistics", "Computed dynamically from the filtered dataset.")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1: kpi_card("📄", f"{len(fdf):,}", "Records")
    with s2: kpi_card("🏙️", fdf["city"].nunique(), "Cities")
    with s3: kpi_card("❓", f"{fdf.isnull().sum().sum():,}", "Missing Values")
    with s4: kpi_card("👥", f"{fdf.duplicated().sum():,}", "Duplicate Rows")
    with s5: kpi_card("🔢", fdf.select_dtypes(include=np.number).shape[1], "Numeric Features")
    with s6: kpi_card("🏷️", fdf.select_dtypes(include="object").shape[1], "Categorical Features")


# ============================================================================
# PAGE: ABOUT
# ============================================================================
def render_about(dq_text: str):
    section_header("ℹ️", "About This Project", "Dataset, model, and data-quality background.")

    st.markdown(
        """
        <div class="glass-card">
        <p style="color:#C7D0DD;">Pakistan Air Intelligence is an AI-powered environmental analytics platform built on
        90 days of hourly air-quality and weather observations across 10 major Pakistani cities
        (Nov 6, 2025 – Feb 4, 2026). A tuned Gradient Boosting Regressor — benchmarked against GRU and LSTM
        deep-learning baselines — powers the next-hour PM2.5 prediction engine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    section_header("📋", "Data Quality Report", "Summary of the correction applied in dataset version v2.")

    lines = dq_text.split("\n")
    summary_lines, in_summary = [], False
    for line in lines:
        if line.strip().startswith("## Summary"):
            in_summary = True
            continue
        if in_summary and line.strip().startswith("##"):
            break
        if in_summary and line.strip():
            summary_lines.append(line.strip())
    summary_text = " ".join(summary_lines)

    dqc1, dqc2 = st.columns([1.4, 1])
    with dqc1:
        st.markdown(
            f"""<div class="glass-card">
            <b>Dataset version:</b> v2 (corrected)<br>
            <b>Summary:</b> {summary_text}
            </div>""",
            unsafe_allow_html=True,
        )
    with dqc2:
        kpi_card("✅", "5,040", "Rows Corrected", "Weather columns, Nov 6–26 2025")
        kpi_card("✅", "0", "Missing Values Remaining")

    with st.expander("📄 View full Data Quality Report"):
        st.markdown(dq_text)

    st.write("")
    section_header("📁", "Project Assets", "Everything this dashboard is built on.")
    st.markdown(
        """
        <div class="glass-card">
        <ul style="color:#8B96A8;line-height:2;">
        <li><code>assets/models/tuned_gradient_boosting_pm25.joblib</code> — trained prediction model</li>
        <li><code>assets/models/pm25_preprocessor.joblib</code> — fitted ColumnTransformer (scaling + one-hot encoding)</li>
        <li><code>assets/models/feature_importance.csv</code> — feature importance from the trained model</li>
        <li><code>assets/models/final_model_comparison.csv</code> — Gradient Boosting vs GRU vs LSTM benchmark</li>
        <li><code>assets/dataset/pakistan_air_quality_final_clean_v2.csv</code> — hourly dataset, 10 cities</li>
        <li><code>assets/pakistan-air-intelligence.ipynb</code> — full EDA, feature engineering & training notebook</li>
        <li><code>assets/Pakistan_Air_Intelligence_SRS.pdf</code> — software requirements specification</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# FOOTER
# ============================================================================
def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#E9EDF4;font-size:1rem;">Pakistan Air Intelligence</div>
            <div style="margin-top:0.2rem;">AI-Powered Environmental Analytics</div>
            <div class="footer-tags">
                <span class="footer-tag">Machine Learning</span>
                <span class="footer-tag">Data Analytics</span>
                <span class="footer-tag">PM2.5 Prediction</span>
                <span class="footer-tag">Pakistan Air Quality</span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#5B6576;">Built with Python + Streamlit + Machine Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    render_css()

    # ---- Safe asset loading with friendly error handling ----
    missing = [p for p in [DATASET_PATH, MODEL_PATH, PREPROCESSOR_PATH, FEATURE_IMPORTANCE_PATH,
                            MODEL_COMPARISON_PATH, DQ_REPORT_PATH] if not p.exists()]
    if missing:
        st.error("⚠️ Some required project files are missing:\n\n" +
                  "\n".join(f"- `{p.relative_to(BASE_DIR)}`" for p in missing) +
                  "\n\nPlease make sure the `assets/` folder is present next to `app.py`.")
        st.stop()

    try:
        with st.spinner("Initializing AI Prediction Engine..."):
            df = load_data()
            model = load_model()
            preprocessor = load_preprocessor()
            importance_df = load_feature_importance()
            comparison_df = load_model_comparison()
            dq_text = load_dq_report_text()
            df_eng = engineer_features(df)
    except Exception as e:
        st.error("⚠️ Something went wrong while loading the project's data or ML artifacts. "
                  "Please verify the files inside `assets/` are not corrupted.")
        st.stop()
        return

    page = render_sidebar()

    if page == "Overview":
        render_overview(df)
    elif page == "AI Prediction":
        render_prediction(df, df_eng, model, preprocessor)
    elif page == "City Intelligence":
        render_city_intelligence(df)
    elif page == "Air Quality Analytics":
        render_analytics(df)
    elif page == "Trends & Patterns":
        render_trends(df)
    elif page == "Model Intelligence":
        render_model_intelligence(comparison_df, importance_df)
    elif page == "Data Explorer":
        render_data_explorer(df)
    elif page == "About":
        render_about(dq_text)

    render_footer()


if __name__ == "__main__":
    main()
