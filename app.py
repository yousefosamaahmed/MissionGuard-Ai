from __future__ import annotations

import base64
import html
import inspect
import json
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from database.connection import (
    check_database_connection,
    database_is_enabled,
    database_status_message,
)
from database.repositories.analysis import (
    get_analysis_run,
    get_latest_completed_analysis_run,
    get_prediction_risk_summary,
    list_analysis_prediction_records,
    list_incidents,
    update_analysis_run_mission_health,
    update_incident_review,
)
from database.repositories.models import (
    list_model_versions,
)
from database.repositories.telemetry_sessions import (
    get_telemetry_session,
    list_telemetry_sessions,
)
from database.services.opssat_inference_service import (
    persist_uploaded_opssat_analysis,
    run_real_opssat_analysis,
)
from src.mission_health import calculate_mission_health_score
from src.opssat import (
    CHANNEL_NAMES,
    assess_data_drift,
    detect_and_prepare_upload,
    evaluate_binary_predictions,
    evaluate_event_detection,
    load_artifact,
    predict_feature_rows,
    validate_features_against_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "opssat" / "raw" / "dataset.csv"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "opssat" / "raw" / "segments.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "opssat_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "opssat_metrics.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "opssat" / "processed" / "official_test_predictions.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "opssat_metadata.json"
SAMPLES_DIR = PROJECT_ROOT / "data" / "opssat" / "upload_samples"
ASSETS_DIR = PROJECT_ROOT / "assets"
TEAM_ASSETS_DIR = ASSETS_DIR / "team"
YOUSSEF_PHOTO_PATH = TEAM_ASSETS_DIR / "youssef-osama-soliman.png"
SHEREEN_PHOTO_PATH = TEAM_ASSETS_DIR / "shereen-ahmed-hazem.jpeg"

st.set_page_config(
    page_title="MissionGuard AI — Spacecraft Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("MISSIONGUARD")
st.sidebar.caption("SPACECRAFT INTELLIGENCE PLATFORM / OPS-SAT")

appearance_mode = st.sidebar.radio(
    "Appearance",
    ["Dark", "Light"],
    horizontal=True,
    key="missionguard_appearance_mode",
    help=(
        "Switch the full MissionGuard interface, charts, "
        "code blocks, and Streamlit connection dialog "
        "between Dark and Light themes."
    ),
)

if appearance_mode == "Dark":
    theme = {
        "background": "#050505",
        "sidebar": "#0a0a0a",
        "surface": "#101010",
        "surface_alt": "#181818",
        "text": "#f7f6f1",
        "muted": "#aaa9a3",
        "border": "#30302d",
        "primary": "#f5f4ef",
        "primary_hover": "#ffffff",
        "success": "#8ee8b5",
        "warning": "#ffd27a",
        "danger": "#ff7a7a",
        "info": "#9acbff",
        "grid": "rgba(247,246,241,0.12)",
        "hero_start": "#f4f3ee",
        "hero_end": "#080808",
        "code_background": "#090909",
        "code_text": "#f7f6f1",
        "modal_background": "#0d0d0d",
        "modal_text": "#f7f6f1",
        "modal_code_background": "#070707",
        "modal_code_text": "#f7f6f1",
        "modal_border": "#333330",
        "modal_shadow": "0 26px 80px rgba(0,0,0,0.72)",
    }
else:
    theme = {
        "background": "#efeee8",
        "sidebar": "#f7f6f1",
        "surface": "#ffffff",
        "surface_alt": "#f0efe9",
        "text": "#090909",
        "muted": "#5e5d58",
        "border": "#cecdc5",
        "primary": "#0a0a0a",
        "primary_hover": "#252525",
        "success": "#187a4a",
        "warning": "#9b5a00",
        "danger": "#b32d3e",
        "info": "#195f91",
        "grid": "rgba(9,9,9,0.10)",
        "hero_start": "#f7f6f1",
        "hero_end": "#090909",
        "code_background": "#141414",
        "code_text": "#f7f6f1",
        "modal_background": "#ffffff",
        "modal_text": "#090909",
        "modal_code_background": "#141414",
        "modal_code_text": "#f7f6f1",
        "modal_border": "#cecdc5",
        "modal_shadow": "0 26px 80px rgba(9,9,9,0.22)",
    }

st.markdown(
    f"""
<style>
:root {{
  color-scheme: {appearance_mode.lower()};
  --mg-bg: {theme['background']};
  --mg-sidebar: {theme['sidebar']};
  --mg-surface: {theme['surface']};
  --mg-surface-alt: {theme['surface_alt']};
  --mg-text: {theme['text']};
  --mg-muted: {theme['muted']};
  --mg-border: {theme['border']};
  --mg-primary: {theme['primary']};
  --mg-code-bg: {theme['code_background']};
  --mg-code-text: {theme['code_text']};
}}

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
  background: var(--mg-bg) !important;
  color: var(--mg-text) !important;
}}

[data-testid="stHeader"] {{
  background: color-mix(in srgb, var(--mg-bg) 92%, transparent) !important;
}}

[data-testid="stToolbar"],
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] svg {{
  color: var(--mg-text) !important;
  fill: var(--mg-text) !important;
}}

[data-testid="stSidebar"] {{
  background: var(--mg-sidebar) !important;
  border-right: 1px solid var(--mg-border) !important;
}}

[data-testid="stSidebar"] * {{
  color: var(--mg-text);
}}

/*
Only theme the actual application and sidebar text.
Do not apply a global div/span rule because Streamlit
mounts internal system dialogs outside the main app tree.
*/
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] li,
[data-testid="stMain"] th,
[data-testid="stMain"] td,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li {{
  color: var(--mg-text);
}}

[data-testid="stCaptionContainer"],
.stCaption,
small {{
  color: var(--mg-muted) !important;
}}

[data-testid="stMetric"] {{
  background: var(--mg-surface) !important;
  border: 1px solid var(--mg-border) !important;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 5px 18px rgba(15,23,42,0.08);
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {{
  color: var(--mg-muted) !important;
}}

[data-testid="stMetricValue"] {{
  color: var(--mg-text) !important;
}}

.stButton > button,
.stDownloadButton > button {{
  background: var(--mg-primary) !important;
  color: #ffffff !important;
  border: 1px solid var(--mg-primary) !important;
  border-radius: 9px;
  font-weight: 700;
}}

.stButton > button *,
.stDownloadButton > button * {{
  color: #ffffff !important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
  filter: brightness(1.08);
  border-color: var(--mg-primary) !important;
}}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-testid="stFileUploaderDropzone"] {{
  background: var(--mg-surface) !important;
  color: var(--mg-text) !important;
  border-color: var(--mg-border) !important;
}}

[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="textarea"] *,
[data-testid="stFileUploaderDropzone"] * {{
  color: var(--mg-text) !important;
}}

[data-baseweb="popover"],
[role="listbox"],
[role="option"] {{
  background: var(--mg-surface) !important;
  color: var(--mg-text) !important;
}}

[data-baseweb="popover"] *,
[role="listbox"] *,
[role="option"] * {{
  color: var(--mg-text) !important;
}}

[data-testid="stExpander"] {{
  background: var(--mg-surface) !important;
  border: 1px solid var(--mg-border) !important;
  border-radius: 12px;
  overflow: hidden;
}}

/* Keep expander title, arrow, and body readable in both themes. */
[data-testid="stExpander"] details {{
  background: var(--mg-surface) !important;
  color: var(--mg-text) !important;
}}

[data-testid="stExpander"] summary {{
  background: var(--mg-surface-alt) !important;
  color: var(--mg-text) !important;
  border-bottom: 1px solid var(--mg-border) !important;
}}

[data-testid="stExpander"] summary *,
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary path {{
  color: var(--mg-text) !important;
  fill: var(--mg-text) !important;
  stroke: var(--mg-text) !important;
}}

[data-testid="stExpanderDetails"],
[data-testid="stExpander"] details > div {{
  background: var(--mg-surface) !important;
  color: var(--mg-text) !important;
}}

[data-testid="stDataFrame"] {{
  background: var(--mg-surface) !important;
  border: 1px solid var(--mg-border) !important;
  border-radius: 12px;
  overflow: hidden;
}}

[data-testid="stAlert"] {{
  border: 1px solid var(--mg-border) !important;
}}

[data-testid="stAlert"] * {{
  color: var(--mg-text) !important;
}}

code,
pre,
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code {{
  background: var(--mg-code-bg) !important;
  color: var(--mg-code-text) !important;
}}

code *,
pre *,
[data-testid="stCodeBlock"] *,
[data-testid="stCodeBlock"] code span {{
  color: var(--mg-code-text) !important;
}}

[data-testid="stCodeBlock"] pre {{
  white-space: pre !important;
  overflow-x: auto !important;
  tab-size: 2;
  line-height: 1.55;
}}

.mg-json-block {{
  display: block;
  background: var(--mg-code-bg) !important;
  color: var(--mg-code-text) !important;
  border: 1px solid var(--mg-border);
  border-radius: 10px;
  padding: 16px 18px;
  margin: 0;
  max-height: 680px;
  overflow: auto;
  white-space: pre;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.88rem;
  line-height: 1.5;
}}

.hero {{
  padding: 26px 30px;
  border-radius: 22px;
  background: linear-gradient(120deg, {theme['hero_start']}, {theme['hero_end']});
  margin-bottom: 20px;
  box-shadow: 0 12px 30px rgba(4,12,28,0.16);
}}

.hero h1,
.hero p {{
  color: #ffffff !important;
}}

.hero p {{
  max-width: 980px;
  color: #e8f2ff !important;
}}

.badge {{
  display: inline-block;
  padding: 5px 10px;
  margin-right: 6px;
  border-radius: 999px;
  background: #dbeafe;
  color: #102a50 !important;
  font-weight: 800;
  font-size: 0.78rem;
}}

.card {{
  background: var(--mg-surface) !important;
  color: var(--mg-text) !important;
  border: 1px solid var(--mg-border);
  border-radius: 14px;
  padding: 16px 18px;
  margin: 8px 0;
}}

.card * {{
  color: var(--mg-text) !important;
}}

[data-testid="stPlotlyChart"] {{
  background: var(--mg-surface) !important;
  border: 1px solid var(--mg-border) !important;
  border-radius: 14px;
  padding: 8px 10px 2px 10px;
  margin: 10px 0 26px 0;
  overflow: hidden;
}}

[data-testid="stPlotlyChart"] > div {{
  width: 100% !important;
}}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {{
  margin-top: 0.55rem !important;
  margin-bottom: 0.85rem !important;
}}

[data-testid="stMainBlockContainer"] {{
  padding-bottom: 3rem;
}}

@media (max-width: 900px) {{
  [data-testid="stMetric"] {{
    padding: 12px 13px;
  }}

  [data-testid="stPlotlyChart"] {{
    padding: 4px 4px 0 4px;
    margin-bottom: 22px;
  }}

  .hero {{
    padding: 20px 20px;
    border-radius: 16px;
  }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# Astra-inspired visual layer. This only changes presentation; all
# MissionGuard data, models, workflows, and page logic remain untouched.
astra_button_foreground = "#090909" if appearance_mode == "Dark" else "#ffffff"
astra_inverse_surface = "#f5f4ef" if appearance_mode == "Dark" else "#0a0a0a"
astra_inverse_text = "#090909" if appearance_mode == "Dark" else "#f7f6f1"
astra_star_opacity = "0.34" if appearance_mode == "Dark" else "0.14"

st.markdown(
    f"""
<style>
:root {{
  --astra-ink: {theme['text']};
  --astra-paper: {theme['surface']};
  --astra-inverse: {astra_inverse_surface};
  --astra-inverse-text: {astra_inverse_text};
  --astra-button-fg: {astra_button_foreground};
}}

html {{
  scroll-behavior: smooth;
}}

body,
.stApp {{
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.stApp::before {{
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: {astra_star_opacity};
  background-image:
    radial-gradient(circle at 12% 18%, rgba(255,255,255,.95) 0 1px, transparent 1.5px),
    radial-gradient(circle at 76% 23%, rgba(255,255,255,.85) 0 1px, transparent 1.5px),
    radial-gradient(circle at 42% 72%, rgba(255,255,255,.75) 0 1px, transparent 1.4px),
    radial-gradient(circle at 88% 84%, rgba(255,255,255,.9) 0 1px, transparent 1.4px);
  background-size: 92px 92px, 137px 137px, 173px 173px, 211px 211px;
}}

[data-testid="stMain"],
[data-testid="stSidebar"] {{
  position: relative;
  z-index: 1;
}}

[data-testid="stMainBlockContainer"] {{
  max-width: 1580px;
  padding-top: 1.55rem;
  padding-left: clamp(1rem, 2.2vw, 2.5rem);
  padding-right: clamp(1rem, 2.2vw, 2.5rem);
}}

[data-testid="stSidebar"] {{
  box-shadow: 18px 0 70px rgba(0,0,0,.12);
}}

[data-testid="stSidebar"] > div:first-child {{
  padding-top: 1.35rem;
}}

[data-testid="stSidebar"] h1 {{
  font-size: 1.35rem !important;
  letter-spacing: -0.055em;
  font-weight: 850 !important;
  text-transform: uppercase;
  margin-bottom: 1.15rem !important;
}}

[data-testid="stSidebar"] h1::before {{
  content: "✦";
  display: inline-grid;
  place-items: center;
  width: 2rem;
  height: 2rem;
  margin-right: .55rem;
  border-radius: 50%;
  color: var(--astra-inverse-text);
  background: var(--astra-inverse);
  font-size: .88rem;
  vertical-align: middle;
}}

[data-testid="stSidebar"] div[role="radiogroup"] {{
  gap: .35rem;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label {{
  border: 1px solid transparent;
  border-radius: 999px;
  padding: .38rem .62rem;
  transition: transform .22s ease, background .22s ease, border-color .22s ease;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
  transform: translateX(3px);
  background: var(--mg-surface-alt);
  border-color: var(--mg-border);
}}

[data-testid="stSidebar"] hr {{
  border-color: var(--mg-border) !important;
  opacity: .7;
}}

.hero {{
  position: relative;
  min-height: 430px;
  overflow: hidden;
  padding: clamp(1.4rem, 3.5vw, 3.5rem);
  margin-bottom: 1.15rem;
  border: 1px solid var(--mg-border);
  border-radius: 34px;
  background:
    linear-gradient(104deg, #f4f3ee 0 45%, #0a0a0a 45.2% 100%);
  box-shadow: 0 26px 80px rgba(0,0,0,.24);
  isolation: isolate;
}}

.hero::before {{
  content: "";
  position: absolute;
  inset: 0 0 0 45%;
  z-index: -1;
  opacity: .72;
  background-image:
    radial-gradient(circle at 15% 20%, white 0 1px, transparent 1.5px),
    radial-gradient(circle at 78% 32%, white 0 1px, transparent 1.4px),
    radial-gradient(circle at 42% 66%, white 0 1px, transparent 1.4px),
    radial-gradient(circle at 90% 80%, white 0 1px, transparent 1.5px),
    radial-gradient(ellipse at 55% 52%, rgba(255,255,255,.18), transparent 46%);
  background-size: 78px 78px, 126px 126px, 164px 164px, 218px 218px, 100% 100%;
}}

.hero::after {{
  content: "";
  position: absolute;
  width: 9.5rem;
  height: 9.5rem;
  left: 27%;
  top: 20%;
  border-radius: 50%;
  background: #090909;
  box-shadow: 2.1rem -.75rem 0 -1.7rem #090909;
  z-index: -1;
}}

.hero-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(250px, .46fr);
  gap: 2rem;
  min-height: 330px;
  align-items: end;
}}

.hero-main {{
  align-self: end;
}}

.hero-kicker {{
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  padding: .42rem .72rem;
  border: 1px solid rgba(255,255,255,.38);
  border-radius: 999px;
  color: #ffffff !important;
  background: rgba(0,0,0,.26);
  backdrop-filter: blur(12px);
  font-size: .75rem;
  font-weight: 750;
  letter-spacing: .08em;
  text-transform: uppercase;
}}

.hero-title {{
  margin: 1.5rem 0 .6rem !important;
  color: #ffffff !important;
  font-size: clamp(4.2rem, 10.3vw, 10.4rem) !important;
  line-height: .78 !important;
  letter-spacing: -.085em !important;
  font-weight: 900 !important;
  mix-blend-mode: difference;
  white-space: nowrap;
}}

.hero-copy {{
  display: inline-block;
  max-width: min(48rem, 100%);
  margin: .35rem 0 0 !important;
  padding: .9rem 1.05rem;
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 16px;
  color: #ffffff !important;
  background: rgba(7,7,7,.84);
  box-shadow: 0 14px 34px rgba(0,0,0,.24);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  text-shadow: 0 1px 2px rgba(0,0,0,.72);
  font-size: clamp(.92rem, 1.25vw, 1.08rem);
  font-weight: 560;
  line-height: 1.58;
}}

.hero-aside {{
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-self: stretch;
  padding: .25rem 0 .25rem 1rem;
  color: #ffffff;
}}

.hero-aside-label {{
  color: rgba(255,255,255,.64) !important;
  font-size: .72rem;
  letter-spacing: .14em;
  text-transform: uppercase;
}}

.hero-aside h3 {{
  color: #ffffff !important;
  font-size: clamp(1.3rem, 2vw, 2.2rem) !important;
  line-height: 1.02 !important;
  letter-spacing: -.05em !important;
  max-width: 14rem;
}}

.hero-service-list {{
  display: grid;
  gap: .55rem;
  margin-top: 1rem;
}}

.hero-service-list span {{
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: .48rem;
  border-bottom: 1px solid rgba(255,255,255,.19);
  color: rgba(255,255,255,.88) !important;
  font-size: .82rem;
}}

.hero-service-list b {{
  color: rgba(255,255,255,.46) !important;
  font-weight: 600;
}}

.astra-marquee {{
  display: flex;
  gap: 2.2rem;
  overflow: hidden;
  margin: 0 0 1.55rem;
  padding: .75rem 1rem;
  border: 1px solid var(--mg-border);
  border-radius: 999px;
  background: var(--mg-surface);
  color: var(--mg-text);
  white-space: nowrap;
}}

.astra-marquee-track {{
  display: flex;
  gap: 2.2rem;
  min-width: max-content;
  animation: astra-marquee 28s linear infinite;
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .12em;
  text-transform: uppercase;
}}

.astra-marquee-track span::after {{
  content: "✦";
  margin-left: 2.2rem;
}}

@keyframes astra-marquee {{
  from {{ transform: translateX(0); }}
  to {{ transform: translateX(-50%); }}
}}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {{
  letter-spacing: -.045em !important;
  font-weight: 820 !important;
}}

[data-testid="stMain"] h1 {{
  font-size: clamp(2.15rem, 4vw, 4.25rem) !important;
}}

[data-testid="stMain"] h2 {{
  font-size: clamp(1.55rem, 2.6vw, 2.8rem) !important;
}}

[data-testid="stMetric"] {{
  min-height: 118px;
  border-radius: 22px !important;
  padding: 1.1rem 1.15rem !important;
  box-shadow: none !important;
  transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease;
}}

[data-testid="stMetric"]:hover {{
  transform: translateY(-4px);
  border-color: var(--mg-text) !important;
  box-shadow: 0 18px 40px rgba(0,0,0,.12) !important;
}}

[data-testid="stMetricValue"] {{
  font-size: clamp(1.65rem, 2.4vw, 2.55rem) !important;
  letter-spacing: -.055em;
}}

.stButton > button,
.stDownloadButton > button {{
  min-height: 2.8rem;
  color: var(--astra-button-fg) !important;
  border-radius: 999px !important;
  letter-spacing: -.015em;
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}}

.stButton > button *,
.stDownloadButton > button * {{
  color: var(--astra-button-fg) !important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 12px 28px rgba(0,0,0,.22);
}}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-testid="stFileUploaderDropzone"] {{
  border-radius: 18px !important;
  min-height: 3rem;
}}

[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"] {{
  border-radius: 24px !important;
  box-shadow: none !important;
}}

[data-testid="stExpander"] summary {{
  padding: .95rem 1.15rem !important;
}}

[data-testid="stPlotlyChart"] {{
  padding: 1rem 1rem .35rem !important;
  transition: transform .24s ease, border-color .24s ease;
}}

[data-testid="stPlotlyChart"]:hover {{
  transform: translateY(-3px);
  border-color: var(--mg-text) !important;
}}

[data-testid="stAlert"] {{
  border-radius: 18px !important;
}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: .5rem;
  padding: .35rem;
  border: 1px solid var(--mg-border);
  border-radius: 999px;
  background: var(--mg-surface);
}}

[data-testid="stTabs"] [role="tab"] {{
  border-radius: 999px;
  padding-left: 1rem;
  padding-right: 1rem;
}}

.mg-json-block,
[data-testid="stCodeBlock"] {{
  border-radius: 22px !important;
}}

.card {{
  border-radius: 24px !important;
  padding: 1.35rem 1.45rem !important;
}}

/* Gentle appear effect similar to the reference template. */
[data-testid="stMainBlockContainer"] > div > div:not(:first-child) {{
  animation: astra-rise .55s cubic-bezier(.2,.75,.25,1) both;
}}

@keyframes astra-rise {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 980px) {{
  .hero {{
    min-height: 520px;
    background: linear-gradient(165deg, #f4f3ee 0 40%, #0a0a0a 40.2% 100%);
  }}

  .hero::before {{ inset: 40% 0 0 0; }}
  .hero::after {{ left: 60%; top: 13%; width: 6rem; height: 6rem; }}

  .hero-grid {{
    grid-template-columns: 1fr;
    align-items: end;
  }}

  .hero-title {{
    white-space: normal;
    font-size: clamp(3.6rem, 16vw, 7rem) !important;
  }}

  .hero-aside {{
    padding-left: 0;
  }}
}}

@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{
    padding-left: .75rem;
    padding-right: .75rem;
  }}

  .hero {{
    min-height: 570px;
    border-radius: 24px;
    padding: 1.25rem;
  }}

  .hero-title {{
    font-size: clamp(3rem, 19vw, 5.5rem) !important;
  }}

  [data-testid="stMetric"] {{
    min-height: 100px;
  }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
    scroll-behavior: auto !important;
  }}
}}
</style>
""",
    unsafe_allow_html=True,
)


def image_data_uri(path: Path) -> str:
    """Return a local image as a self-contained data URI."""

    if not path.exists():
        return ""

    suffix = path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def navigate_to(page_name: str) -> None:
    """Navigate between MissionGuard workspaces from CTA buttons."""

    st.session_state["workspace_page"] = page_name


def render_page_header(
    index: str,
    eyebrow: str,
    title: str,
    description: str,
) -> None:
    """Render a compact cinematic heading for a feature workspace."""

    st.markdown(
        f"""
<div class="mg-page-hero">
  <div class="mg-page-index">{html.escape(index)}</div>
  <div class="mg-page-copy">
    <span>{html.escape(eyebrow)}</span>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(description)}</p>
  </div>
  <div class="mg-page-orbit" aria-hidden="true"><i></i><b></b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def install_streamlit_theme_bridge() -> None:
    """
    Apply the selected MissionGuard theme to Streamlit's
    native connection-error dialog.

    Streamlit mounts that system dialog outside the normal
    app content tree, so a small parent-document bridge is
    used in addition to the regular application CSS.
    """

    bridge_colors = {
        "mode": appearance_mode.lower(),
        "background": theme["background"],
        "text": theme["text"],
        "modalBackground": theme["modal_background"],
        "modalText": theme["modal_text"],
        "modalBorder": theme["modal_border"],
        "modalShadow": theme["modal_shadow"],
        "codeBackground": theme["modal_code_background"],
        "codeText": theme["modal_code_text"],
    }

    bridge_script = f"""
<script>
(() => {{
  const parentWindow = window.parent;
  const parentDocument = parentWindow.document;
  const colors = {json.dumps(bridge_colors)};

  parentDocument.documentElement.style.colorScheme = colors.mode;

  const styleId = "missionguard-native-theme";
  let styleElement = parentDocument.getElementById(styleId);

  if (!styleElement) {{
    styleElement = parentDocument.createElement("style");
    styleElement.id = styleId;
    parentDocument.head.appendChild(styleElement);
  }}

  styleElement.textContent = `
    [data-baseweb="modal"] [role="dialog"],
    [data-testid="stConnectionStatus"],
    div[role="dialog"] {{
      background-color: ${{colors.modalBackground}} !important;
      color: ${{colors.modalText}} !important;
      border: 1px solid ${{colors.modalBorder}} !important;
      box-shadow: ${{colors.modalShadow}} !important;
    }}

    [data-baseweb="modal"] [role="dialog"] h1,
    [data-baseweb="modal"] [role="dialog"] h2,
    [data-baseweb="modal"] [role="dialog"] h3,
    [data-baseweb="modal"] [role="dialog"] h4,
    [data-baseweb="modal"] [role="dialog"] p,
    [data-baseweb="modal"] [role="dialog"] span,
    [data-testid="stConnectionStatus"] h1,
    [data-testid="stConnectionStatus"] h2,
    [data-testid="stConnectionStatus"] h3,
    [data-testid="stConnectionStatus"] h4,
    [data-testid="stConnectionStatus"] p,
    [data-testid="stConnectionStatus"] span,
    div[role="dialog"] h1,
    div[role="dialog"] h2,
    div[role="dialog"] h3,
    div[role="dialog"] h4,
    div[role="dialog"] p,
    div[role="dialog"] span {{
      color: ${{colors.modalText}} !important;
    }}

    [data-baseweb="modal"] [role="dialog"] button,
    [data-baseweb="modal"] [role="dialog"] button svg,
    [data-testid="stConnectionStatus"] button,
    [data-testid="stConnectionStatus"] button svg,
    div[role="dialog"] button,
    div[role="dialog"] button svg {{
      color: ${{colors.modalText}} !important;
      fill: ${{colors.modalText}} !important;
      stroke: ${{colors.modalText}} !important;
    }}

    [data-baseweb="modal"] [role="dialog"] pre,
    [data-baseweb="modal"] [role="dialog"] code,
    [data-testid="stConnectionStatus"] pre,
    [data-testid="stConnectionStatus"] code,
    div[role="dialog"] pre,
    div[role="dialog"] code {{
      background-color: ${{colors.codeBackground}} !important;
      color: ${{colors.codeText}} !important;
    }}
  `;

  function findConnectionPanel(titleNode) {{
    let node = titleNode;

    while (node && node !== parentDocument.body) {{
      const rectangle = node.getBoundingClientRect();
      const computed = parentWindow.getComputedStyle(node);
      const background = computed.backgroundColor;
      const radius = Number.parseFloat(computed.borderRadius) || 0;

      const hasVisibleBackground = (
        background &&
        background !== "rgba(0, 0, 0, 0)" &&
        background !== "transparent"
      );

      if (
        rectangle.width >= 280 &&
        rectangle.height >= 130 &&
        hasVisibleBackground &&
        radius >= 8
      ) {{
        return node;
      }}

      node = node.parentElement;
    }}

    return (
      titleNode.closest('[role="dialog"]') ||
      titleNode.closest('[data-baseweb="modal"]') ||
      titleNode.closest('[data-testid="stConnectionStatus"]')
    );
  }}

  function applyConnectionTheme() {{
    const possibleTitles = Array.from(
      parentDocument.querySelectorAll("h1, h2, h3, h4, p, span, div")
    );

    const connectionTitles = possibleTitles.filter((element) => (
      element.textContent &&
      element.textContent.trim() === "Connection error"
    ));

    connectionTitles.forEach((titleNode) => {{
      const panel = findConnectionPanel(titleNode);

      if (!panel) {{
        return;
      }}

      panel.style.setProperty(
        "background-color",
        colors.modalBackground,
        "important"
      );
      panel.style.setProperty(
        "color",
        colors.modalText,
        "important"
      );
      panel.style.setProperty(
        "border",
        `1px solid ${{colors.modalBorder}}`,
        "important"
      );
      panel.style.setProperty(
        "box-shadow",
        colors.modalShadow,
        "important"
      );

      panel.querySelectorAll("*").forEach((element) => {{
        element.style.setProperty(
          "color",
          colors.modalText,
          "important"
        );
      }});

      panel.querySelectorAll("button, button svg").forEach((element) => {{
        element.style.setProperty(
          "color",
          colors.modalText,
          "important"
        );
        element.style.setProperty(
          "fill",
          colors.modalText,
          "important"
        );
        element.style.setProperty(
          "stroke",
          colors.modalText,
          "important"
        );
      }});

      const commandLeaf = Array.from(
        panel.querySelectorAll("*")
      ).find((element) => (
        element.children.length === 0 &&
        element.textContent &&
        element.textContent.includes(
          "streamlit run yourscript.py"
        )
      ));

      if (commandLeaf) {{
        let commandBox = (
          commandLeaf.closest("pre, code") ||
          commandLeaf.parentElement
        );

        for (let step = 0; step < 4 && commandBox; step += 1) {{
          const rectangle = commandBox.getBoundingClientRect();

          if (
            rectangle.width >= 180 &&
            rectangle.height >= 36
          ) {{
            break;
          }}

          commandBox = commandBox.parentElement;
        }}

        if (commandBox) {{
          commandBox.style.setProperty(
            "background-color",
            colors.codeBackground,
            "important"
          );
          commandBox.style.setProperty(
            "color",
            colors.codeText,
            "important"
          );

          commandBox.querySelectorAll("*").forEach((element) => {{
            element.style.setProperty(
              "color",
              colors.codeText,
              "important"
            );
          }});
        }}
      }}
    }});
  }}

  if (parentWindow.__missionGuardThemeObserver) {{
    parentWindow.__missionGuardThemeObserver.disconnect();
  }}

  const observer = new MutationObserver(() => {{
    applyConnectionTheme();
  }});

  observer.observe(
    parentDocument.documentElement,
    {{
      childList: true,
      subtree: true,
      attributes: true,
    }}
  );

  parentWindow.__missionGuardThemeObserver = observer;
  parentWindow.__missionGuardApplyConnectionTheme = (
    applyConnectionTheme
  );

  applyConnectionTheme();
}})();
</script>
"""

    html_parameters = inspect.signature(st.html).parameters

    if "unsafe_allow_javascript" in html_parameters:
        st.html(
            bridge_script,
            width="content",
            unsafe_allow_javascript=True,
        )
    else:
        components.html(
            bridge_script,
            height=0,
            scrolling=False,
        )


install_streamlit_theme_bridge()


def stretch_width_kwargs(element_callable: object) -> dict[str, object]:
    """Return full-width kwargs compatible with old and new Streamlit."""

    try:
        width_parameter = inspect.signature(element_callable).parameters.get("width")
    except (TypeError, ValueError):
        width_parameter = None

    # Recent Streamlit releases accept width="stretch". Older releases
    # (including the project's declared minimum 1.42) use
    # use_container_width=True instead.
    if width_parameter is not None and "Width" in str(width_parameter.annotation):
        return {"width": "stretch"}

    return {"use_container_width": True}


def style_figure(figure: go.Figure) -> go.Figure:
    axis_font = {
        "color": theme["text"],
    }

    default_margin = {
        "t": 72,
        "r": 36,
        "b": 72,
        "l": 64,
    }

    current_margin = getattr(figure.layout, "margin", None)
    merged_margin = {
        "t": getattr(current_margin, "t", None) or default_margin["t"],
        "r": getattr(current_margin, "r", None) or default_margin["r"],
        "b": getattr(current_margin, "b", None) or default_margin["b"],
        "l": getattr(current_margin, "l", None) or default_margin["l"],
    }

    trace_count = len(getattr(figure, "data", []))
    default_height = 520 if trace_count > 1 else 460

    figure.update_layout(
        template=(
            "plotly_dark"
            if appearance_mode == "Dark"
            else "plotly_white"
        ),
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        height=figure.layout.height or default_height,
        margin=merged_margin,
        font={
            "color": theme["text"],
        },
        title_font={
            "color": theme["text"],
            "size": 20,
        },
        title={
            "y": 0.98,
            "x": 0.02,
            "xanchor": "left",
        },
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {
                "color": theme["text"],
            },
        },
        hoverlabel={
            "bgcolor": theme["surface_alt"],
            "font": {
                "color": theme["text"],
            },
            "bordercolor": theme["border"],
        },
        coloraxis_colorbar={
            "tickfont": axis_font,
            "title": {
                "font": axis_font,
            },
            "outlinecolor": theme["border"],
        },
    )

    figure.update_xaxes(
        automargin=True,
        color=theme["text"],
        tickfont=axis_font,
        title_font=axis_font,
        gridcolor=theme["grid"],
        linecolor=theme["border"],
        zerolinecolor=theme["border"],
    )

    figure.update_yaxes(
        automargin=True,
        color=theme["text"],
        tickfont=axis_font,
        title_font=axis_font,
        gridcolor=theme["grid"],
        linecolor=theme["border"],
        zerolinecolor=theme["border"],
    )

    figure.update_annotations(
        font={
            "color": theme["text"],
        }
    )

    return figure


def show_chart(figure: go.Figure) -> None:
    st.plotly_chart(
        style_figure(figure),
        **stretch_width_kwargs(st.plotly_chart),

        # Disable Streamlit's native Plotly theme. The app
        # already applies its own Dark/Light theme above.
        # Leaving the Streamlit theme enabled can reapply
        # dark-mode axis colors while the custom Light mode
        # is active.
        theme=None,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


def risk_color(level: str) -> str:
    return {
        "Normal": theme["success"],
        "Watch": theme["info"],
        "Warning": theme["warning"],
        "Critical": theme["danger"],
    }.get(level, theme["muted"])


@st.cache_resource
def load_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python scripts/train_opssat.py` to create the OPSSAT model artifact.")
    return load_artifact(MODEL_PATH)


@st.cache_data(ttl=30, show_spinner=False)
def load_database_status() -> dict[str, object]:
    """Return a safe PostgreSQL status payload for the sidebar."""

    if not database_is_enabled():
        return {
            "enabled": False,
            "connected": False,
            "database_name": None,
            "schema_name": None,
            "table_count": 0,
            "missing_required_tables": [],
            "error": database_status_message(),
        }

    try:
        database_info = check_database_connection()
        return {
            "enabled": True,
            "connected": True,
            "database_name": database_info["database_name"],
            "schema_name": database_info["configured_schema"],
            "table_count": database_info["table_count"],
            "missing_required_tables": database_info[
                "missing_required_tables"
            ],
            "error": None,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "connected": False,
            "database_name": None,
            "schema_name": None,
            "table_count": 0,
            "missing_required_tables": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def pgadmin_browser_details() -> dict[str, str]:
    """Return safe local pgAdmin connection details for the sidebar."""

    port = os.getenv("PGADMIN_PORT", "5050").strip() or "5050"
    email = (
        os.getenv("PGADMIN_DEFAULT_EMAIL", "admin@missionguard.com").strip()
        or "admin@missionguard.com"
    )
    database_name = os.getenv("POSTGRES_DB", "missionguard_ai").strip() or "missionguard_ai"
    database_user = os.getenv("POSTGRES_USER", "missionguard").strip() or "missionguard"

    return {
        "url": f"http://localhost:{port}",
        "email": email,
        "server_name": "MissionGuard PostgreSQL",
        "host": "postgres",
        "port": "5432",
        "database": database_name,
        "username": database_user,
    }


@st.cache_data
def load_official_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = [DATASET_PATH, SEGMENTS_PATH, PREDICTIONS_PATH, METRICS_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing packaged OPSSAT files: " + ", ".join(missing))
    dataset = pd.read_csv(DATASET_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    segments["timestamp"] = pd.to_datetime(segments["timestamp"], errors="coerce", utc=True)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return dataset, segments, predictions, metrics, metadata

def to_uuid(value: object) -> UUID:
    """
    Convert a PostgreSQL UUID value to UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def find_latest_opssat_session_id() -> UUID:
    """
    Return the latest imported OPS-SAT telemetry session.
    """

    sessions = list_telemetry_sessions()

    for telemetry_session in sessions:
        if (
            telemetry_session.get("source_type")
            == "opssat"
        ):
            return to_uuid(
                telemetry_session["id"]
            )

    raise RuntimeError(
        "No OPS-SAT telemetry session was found "
        "in PostgreSQL."
    )


def find_hybrid_model_version_id() -> UUID:
    """
    Return the registered OPS-SAT Hybrid model version.
    """

    models = list_model_versions()

    for model in models:
        if (
            model.get("model_name")
            == "OPSSAT Hybrid"
            and model.get("version") == "1.0"
        ):
            return to_uuid(
                model["id"]
            )

    raise RuntimeError(
        "OPSSAT Hybrid version 1.0 was not found "
        "in PostgreSQL."
    )


def find_latest_completed_opssat_analysis(
    model_version_id: UUID,
) -> tuple[UUID, dict[str, object]] | None:
    """
    Return the newest completed analysis across official
    OPS-SAT and uploaded OPS-SAT-compatible sessions.
    """

    sessions = list_telemetry_sessions()

    for telemetry_session in sessions:
        source_type = str(
            telemetry_session.get(
                "source_type",
                "",
            )
        )

        if source_type not in {
            "opssat",
            "uploaded_csv",
        }:
            continue

        telemetry_session_id = to_uuid(
            telemetry_session["id"]
        )

        latest_run = (
            get_latest_completed_analysis_run(
                session_id=telemetry_session_id,
                model_version_id=model_version_id,
            )
        )

        if latest_run is not None:
            return (
                telemetry_session_id,
                latest_run,
            )

    return None


def initialize_database_analysis_state() -> None:
    
    """
    Initialize Streamlit session state.
    """

    default_values = {
        "opssat_analysis_running": False,
        "opssat_analysis_completed": False,
        "opssat_analysis_run_id": None,
        "opssat_analysis_result": None,
        "opssat_analysis_error": None,
        "opssat_analysis_hydrated": False,
        "opssat_analysis_restored": False,
        "opssat_database_analysis_frame": None,
        "opssat_database_incidents": None,
        "opssat_database_raw_view": None,
        "opssat_database_source_type": None,
        "opssat_database_source_file_name": None,
        "opssat_upload_signature": None,
        "opssat_upload_save_running": False,
        "opssat_upload_save_result": None,
        "opssat_upload_save_error": None,
    }

    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value



def _json_dictionary(
    value: object,
) -> dict[str, object]:
    """
    Return a JSON-like dictionary or an empty dictionary.
    """

    if isinstance(value, dict):
        return {
            str(key): item
            for key, item in value.items()
        }

    return {}


def build_database_analysis_frame(
    prediction_records: list[dict[str, object]],
) -> pd.DataFrame:
    """
    Convert PostgreSQL prediction rows into the DataFrame
    schema expected by the existing Streamlit dashboard.
    """

    rows: list[dict[str, object]] = []

    for record in prediction_records:
        sample_metadata = _json_dictionary(
            record.get("sample_metadata")
        )

        feature_values = _json_dictionary(
            record.get("feature_values")
        )

        prediction_metadata = _json_dictionary(
            record.get("prediction_metadata")
        )

        feature_contributions = _json_dictionary(
            record.get("feature_contributions")
        )

        predicted_anomaly = bool(
            record.get("predicted_anomaly", False)
        )

        ground_truth_value = record.get(
            "ground_truth_label"
        )

        ground_truth_label: float = float("nan")

        if ground_truth_value is not None:
            ground_truth_label = float(
                bool(ground_truth_value)
            )

        segment_value = sample_metadata.get(
            "segment",
            record.get("sample_index", 0),
        )

        channel_value = str(
            sample_metadata.get(
                "channel",
                "unknown",
            )
        )

        row: dict[str, object] = {
            "segment": int(segment_value),
            "anomaly": ground_truth_label,
            "train": sample_metadata.get(
                "train_flag",
                float("nan"),
            ),
            "channel": channel_value,
            "prediction": int(predicted_anomaly),
            "prediction_label": str(
                prediction_metadata.get(
                    "prediction_label",
                    (
                        "Anomaly"
                        if predicted_anomaly
                        else "Normal"
                    ),
                )
            ),
            "risk_level": str(
                record.get(
                    "risk_level",
                    "Normal",
                )
            ),
            "hybrid_score": float(
                record.get(
                    "risk_score",
                    0.0,
                )
            ),
            "confidence": float(
                record.get(
                    "confidence_score",
                    0.0,
                )
            ),
            "decision_margin": float(
                prediction_metadata.get(
                    "decision_margin",
                    record.get(
                        "confidence_score",
                        0.0,
                    ),
                )
            ),
            "isolation_score": float(
                record.get(
                    "isolation_score",
                    0.0,
                )
                or 0.0
            ),
            "supervised_score": float(
                prediction_metadata.get(
                    "supervised_score",
                    0.0,
                )
            ),
            "top_feature": str(
                record.get(
                    "top_feature",
                    "unknown",
                )
            ),
            "top_feature_contribution": float(
                prediction_metadata.get(
                    "top_feature_contribution",
                    0.0,
                )
            ),
            "feature_contributions": (
                feature_contributions
            ),
            "explanation": str(
                record.get(
                    "explanation",
                    "",
                )
            ),
            "telemetry_sample_id": int(
                record.get(
                    "telemetry_sample_id",
                    0,
                )
            ),
            "sample_index": int(
                record.get(
                    "sample_index",
                    0,
                )
            ),
            "timestamp": record.get(
                "timestamp"
            ),
            "segment_identifier": record.get(
                "segment_identifier"
            ),
            "human_review_required": bool(
                record.get(
                    "human_review_required",
                    False,
                )
            ),
            "out_of_distribution": bool(
                record.get(
                    "out_of_distribution",
                    False,
                )
            ),
        }

        row.update(
            feature_values
        )

        rows.append(
            row
        )

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:
        return dataframe

    dataframe.sort_values(
        by="sample_index",
        inplace=True,
    )

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    return dataframe


def load_database_analysis_details(
    analysis_run_id: UUID,
) -> None:
    """
    Load persisted predictions, incidents, and source
    context into Streamlit session state.
    """

    prediction_records = (
        list_analysis_prediction_records(
            analysis_run_id=analysis_run_id,
            schema_name=(
                "opssat_segment_features"
            ),
        )
    )

    incident_records = list_incidents(
        analysis_run_id
    )

    st.session_state[
        "opssat_database_analysis_frame"
    ] = build_database_analysis_frame(
        prediction_records
    )

    st.session_state[
        "opssat_database_incidents"
    ] = incident_records

    st.session_state[
        "opssat_database_raw_view"
    ] = None

    st.session_state[
        "opssat_database_source_type"
    ] = None

    st.session_state[
        "opssat_database_source_file_name"
    ] = None

    analysis_run = get_analysis_run(
        analysis_run_id
    )

    if analysis_run is None:
        return

    telemetry_session_id = to_uuid(
        analysis_run["session_id"]
    )

    telemetry_session = get_telemetry_session(
        telemetry_session_id
    )

    if telemetry_session is None:
        return

    source_type = str(
        telemetry_session.get(
            "source_type",
            "",
        )
    )

    source_file_name = telemetry_session.get(
        "source_file_name"
    )

    st.session_state[
        "opssat_database_source_type"
    ] = source_type

    st.session_state[
        "opssat_database_source_file_name"
    ] = (
        str(source_file_name)
        if source_file_name
        else None
    )

    if source_type != "uploaded_csv":
        return

    session_metadata = _json_dictionary(
        telemetry_session.get(
            "metadata"
        )
    )

    raw_file_path_value = session_metadata.get(
        "raw_file_path"
    )

    if not isinstance(
        raw_file_path_value,
        str,
    ) or not raw_file_path_value.strip():
        return

    raw_file_path = Path(
        raw_file_path_value
    )

    if not raw_file_path.is_absolute():
        raw_file_path = (
            PROJECT_ROOT
            / raw_file_path
        )

    if not raw_file_path.exists():
        return

    try:
        persisted_upload = pd.read_csv(
            raw_file_path
        )

        (
            _,
            persisted_raw_view,
            _,
        ) = detect_and_prepare_upload(
            persisted_upload
        )

    except Exception:
        return

    if isinstance(
        persisted_raw_view,
        pd.DataFrame,
    ) and not persisted_raw_view.empty:
        st.session_state[
            "opssat_database_raw_view"
        ] = persisted_raw_view.copy()


def hydrate_latest_database_analysis() -> None:
    """
    Restore the latest completed official or uploaded
    OPS-SAT analysis from PostgreSQL.
    """

    if bool(
        st.session_state[
            "opssat_analysis_hydrated"
        ]
    ):
        return

    st.session_state[
        "opssat_analysis_hydrated"
    ] = True

    try:
        model_version_id = (
            find_hybrid_model_version_id()
        )

        latest_context = (
            find_latest_completed_opssat_analysis(
                model_version_id
            )
        )

        if latest_context is None:
            return

        _, latest_run = latest_context

        analysis_run_id = to_uuid(
            latest_run["id"]
        )

        prediction_summary = (
            get_prediction_risk_summary(
                analysis_run_id
            )
        )

        load_database_analysis_details(
            analysis_run_id
        )

        st.session_state[
            "opssat_analysis_run_id"
        ] = str(
            analysis_run_id
        )

        st.session_state[
            "opssat_analysis_result"
        ] = {
            "analysis_run_id": str(
                analysis_run_id
            ),
            "total_predictions": int(
                prediction_summary[
                    "total_predictions"
                ]
            ),
            "total_anomalies": int(
                prediction_summary[
                    "total_anomalies"
                ]
            ),
            "total_incidents": int(
                latest_run[
                    "total_incidents"
                ]
            ),
            "mean_risk_score": float(
                prediction_summary[
                    "mean_risk_score"
                ]
            ),
            "maximum_risk_score": float(
                prediction_summary[
                    "maximum_risk_score"
                ]
            ),
            "mission_health_score": (
                float(latest_run["mission_health_score"])
                if latest_run.get("mission_health_score")
                is not None
                else None
            ),
        }

        st.session_state[
            "opssat_analysis_completed"
        ] = True

        st.session_state[
            "opssat_analysis_restored"
        ] = True

        st.session_state[
            "opssat_analysis_error"
        ] = None

    except Exception as exc:
        st.session_state[
            "opssat_analysis_error"
        ] = (
            "Could not restore the latest "
            "PostgreSQL analysis: "
            f"{type(exc).__name__}: {exc}"
        )


mission_accent = "#ff5a36" if appearance_mode == "Dark" else "#d83b18"
mission_glow = "rgba(255,90,54,.34)" if appearance_mode == "Dark" else "rgba(216,59,24,.22)"

st.markdown(
    f"""
<style>
:root {{
  --mg-accent: {mission_accent};
  --mg-glow: {mission_glow};
}}

#MainMenu, footer {{ visibility: hidden; }}

.mg-topbar {{
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 1rem;
  min-height: 3.4rem;
  margin: -.35rem 0 1.15rem;
  padding: .55rem .75rem;
  border-bottom: 1px solid var(--mg-border);
  font-size: .68rem;
  letter-spacing: .12em;
  text-transform: uppercase;
}}

.mg-wordmark {{
  display: flex;
  align-items: center;
  gap: .65rem;
  color: var(--mg-text);
  font-weight: 850;
}}

.mg-wordmark span {{
  display: inline-grid;
  place-items: center;
  width: 2rem;
  height: 2rem;
  border: 1px solid var(--mg-text);
  border-radius: 50%;
  font-size: .58rem;
}}

.mg-topbar-meta {{
  color: var(--mg-muted);
  text-align: center;
}}

.mg-topbar-meta b {{ color: var(--mg-accent); }}

.mg-live-pill {{
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  padding: .42rem .65rem;
  border: 1px solid var(--mg-border);
  border-radius: 999px;
  color: var(--mg-text);
  font-weight: 800;
}}

.mg-live-pill i {{
  width: .42rem;
  height: .42rem;
  border-radius: 50%;
  background: var(--mg-accent);
  box-shadow: 0 0 0 .25rem var(--mg-glow);
}}

.mg-home-hero {{
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1.12fr) minmax(320px, .88fr);
  min-height: 690px;
  overflow: hidden;
  border: 1px solid #2a2a2a;
  border-radius: 0;
  background:
    radial-gradient(circle at 78% 54%, rgba(255,90,54,.17), transparent 24%),
    radial-gradient(circle at 76% 45%, rgba(255,255,255,.09), transparent 32%),
    linear-gradient(115deg, #050505 0 58%, #0c0c0c 58% 100%);
  color: #fff;
  box-shadow: 0 40px 100px rgba(0,0,0,.34);
  isolation: isolate;
}}

.mg-home-hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  opacity: .7;
  background-image:
    radial-gradient(circle at 10% 14%, #fff 0 1px, transparent 1.4px),
    radial-gradient(circle at 30% 74%, #fff 0 1px, transparent 1.4px),
    radial-gradient(circle at 66% 17%, #fff 0 1px, transparent 1.4px),
    radial-gradient(circle at 90% 80%, #fff 0 1px, transparent 1.4px);
  background-size: 113px 113px, 171px 171px, 227px 227px, 293px 293px;
}}

.mg-home-copy-area {{
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  padding: clamp(2rem, 5vw, 5rem);
  padding-right: 1rem;
}}

.mg-eyebrow {{
  display: inline-flex;
  width: fit-content;
  align-items: center;
  gap: .6rem;
  margin-bottom: 1.7rem;
  color: rgba(255,255,255,.72);
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
}}

.mg-eyebrow::before {{
  content: "";
  width: 2.5rem;
  height: 1px;
  background: var(--mg-accent);
}}

.mg-display {{
  margin: 0 !important;
  color: #fff !important;
  font-size: clamp(5rem, 11.5vw, 11.5rem) !important;
  line-height: .72 !important;
  letter-spacing: -.09em !important;
  font-weight: 930 !important;
  text-transform: uppercase;
}}

.mg-display span {{ color: var(--mg-accent) !important; }}

.mg-home-copy {{
  max-width: 47rem;
  margin: 2.2rem 0 1.5rem !important;
  color: rgba(255,255,255,.82) !important;
  font-size: clamp(1rem, 1.45vw, 1.23rem);
  line-height: 1.62;
}}

.mg-hero-tags {{
  display: flex;
  flex-wrap: wrap;
  gap: .6rem;
}}

.mg-hero-tags span {{
  padding: .48rem .7rem;
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 999px;
  color: rgba(255,255,255,.82);
  background: rgba(0,0,0,.28);
  font-size: .68rem;
  font-weight: 750;
  letter-spacing: .08em;
  text-transform: uppercase;
}}

.mg-orbit-stage {{
  position: relative;
  min-height: 690px;
  overflow: hidden;
}}

.mg-planet {{
  position: absolute;
  width: clamp(320px, 40vw, 610px);
  aspect-ratio: 1;
  right: clamp(-110px, -5vw, -35px);
  top: 50%;
  transform: translateY(-50%);
  border-radius: 50%;
  background:
    radial-gradient(circle at 37% 29%, rgba(255,255,255,.86) 0 1.6%, transparent 1.9%),
    radial-gradient(circle at 62% 37%, rgba(255,255,255,.15) 0 7%, transparent 7.5%),
    radial-gradient(circle at 42% 68%, rgba(255,90,54,.4) 0 9%, transparent 9.5%),
    radial-gradient(circle at 31% 31%, #565656 0 3%, #242424 31%, #0c0c0c 68%, #000 100%);
  box-shadow:
    -36px 2px 70px rgba(255,255,255,.08),
    -55px 0 120px var(--mg-glow),
    inset 26px -18px 65px rgba(0,0,0,.78);
}}

.mg-planet::before,
.mg-planet::after {{
  content: "";
  position: absolute;
  inset: -15%;
  border: 1px solid rgba(255,255,255,.19);
  border-radius: 50%;
  transform: rotate(-22deg) scaleY(.38);
}}

.mg-planet::after {{
  inset: -31%;
  border-color: rgba(255,90,54,.4);
  transform: rotate(31deg) scaleY(.26);
}}

.mg-satellite {{
  position: absolute;
  left: 9%;
  top: 23%;
  width: 72px;
  height: 26px;
  border: 1px solid rgba(255,255,255,.7);
  background: rgba(255,255,255,.08);
  transform: rotate(-17deg);
  box-shadow: 0 0 35px rgba(255,255,255,.16);
}}

.mg-satellite::before,
.mg-satellite::after {{
  content: "";
  position: absolute;
  top: 4px;
  width: 44px;
  height: 18px;
  border: 1px solid rgba(255,90,54,.72);
  background: repeating-linear-gradient(90deg, rgba(255,90,54,.32) 0 5px, transparent 5px 8px);
}}

.mg-satellite::before {{ right: 78px; }}
.mg-satellite::after {{ left: 78px; }}

.mg-orbit-label {{
  position: absolute;
  right: 2rem;
  bottom: 2rem;
  max-width: 18rem;
  padding: 1rem;
  border-left: 2px solid var(--mg-accent);
  color: rgba(255,255,255,.75);
  background: rgba(0,0,0,.46);
  backdrop-filter: blur(12px);
  font-size: .75rem;
  line-height: 1.55;
  letter-spacing: .06em;
  text-transform: uppercase;
}}

.mg-launch-actions {{ margin: 1rem 0 2.3rem; }}

.mg-stat-strip {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  margin: 0 0 4.5rem;
  border: 1px solid var(--mg-border);
  border-top: 0;
  background: var(--mg-surface);
}}

.mg-stat {{
  min-height: 128px;
  padding: 1.25rem 1.35rem;
  border-right: 1px solid var(--mg-border);
}}

.mg-stat:last-child {{ border-right: 0; }}
.mg-stat span {{ color: var(--mg-muted); font-size: .66rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
.mg-stat b {{ display: block; margin-top: .55rem; color: var(--mg-text); font-size: clamp(1.65rem, 3vw, 2.8rem); letter-spacing: -.06em; }}

.mg-section {{ margin: 4.75rem 0; }}
.mg-section-head {{ display: grid; grid-template-columns: .4fr 1fr; gap: 2rem; margin-bottom: 2rem; align-items: end; }}
.mg-section-no {{ color: var(--mg-accent); font-size: .72rem; font-weight: 850; letter-spacing: .14em; text-transform: uppercase; }}
.mg-section h2 {{ margin: 0 !important; color: var(--mg-text) !important; font-size: clamp(2.6rem, 6vw, 6rem) !important; line-height: .9 !important; text-transform: uppercase; }}
.mg-section-lead {{ max-width: 54rem; color: var(--mg-muted) !important; font-size: 1.05rem; line-height: 1.65; }}

.mg-feature-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--mg-border); border-left: 1px solid var(--mg-border); }}
.mg-feature-card {{ position: relative; min-height: 255px; padding: 1.45rem; border-right: 1px solid var(--mg-border); border-bottom: 1px solid var(--mg-border); background: var(--mg-surface); overflow: hidden; transition: background .24s ease, transform .24s ease; }}
.mg-feature-card:hover {{ background: var(--mg-surface-alt); transform: translateY(-4px); }}
.mg-feature-card .no {{ color: var(--mg-accent); font-size: .68rem; font-weight: 850; letter-spacing: .12em; }}
.mg-feature-card h3 {{ margin: 3rem 0 .7rem !important; font-size: 1.5rem !important; text-transform: uppercase; }}
.mg-feature-card p {{ color: var(--mg-muted) !important; line-height: 1.55; }}
.mg-feature-card::after {{ content: "↗"; position: absolute; right: 1.25rem; top: 1.1rem; color: var(--mg-muted); font-size: 1.2rem; }}

.mg-process-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--mg-border); border: 1px solid var(--mg-border); }}
.mg-process-step {{ min-height: 260px; padding: 1.45rem; background: var(--mg-surface); }}
.mg-process-step b {{ color: var(--mg-accent); font-size: 2.8rem; letter-spacing: -.08em; }}
.mg-process-step h3 {{ margin-top: 3.4rem !important; text-transform: uppercase; }}
.mg-process-step p {{ color: var(--mg-muted) !important; line-height: 1.55; }}

.mg-proof-band {{
  position: relative;
  display: grid;
  grid-template-columns: 1fr .75fr;
  min-height: 440px;
  overflow: hidden;
  margin: 5rem 0;
  padding: clamp(2rem, 5vw, 4.5rem);
  color: #fff;
  background:
    linear-gradient(90deg, rgba(0,0,0,.96) 0 55%, rgba(0,0,0,.48) 76%, rgba(0,0,0,.92) 100%),
    radial-gradient(circle at 78% 48%, #383838, #080808 47%, #000 70%);
  border: 1px solid #2e2e2e;
}}
.mg-proof-band::after {{ content: ""; position: absolute; width: 290px; height: 290px; right: 8%; top: 50%; transform: translateY(-50%); border-radius: 50%; background: radial-gradient(circle at 34% 26%, #777, #242424 28%, #050505 72%); box-shadow: 0 0 90px var(--mg-glow); }}
.mg-proof-copy {{ position: relative; z-index: 1; align-self: end; }}
.mg-proof-copy h2 {{ color: #fff !important; font-size: clamp(2.6rem, 6vw, 6rem) !important; line-height: .88 !important; text-transform: uppercase; }}
.mg-proof-copy p {{ max-width: 42rem; color: rgba(255,255,255,.7) !important; line-height: 1.65; }}

.mg-page-hero {{
  position: relative;
  display: grid;
  grid-template-columns: 90px minmax(0, 1fr) 240px;
  gap: 1.4rem;
  align-items: end;
  min-height: 260px;
  margin-bottom: 1.5rem;
  padding: 2rem;
  overflow: hidden;
  border: 1px solid var(--mg-border);
  background: linear-gradient(118deg, var(--mg-surface) 0 68%, var(--mg-surface-alt) 68% 100%);
}}
.mg-page-index {{ align-self: start; color: var(--mg-accent); font-size: .72rem; font-weight: 850; letter-spacing: .14em; }}
.mg-page-copy span {{ color: var(--mg-muted); font-size: .7rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }}
.mg-page-copy h1 {{ margin: .6rem 0 .7rem !important; font-size: clamp(2.7rem, 6vw, 6.5rem) !important; line-height: .86 !important; text-transform: uppercase; }}
.mg-page-copy p {{ max-width: 56rem; color: var(--mg-muted) !important; line-height: 1.6; }}
.mg-page-orbit {{ position: relative; width: 190px; height: 190px; justify-self: end; }}
.mg-page-orbit::before, .mg-page-orbit::after {{ content: ""; position: absolute; inset: 20px; border: 1px solid var(--mg-border); border-radius: 50%; }}
.mg-page-orbit::after {{ inset: 0; border-color: color-mix(in srgb, var(--mg-accent) 58%, transparent); transform: scaleY(.34) rotate(-22deg); }}
.mg-page-orbit i {{ position: absolute; width: 74px; height: 74px; left: 58px; top: 58px; border-radius: 50%; background: radial-gradient(circle at 35% 30%, var(--mg-text), #494949 20%, #090909 75%); box-shadow: 0 0 48px var(--mg-glow); }}
.mg-page-orbit b {{ position: absolute; width: 9px; height: 9px; right: 18px; top: 85px; border-radius: 50%; background: var(--mg-accent); box-shadow: 0 0 16px var(--mg-accent); }}

.mg-team-intro {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; margin: 1.4rem 0 2.5rem; background: var(--mg-border); border: 1px solid var(--mg-border); }}
.mg-team-intro > div {{ padding: 1.6rem; background: var(--mg-surface); }}
.mg-team-intro p {{ color: var(--mg-muted) !important; line-height: 1.65; }}
.mg-team-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.2rem; }}
.mg-team-card {{ display: grid; grid-template-columns: minmax(220px, .85fr) minmax(0, 1.15fr); min-height: 510px; overflow: hidden; border: 1px solid var(--mg-border); background: var(--mg-surface); }}
.mg-team-photo-wrap {{ position: relative; min-height: 510px; overflow: hidden; background: #050505; }}
.mg-team-photo-wrap::after {{ content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, transparent 55%, rgba(0,0,0,.52)); pointer-events: none; }}
.mg-team-photo {{ width: 100%; height: 100%; object-fit: cover; display: block; filter: saturate(.9) contrast(1.04); }}
.mg-team-photo.shereen {{ object-position: center 22%; }}
.mg-team-info {{ display: flex; flex-direction: column; justify-content: flex-end; padding: 1.7rem; }}
.mg-team-role {{ color: var(--mg-accent); font-size: .7rem; font-weight: 850; letter-spacing: .13em; text-transform: uppercase; }}
.mg-team-info h2 {{ margin: .7rem 0 1rem !important; font-size: clamp(2rem, 3.2vw, 3.8rem) !important; line-height: .92 !important; text-transform: uppercase; }}
.mg-team-info p {{ color: var(--mg-muted) !important; line-height: 1.65; }}
.mg-contact-link {{ display: inline-flex; margin-top: 1.2rem; color: var(--mg-text) !important; font-weight: 800; text-decoration: none; border-bottom: 1px solid var(--mg-accent); padding-bottom: .2rem; width: fit-content; }}
.mg-skill-cloud {{ display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1rem; }}
.mg-skill-cloud span {{ padding: .4rem .58rem; border: 1px solid var(--mg-border); border-radius: 999px; color: var(--mg-muted); font-size: .68rem; }}

[data-testid="stSidebar"] .stRadio > label {{
  color: var(--mg-muted) !important;
  font-size: .68rem !important;
  font-weight: 850 !important;
  letter-spacing: .14em !important;
  text-transform: uppercase;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label {{
  border-radius: 0 !important;
  border-left: 2px solid transparent !important;
  padding: .58rem .7rem !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
  border-left-color: var(--mg-accent) !important;
  background: var(--mg-surface-alt) !important;
}}

@media (max-width: 1100px) {{
  .mg-home-hero {{ grid-template-columns: 1fr; min-height: 850px; }}
  .mg-home-copy-area {{ min-height: 500px; padding-right: 2rem; }}
  .mg-orbit-stage {{ position: absolute; inset: 42% 0 0 35%; min-height: 480px; }}
  .mg-stat-strip {{ grid-template-columns: repeat(2, 1fr); }}
  .mg-stat:nth-child(2) {{ border-right: 0; }}
  .mg-stat:nth-child(-n+2) {{ border-bottom: 1px solid var(--mg-border); }}
  .mg-feature-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .mg-process-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .mg-team-grid {{ grid-template-columns: 1fr; }}
}}

@media (max-width: 760px) {{
  .mg-topbar {{ grid-template-columns: 1fr auto; }}
  .mg-topbar-meta {{ display: none; }}
  .mg-home-hero {{ min-height: 760px; }}
  .mg-home-copy-area {{ min-height: 540px; padding: 1.35rem; justify-content: flex-start; padding-top: 4rem; }}
  .mg-display {{ font-size: clamp(4.2rem, 23vw, 7.2rem) !important; }}
  .mg-orbit-stage {{ inset: 48% -14% -5% 4%; }}
  .mg-planet {{ right: -100px; }}
  .mg-stat-strip, .mg-feature-grid, .mg-process-grid, .mg-team-intro {{ grid-template-columns: 1fr; }}
  .mg-stat {{ border-right: 0; border-bottom: 1px solid var(--mg-border); }}
  .mg-section-head {{ grid-template-columns: 1fr; gap: .7rem; }}
  .mg-proof-band {{ grid-template-columns: 1fr; min-height: 520px; }}
  .mg-proof-band::after {{ width: 240px; height: 240px; right: -40px; top: 32%; }}
  .mg-page-hero {{ grid-template-columns: 1fr; min-height: 310px; }}
  .mg-page-orbit {{ position: absolute; right: -40px; top: 15px; opacity: .38; }}
  .mg-team-card {{ grid-template-columns: 1fr; }}
  .mg-team-photo-wrap {{ min-height: 430px; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="mg-topbar">
  <div class="mg-wordmark"><span>MG</span> MISSIONGUARD AI</div>
  <div class="mg-topbar-meta">REAL ESA OPS-SAT TELEMETRY <b>•</b> EXPLAINABLE MISSION INTELLIGENCE</div>
  <div class="mg-live-pill"><i></i> SYSTEM READY</div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    artifact = load_model_artifact()
    official_dataset, official_segments, official_predictions, official_metrics, official_metadata = load_official_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

initialize_database_analysis_state()
database_status = load_database_status()
database_available = bool(database_status["connected"])

if database_available:
    hydrate_latest_database_analysis()

with st.sidebar.expander("PostgreSQL / pgAdmin", expanded=True):
    pgadmin_details = pgadmin_browser_details()

    if database_available:
        missing_tables = database_status["missing_required_tables"]

        if isinstance(missing_tables, list) and missing_tables:
            st.warning(
                "PostgreSQL is connected, but the MissionGuard schema "
                "is incomplete."
            )
            st.caption("Missing tables: " + ", ".join(missing_tables))
        else:
            st.success("PostgreSQL connected")
            st.caption(
                "Database: "
                + str(database_status["database_name"])
                + " | Schema: "
                + str(database_status["schema_name"])
                + " | Tables: "
                + str(database_status["table_count"])
            )

        st.markdown(
            f"[Open pgAdmin]({pgadmin_details['url']})",
            unsafe_allow_html=False,
        )
        st.caption(
            "pgAdmin login: " + pgadmin_details["email"]
        )
        st.caption(
            "Registered server: "
            + pgadmin_details["server_name"]
            + " | Host: "
            + pgadmin_details["host"]
            + " | Port: "
            + pgadmin_details["port"]
            + " | Database: "
            + pgadmin_details["database"]
            + " | User: "
            + pgadmin_details["username"]
        )
        st.caption(
            "Use the PostgreSQL password stored locally in .env. "
            "Do not upload .env to GitHub."
        )
    elif bool(database_status.get("enabled")):
        st.warning("PostgreSQL is configured but unavailable")
        st.caption(str(database_status["error"]))
        st.caption(
            "Start the full stack with START_DOCKER_WINDOWS.bat, "
            "then refresh this status."
        )
    else:
        st.info("Local analysis mode")
        st.caption(str(database_status["error"]))
        st.caption(
            "For PostgreSQL and pgAdmin, close this local run and use "
            "START_DOCKER_WINDOWS.bat."
        )

    if st.button(
        "Refresh Database Status",
        key="refresh_database_connection_status",
        **stretch_width_kwargs(st.button),
    ):
        load_database_status.clear()
        st.rerun()


source_options = [
    "Official OPSSAT-AD Test Split",
    "Upload Real OPSSAT CSV",
]

if database_available:
    source_options.insert(
        1,
        "Latest PostgreSQL OPS-SAT Run",
    )

source_mode = st.sidebar.radio(
    "Telemetry source",
    source_options,
)

upload_validation = None
upload_features: pd.DataFrame | None = None
uploaded_bytes: bytes | None = None
upload_signature: str | None = None
raw_view: pd.DataFrame | None = None
source_name = "Official OPSSAT-AD test split"
database_incidents: list[dict[str, object]] = []
current_upload_saved = False

if source_mode == "Official OPSSAT-AD Test Split":
    analysis = official_predictions.copy()
    test_ids = set(
        analysis["segment"].astype(int).tolist()
    )
    raw_view = official_segments[
        official_segments["segment"].isin(
            test_ids
        )
    ].copy()
    st.sidebar.success(
        "Loaded the packaged official unseen "
        "OPSSAT-AD test split."
    )

elif source_mode == "Latest PostgreSQL OPS-SAT Run":
    stored_database_frame = st.session_state[
        "opssat_database_analysis_frame"
    ]

    stored_incidents = st.session_state[
        "opssat_database_incidents"
    ]

    if isinstance(
        stored_database_frame,
        pd.DataFrame,
    ) and not stored_database_frame.empty:
        analysis = stored_database_frame.copy()

        if isinstance(
            stored_incidents,
            list,
        ):
            database_incidents = stored_incidents

        database_run_id = st.session_state[
            "opssat_analysis_run_id"
        ]

        database_source_type = (
            st.session_state[
                "opssat_database_source_type"
            ]
        )

        database_source_file_name = (
            st.session_state[
                "opssat_database_source_file_name"
            ]
        )

        if database_source_type == "uploaded_csv":
            source_name = (
                "PostgreSQL uploaded CSV — "
                + str(
                    database_source_file_name
                    or "uploaded file"
                )
                + " — run "
                + str(database_run_id)
            )

            stored_raw_view = (
                st.session_state[
                    "opssat_database_raw_view"
                ]
            )

            if isinstance(
                stored_raw_view,
                pd.DataFrame,
            ) and not stored_raw_view.empty:
                raw_view = stored_raw_view.copy()
            else:
                raw_view = None

        else:
            source_name = (
                "PostgreSQL analysis run "
                + str(database_run_id)
            )

            database_segment_ids = set(
                analysis["segment"].astype(int).tolist()
            )

            raw_view = official_segments[
                official_segments["segment"].isin(
                    database_segment_ids
                )
            ].copy()

        st.sidebar.success(
            "Loaded persisted predictions and incidents "
            "from PostgreSQL."
        )

    else:
        analysis = official_predictions.copy()

        fallback_segment_ids = set(
            analysis["segment"].astype(int).tolist()
        )

        raw_view = official_segments[
            official_segments["segment"].isin(
                fallback_segment_ids
            )
        ].copy()

        source_name = (
            "Awaiting a PostgreSQL analysis run"
        )

        st.sidebar.info(
            "No PostgreSQL run is loaded. Use the "
            "button below to create one."
        )

else:
    uploaded = st.sidebar.file_uploader(
        "Upload raw segments.csv or feature dataset.csv",
        type=["csv"],
        help=(
            "Raw schema: channel, timestamp, value, segment "
            "(plus optional labels). Feature schema: the "
            "official OPSSAT dataset.csv columns."
        ),
    )

    if uploaded is None:
        st.info(
            "Upload a real OPSSAT CSV to begin analysis."
        )
        st.stop()

    try:
        uploaded_bytes = uploaded.getvalue()

        upload_signature = sha256(
            uploaded_bytes
        ).hexdigest()

        if (
            st.session_state[
                "opssat_upload_signature"
            ]
            != upload_signature
        ):
            st.session_state[
                "opssat_upload_signature"
            ] = upload_signature

            st.session_state[
                "opssat_upload_save_result"
            ] = None

            st.session_state[
                "opssat_upload_save_error"
            ] = None

        uploaded_frame = pd.read_csv(
            BytesIO(uploaded_bytes)
        )

        (
            upload_features,
            raw_view,
            upload_validation,
        ) = detect_and_prepare_upload(
            uploaded_frame
        )

        analysis = predict_feature_rows(
            upload_features,
            artifact,
        )

        source_name = uploaded.name

        st.sidebar.success(
            f"Loaded {upload_validation.segments} "
            "real telemetry segment(s)."
        )

    except Exception as exc:
        st.error(
            f"Upload validation failed: {exc}"
        )
        st.stop()

st.sidebar.divider()

if database_available:
    st.sidebar.subheader("PostgreSQL Analysis")

    if source_mode == "Latest PostgreSQL OPS-SAT Run":
        analysis_running = bool(
            st.session_state[
                "opssat_analysis_running"
            ]
        )

        analysis_completed = bool(
            st.session_state[
                "opssat_analysis_completed"
            ]
        )

        run_analysis_button = st.sidebar.button(
            "Run & Save Real Analysis",
            **stretch_width_kwargs(st.button),
            disabled=(
                analysis_running
                or analysis_completed
            ),
            key="run_opssat_database_analysis",
        )

        if run_analysis_button:
            st.session_state[
                "opssat_analysis_running"
            ] = True

            st.session_state[
                "opssat_analysis_error"
            ] = None

            try:
                with st.spinner(
                    "Running the real OPS-SAT model "
                    "and saving results to PostgreSQL..."
                ):
                    telemetry_session_id = (
                        find_latest_opssat_session_id()
                    )

                    model_version_id = (
                        find_hybrid_model_version_id()
                    )

                    database_result = (
                        run_real_opssat_analysis(
                            telemetry_session_id=(
                                telemetry_session_id
                            ),
                            model_version_id=(
                                model_version_id
                            ),
                            artifact_path=MODEL_PATH,
                        )
                    )

                st.session_state[
                    "opssat_analysis_run_id"
                ] = str(
                    database_result.analysis_run_id
                )

                st.session_state[
                    "opssat_analysis_result"
                ] = {
                    "analysis_run_id": str(
                        database_result.analysis_run_id
                    ),
                    "total_predictions": (
                        database_result.total_predictions
                    ),
                    "total_anomalies": (
                        database_result.total_anomalies
                    ),
                    "total_incidents": (
                        database_result.total_incidents
                    ),
                    "mean_risk_score": (
                        database_result.mean_risk_score
                    ),
                    "maximum_risk_score": (
                        database_result.maximum_risk_score
                    ),
                    "mission_health_score": (
                        database_result.mission_health_score
                    ),
                    "mission_health_status": (
                        database_result.mission_health_status
                    ),
                }

                st.session_state[
                    "opssat_analysis_completed"
                ] = True

                load_database_analysis_details(
                    database_result.analysis_run_id
                )

                st.session_state[
                    "opssat_analysis_restored"
                ] = False

            except Exception as exc:
                st.session_state[
                    "opssat_analysis_error"
                ] = (
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                st.session_state[
                    "opssat_analysis_running"
                ] = False

            st.rerun()


        analysis_error = st.session_state[
            "opssat_analysis_error"
        ]

        if analysis_error:
            st.sidebar.error(
                "Database analysis failed."
            )

            st.sidebar.caption(
                str(analysis_error)
            )


        stored_result = st.session_state[
            "opssat_analysis_result"
        ]

        if isinstance(stored_result, dict):
            analysis_restored = bool(
                st.session_state[
                    "opssat_analysis_restored"
                ]
            )

            if analysis_restored:
                st.sidebar.success(
                    "Latest analysis loaded from PostgreSQL."
                )
            else:
                st.sidebar.success(
                    "Analysis saved to PostgreSQL."
                )

            st.sidebar.caption(
                "Run ID: "
                + str(
                    stored_result[
                        "analysis_run_id"
                    ]
                )
            )

            st.sidebar.metric(
                "Predictions",
                int(
                    stored_result[
                        "total_predictions"
                    ]
                ),
            )

            st.sidebar.metric(
                "Detected Anomalies",
                int(
                    stored_result[
                        "total_anomalies"
                    ]
                ),
            )

            st.sidebar.metric(
            "Grouped Incidents",
                int(
                    stored_result[
                        "total_incidents"
                    ]
                ),
            )

            st.sidebar.metric(
                "Mean Risk",
                (
                    f"{float(stored_result['mean_risk_score']):.2f}"
                ),
            )

            st.sidebar.metric(
                "Maximum Risk",
                (
                    f"{float(stored_result['maximum_risk_score']):.2f}"
                ),
            )

            reset_analysis_button = st.sidebar.button(
                "Start Another Database Run",
                **stretch_width_kwargs(st.button),
                key="reset_opssat_database_analysis",
            )

            if reset_analysis_button:
                st.session_state[
                    "opssat_analysis_running"
                ] = False

                st.session_state[
                    "opssat_analysis_completed"
                ] = False

                st.session_state[
                    "opssat_analysis_run_id"
                ] = None

                st.session_state[
                    "opssat_analysis_result"
                ] = None

                st.session_state[
                    "opssat_analysis_error"
                ] = None

                st.session_state[
                    "opssat_analysis_restored"
                ] = False

                st.session_state[
                    "opssat_database_analysis_frame"
                ] = None

                st.session_state[
                    "opssat_database_incidents"
                ] = None

                st.session_state[
                    "opssat_database_raw_view"
                ] = None

                st.session_state[
                    "opssat_database_source_type"
                ] = None

                st.session_state[
                    "opssat_database_source_file_name"
                ] = None

                st.rerun()

    elif source_mode == "Official OPSSAT-AD Test Split":
        st.sidebar.info(
            "Select **Latest PostgreSQL OPS-SAT Run** "
            "to run, save, and inspect persisted results."
        )

    else:
        upload_save_result = st.session_state[
            "opssat_upload_save_result"
        ]

        upload_save_running = bool(
            st.session_state[
                "opssat_upload_save_running"
            ]
        )

        current_upload_saved = (
            isinstance(
                upload_save_result,
                dict,
            )
            and upload_signature is not None
            and upload_save_result.get(
                "upload_signature"
            )
            == upload_signature
        )

        save_uploaded_analysis = st.sidebar.button(
            "Save Upload & Analysis to PostgreSQL",
            **stretch_width_kwargs(st.button),
            disabled=(
                upload_save_running
                or current_upload_saved
                or upload_features is None
                or uploaded_bytes is None
            ),
            key="save_uploaded_opssat_analysis",
        )

        if save_uploaded_analysis:
            st.session_state[
                "opssat_upload_save_running"
            ] = True

            st.session_state[
                "opssat_upload_save_error"
            ] = None

            try:
                if (
                    upload_features is None
                    or uploaded_bytes is None
                    or upload_validation is None
                    or upload_signature is None
                ):
                    raise RuntimeError(
                        "Upload data is not available for persistence."
                    )

                with st.spinner(
                    "Saving the uploaded dataset, telemetry, "
                    "predictions, and grouped incidents..."
                ):
                    persistence_result = (
                        persist_uploaded_opssat_analysis(
                            feature_frame=upload_features,
                            original_file_bytes=(
                                uploaded_bytes
                            ),
                            original_file_name=(
                                uploaded.name
                            ),
                            model_version_id=(
                                find_hybrid_model_version_id()
                            ),
                            artifact_path=MODEL_PATH,
                            project_root=PROJECT_ROOT,
                            upload_kind=str(
                                upload_validation.kind
                            ),
                            validation_metadata={
                                "input_rows": len(
                                    uploaded_frame
                                ),
                                "accepted_rows": int(
                                    upload_validation.rows
                                ),
                                "segments": int(
                                    upload_validation.segments
                                ),
                                "channels": int(
                                    upload_validation.channels
                                ),
                                "label_coverage": float(
                                    upload_validation.label_coverage
                                ),
                                "removed_rows": int(
                                    upload_validation.removed_rows
                                ),
                                "messages": [
                                    str(message)
                                    for message in (
                                        upload_validation.messages
                                    )
                                ],
                            },
                        )
                    )

                analysis_result = (
                    persistence_result.analysis_result
                )

                st.session_state[
                    "opssat_upload_save_result"
                ] = {
                    "upload_signature": upload_signature,
                    "dataset_id": str(
                        persistence_result.dataset_id
                    ),
                    "telemetry_session_id": str(
                        persistence_result.telemetry_session_id
                    ),
                    "analysis_run_id": str(
                        analysis_result.analysis_run_id
                    ),
                    "total_predictions": int(
                        analysis_result.total_predictions
                    ),
                    "total_anomalies": int(
                        analysis_result.total_anomalies
                    ),
                    "total_incidents": int(
                        analysis_result.total_incidents
                    ),
                    "mean_risk_score": float(
                        analysis_result.mean_risk_score
                    ),
                    "maximum_risk_score": float(
                        analysis_result.maximum_risk_score
                    ),
                    "mission_health_score": float(
                        analysis_result.mission_health_score
                    ),
                    "mission_health_status": str(
                        analysis_result.mission_health_status
                    ),
                    "sha256_hash": (
                        persistence_result.sha256_hash
                    ),
                }

                st.session_state[
                    "opssat_analysis_run_id"
                ] = str(
                    analysis_result.analysis_run_id
                )

                st.session_state[
                    "opssat_analysis_result"
                ] = {
                    "analysis_run_id": str(
                        analysis_result.analysis_run_id
                    ),
                    "total_predictions": int(
                        analysis_result.total_predictions
                    ),
                    "total_anomalies": int(
                        analysis_result.total_anomalies
                    ),
                    "total_incidents": int(
                        analysis_result.total_incidents
                    ),
                    "mean_risk_score": float(
                        analysis_result.mean_risk_score
                    ),
                    "maximum_risk_score": float(
                        analysis_result.maximum_risk_score
                    ),
                    "mission_health_score": float(
                        analysis_result.mission_health_score
                    ),
                    "mission_health_status": str(
                        analysis_result.mission_health_status
                    ),
                }

                st.session_state[
                    "opssat_analysis_completed"
                ] = True

                st.session_state[
                    "opssat_analysis_restored"
                ] = False

                load_database_analysis_details(
                    analysis_result.analysis_run_id
                )

                st.rerun()

            except Exception as exc:
                st.session_state[
                    "opssat_upload_save_error"
                ] = (
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                st.session_state[
                    "opssat_upload_save_running"
                ] = False

        upload_save_error = st.session_state[
            "opssat_upload_save_error"
        ]

        if upload_save_error:
            st.sidebar.error(
                "Uploaded analysis could not be saved."
            )

            st.sidebar.caption(
                str(upload_save_error)
            )

        if current_upload_saved and isinstance(
            upload_save_result,
            dict,
        ):
            st.sidebar.success(
                "Upload, predictions, and incidents "
                "saved to PostgreSQL."
            )

            st.sidebar.caption(
                "Run ID: "
                + str(
                    upload_save_result[
                        "analysis_run_id"
                    ]
                )
            )

            st.sidebar.metric(
                "Saved Predictions",
                int(
                    upload_save_result[
                        "total_predictions"
                    ]
                ),
            )

            st.sidebar.metric(
                "Saved Incidents",
                int(
                    upload_save_result[
                        "total_incidents"
                    ]
                ),
            )

else:
    st.sidebar.subheader("Local Analysis")
    st.sidebar.info(
        "MissionGuard is running without PostgreSQL. "
        "Official telemetry analysis and CSV uploads work normally; "
        "database persistence controls are hidden."
    )

NAV_LABELS = {
    "Home": "00  Launchpad",
    "Mission Overview": "01  Mission Overview",
    "Telemetry Explorer": "02  Telemetry Explorer",
    "Incident Intelligence": "03  Incident Intelligence",
    "Upload & Test": "04  Upload & Test",
    "Model Validation": "05  Model Validation",
    "Data Drift Monitor": "06  Data Drift Monitor",
    "Reports & Responsible AI": "07  Reports & Responsible AI",
    "Dataset & Attribution": "08  Dataset & Attribution",
    "Team & Contact": "09  Team & Contact",
    "IBM Bob Evidence": "10  IBM Bob Evidence",
}

page = st.sidebar.radio(
    "Mission Workspaces",
    list(NAV_LABELS),
    key="workspace_page",
    format_func=lambda option: NAV_LABELS[option],
)

st.sidebar.divider()
st.sidebar.caption(f"Model: {artifact.get('artifact_version', 'OPSSAT model')}")
st.sidebar.caption(f"Segments analyzed: {len(analysis):,}")
st.sidebar.caption(f"Channels represented: {analysis['channel'].nunique():,}")


def prediction_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    anomaly_count = int((frame["prediction"] == 1).sum())
    anomaly_rate = anomaly_count / max(len(frame), 1) * 100
    peak_score = float(frame["hybrid_score"].max())
    if peak_score >= 82:
        status = "Critical"
    elif anomaly_count:
        status = "Warning"
    elif float(frame["hybrid_score"].mean()) >= 30:
        status = "Watch"
    else:
        status = "Normal"
    return {
        "segments": len(frame),
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "peak_score": peak_score,
        "mean_score": float(frame["hybrid_score"].mean()),
        "status": status,
    }

def safe_json_dictionary(
    value: object,
) -> dict[str, object]:
    """
    Convert a PostgreSQL JSONB value into a dictionary.
    """

    if isinstance(value, dict):
        return {
            str(key): item
            for key, item in value.items()
        }

    if isinstance(value, str):
        try:
            parsed_value = json.loads(value)

        except json.JSONDecodeError:
            return {}

        if isinstance(parsed_value, dict):
            return {
                str(key): item
                for key, item in parsed_value.items()
            }

    return {}


def safe_json_list(
    value: object,
) -> list[object]:
    """
    Convert a PostgreSQL JSONB value into a list.
    """

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        try:
            parsed_value = json.loads(value)

        except json.JSONDecodeError:
            return []

        if isinstance(parsed_value, list):
            return parsed_value

    return []


def load_current_grouped_incidents(
) -> list[dict[str, object]]:
    """
    Load incidents belonging to the current PostgreSQL run.
    """

    analysis_run_id_value = st.session_state.get(
        "opssat_analysis_run_id"
    )

    if analysis_run_id_value is None:
        return []

    analysis_run_id = to_uuid(
        analysis_run_id_value
    )

    return list_incidents(
        analysis_run_id
    )


def build_grouped_incident_frame(
    incidents: list[dict[str, object]],
) -> pd.DataFrame:
    """
    Convert stored PostgreSQL incidents into a
    dashboard-friendly DataFrame.
    """

    rows: list[dict[str, object]] = []

    for incident in incidents:
        metadata = safe_json_dictionary(
            incident.get("metadata")
        )

        affected_subsystems = safe_json_list(
            incident.get(
                "affected_subsystems"
            )
        )

        segment_ids = safe_json_list(
            metadata.get(
                "segment_ids"
            )
        )

        channel = metadata.get(
            "channel"
        )

        if not channel and affected_subsystems:
            channel = affected_subsystems[0]

        duration_samples = int(
            incident.get(
                "duration_samples"
            )
            or len(segment_ids)
            or 1
        )

        displayed_segments = ", ".join(
            str(segment_id)
            for segment_id in segment_ids[:12]
        )

        if len(segment_ids) > 12:
            displayed_segments += ", ..."

        rows.append(
            {
                "Incident Code": str(
                    incident.get(
                        "incident_code",
                        "",
                    )
                ),
                "Channel": str(
                    channel or "Unknown"
                ),
                "Severity": str(
                    incident.get(
                        "severity",
                        "Watch",
                    )
                ),
                "Start Segment": metadata.get(
                    "start_segment"
                ),
                "End Segment": metadata.get(
                    "end_segment"
                ),
                "Duration": duration_samples,
                "Peak Risk": float(
                    incident.get(
                        "peak_risk_score"
                    )
                    or 0.0
                ),
                "Peak Confidence": float(
                    incident.get(
                        "peak_confidence"
                    )
                    or 0.0
                ),
                "Top Feature": str(
                    incident.get(
                        "top_feature"
                    )
                    or "Unknown"
                ),
                "Review Required": (
                    "Yes"
                    if bool(
                        incident.get(
                            "human_review_required"
                        )
                    )
                    else "No"
                ),
                "Status": str(
                    incident.get(
                        "status",
                        "open",
                    )
                ),
                "Grouped Segments": (
                    displayed_segments
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


health_incidents: list[dict[str, object]] = []

should_load_health_incidents = (
    source_mode == "Latest PostgreSQL OPS-SAT Run"
    or (
        source_mode == "Upload Real OPSSAT CSV"
        and current_upload_saved
    )
)

if should_load_health_incidents:
    try:
        health_incidents = (
            load_current_grouped_incidents()
        )

    except Exception as exc:
        st.sidebar.warning(
            "Mission health could not load current "
            "incident workflow data."
        )
        st.sidebar.caption(
            f"{type(exc).__name__}: {exc}"
        )

mission_health = calculate_mission_health_score(
    frame=analysis,
    incidents=health_incidents,
)

summary = prediction_summary(analysis)
compatibility = validate_features_against_artifact(analysis, artifact)
drift_summary, drift_details = assess_data_drift(analysis, artifact)
row_evaluation = evaluate_binary_predictions(analysis)
event_evaluation, event_ledger = evaluate_event_detection(analysis, raw_view)

if page != "Home" and page != "Team & Contact":
    st.caption(f"Analysis source: {source_name}")

PAGE_HEADERS = {
    "Mission Overview": (
        "01",
        "Mission command",
        "Mission Overview",
        "A concise operational view of mission health, anomaly volume, peak risk, and telemetry stability.",
    ),
    "Telemetry Explorer": (
        "02",
        "Signal inspection",
        "Telemetry Explorer",
        "Inspect real spacecraft telemetry traces, engineered segment features, and model evidence at signal level.",
    ),
    "Incident Intelligence": (
        "03",
        "Operator workflow",
        "Incident Intelligence",
        "Group anomalous segments into reviewable incidents with severity, confidence, evidence, and human decisions.",
    ),
    "Upload & Test": (
        "04",
        "Bring your telemetry",
        "Upload & Test",
        "Validate real OPSSAT CSV files, run explainable predictions, and persist results safely to PostgreSQL.",
    ),
    "Model Validation": (
        "05",
        "Evidence before trust",
        "Model Validation",
        "Review held-out performance, threshold behavior, event detection, and reproducibility metadata.",
    ),
    "Data Drift Monitor": (
        "06",
        "Distribution awareness",
        "Data Drift Monitor",
        "Compare current telemetry with the learned nominal envelope and identify features whose behavior has shifted.",
    ),
    "Reports & Responsible AI": (
        "07",
        "Human-ready outputs",
        "Reports & Responsible AI",
        "Export mission evidence while keeping limitations, operator oversight, and responsible-AI guardrails explicit.",
    ),
    "Dataset & Attribution": (
        "08",
        "Scientific foundation",
        "Dataset & Attribution",
        "Document the real ESA OPS-SAT benchmark, channel coverage, feature schema, license, and reproducible data lineage.",
    ),
    "Team & Contact": (
        "09",
        "The people behind the mission",
        "Team & Contact",
        "A multidisciplinary AI, data, front-end, and visual-design team building mission intelligence for a global challenge.",
    ),
    "IBM Bob Evidence": (
        "10",
        "Challenge development trail",
        "IBM Bob Evidence",
        "A transparent record of prompts, verified implementation outcomes, and engineering decisions used during development.",
    ),
}

if page in PAGE_HEADERS:
    render_page_header(*PAGE_HEADERS[page])

if page == "Home":
    database_label = "CONNECTED" if database_available else "LOCAL MODE"
    team_label = "2 BUILDERS"

    st.markdown(
        f"""
<div class="mg-home-hero">
  <div class="mg-home-copy-area">
    <span class="mg-eyebrow">Global AI Builders Challenge / Space Intelligence</span>
    <h1 class="mg-display">Mission<br><span>Guard</span></h1>
    <p class="mg-home-copy">
      MissionGuard AI transforms real spacecraft telemetry into explainable anomaly alerts, mission-risk intelligence,
      grouped incidents, drift awareness, and operator-ready evidence — without hiding uncertainty behind a black box.
    </p>
    <div class="mg-hero-tags">
      <span>Real ESA OPS-SAT data</span>
      <span>Explainable ML</span>
      <span>PostgreSQL mission archive</span>
      <span>Human-in-the-loop</span>
    </div>
  </div>
  <div class="mg-orbit-stage" aria-hidden="true">
    <div class="mg-satellite"></div>
    <div class="mg-planet"></div>
    <div class="mg-orbit-label">
      From raw telemetry to evidence-backed action. Designed for clarity under mission pressure.
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    action_left, action_right, action_space = st.columns([1.2, 1.05, 4])
    with action_left:
        st.button(
            "START MISSION CONTROL  →",
            key="home_start_mission",
            on_click=navigate_to,
            args=("Mission Overview",),
            **stretch_width_kwargs(st.button),
        )
    with action_right:
        st.button(
            "MEET THE TEAM",
            key="home_team",
            on_click=navigate_to,
            args=("Team & Contact",),
            **stretch_width_kwargs(st.button),
        )

    st.markdown(
        f"""
<div class="mg-stat-strip">
  <div class="mg-stat"><span>Telemetry segments</span><b>{int(summary['segments']):,}</b></div>
  <div class="mg-stat"><span>Detected anomalies</span><b>{int(summary['anomaly_count']):,}</b></div>
  <div class="mg-stat"><span>Mission health</span><b>{float(mission_health['score']):.1f}/100</b></div>
  <div class="mg-stat"><span>System state</span><b>{database_label}</b></div>
</div>

<section class="mg-section">
  <div class="mg-section-head">
    <span class="mg-section-no">01 / Capabilities</span>
    <div>
      <h2>One platform.<br>Every mission signal.</h2>
      <p class="mg-section-lead">Each core capability has its own focused workspace, while every result remains connected through one mission narrative.</p>
    </div>
  </div>
  <div class="mg-feature-grid">
    <article class="mg-feature-card"><span class="no">01</span><h3>Mission Overview</h3><p>Mission health, anomaly rate, peak risk, and stability in one decision-ready command view.</p></article>
    <article class="mg-feature-card"><span class="no">02</span><h3>Telemetry Explorer</h3><p>Inspect raw traces and engineered segment evidence from real spacecraft channels.</p></article>
    <article class="mg-feature-card"><span class="no">03</span><h3>Incident Intelligence</h3><p>Group related anomalies into incidents that operators can review, resolve, and document.</p></article>
    <article class="mg-feature-card"><span class="no">04</span><h3>Upload & Test</h3><p>Validate real CSV telemetry, run predictions, and persist the complete analysis workflow.</p></article>
    <article class="mg-feature-card"><span class="no">05</span><h3>Model Validation</h3><p>Measure segment and event performance using held-out labels and transparent thresholds.</p></article>
    <article class="mg-feature-card"><span class="no">06</span><h3>Drift Monitor</h3><p>See when incoming telemetry no longer resembles the nominal training envelope.</p></article>
    <article class="mg-feature-card"><span class="no">07</span><h3>Reports & Responsible AI</h3><p>Export evidence with limitations, guardrails, and human oversight built into the narrative.</p></article>
    <article class="mg-feature-card"><span class="no">08</span><h3>Dataset & Attribution</h3><p>Trace every insight back to the OPSSAT-AD benchmark, schema, license, and data lineage.</p></article>
    <article class="mg-feature-card"><span class="no">09</span><h3>Mission Archive</h3><p>Store telemetry sessions, predictions, incidents, models, and mission-health outcomes in PostgreSQL.</p></article>
  </div>
</section>

<section class="mg-section">
  <div class="mg-section-head">
    <span class="mg-section-no">02 / Workflow</span>
    <div>
      <h2>From signal<br>to decision.</h2>
      <p class="mg-section-lead">MissionGuard keeps the workflow understandable from ingestion to operator review.</p>
    </div>
  </div>
  <div class="mg-process-grid">
    <article class="mg-process-step"><b>01</b><h3>Ingest</h3><p>Load official or uploaded OPSSAT telemetry with schema, channel, and quality validation.</p></article>
    <article class="mg-process-step"><b>02</b><h3>Detect</h3><p>Combine supervised and unsupervised evidence to identify anomalous segments and risk bands.</p></article>
    <article class="mg-process-step"><b>03</b><h3>Explain</h3><p>Surface top feature contributions, confidence, decision margin, and uncertainty-aware language.</p></article>
    <article class="mg-process-step"><b>04</b><h3>Act</h3><p>Group incidents, support human review, export reports, and preserve the full audit trail.</p></article>
  </div>
</section>

<section class="mg-proof-band">
  <div class="mg-proof-copy">
    <span class="mg-eyebrow">Scientific credibility / Operator clarity</span>
    <h2>Built on real<br>spacecraft telemetry.</h2>
    <p>MissionGuard uses the OPSSAT-AD benchmark derived from ESA's OPS-SAT CubeSat. The platform makes model evidence visible and clearly separates anomaly detection from confirmed hardware diagnosis.</p>
  </div>
</section>

<div class="mg-stat-strip">
  <div class="mg-stat"><span>Data source</span><b>ESA OPS-SAT</b></div>
  <div class="mg-stat"><span>Core approach</span><b>HYBRID ML</b></div>
  <div class="mg-stat"><span>Persistence</span><b>POSTGRESQL</b></div>
  <div class="mg-stat"><span>Team</span><b>{team_label}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

elif page == "Mission Overview":
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Segments", f"{summary['segments']:,}")
    c2.metric("Predicted Anomalies", f"{summary['anomaly_count']:,}")
    c3.metric("Anomaly Rate", f"{summary['anomaly_rate']:.1f}%")
    c4.metric("Peak Hybrid Risk", f"{summary['peak_score']:.1f}/100")
    c5.metric("Overall Status", str(summary["status"]))
    c6.metric("Data Compatibility", drift_summary["compatibility"])

    if summary["status"] == "Critical":
        st.error("At least one segment reached the critical risk band. Operator review is required.")
    elif summary["status"] == "Warning":
        st.warning("The model detected one or more anomalous telemetry segments.")
    elif summary["status"] == "Watch":
        st.info("No segment crossed the anomaly threshold, but the aggregate risk remains elevated.")
    else:
        st.success("The analyzed segments remain within the learned nominal envelope.")

    st.subheader("Prototype Mission Health Score")

    health_status = str(
        mission_health["status"]
    )

    health_score = float(
        mission_health["score"]
    )

    health_status_color = {
        "Nominal": theme["success"],
        "Stable": theme["info"],
        "Degraded": theme["warning"],
        "Critical": theme["danger"],
    }.get(
        health_status,
        theme["muted"],
    )

    health_chart_column, health_detail_column = (
        st.columns(
            [1.05, 1.95]
        )
    )

    with health_chart_column:
        health_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                number={
                    "suffix": "/100",
                    "font": {
                        "color": theme["text"],
                    },
                },
                title={
                    "text": (
                        "Mission Health — "
                        + health_status
                    ),
                    "font": {
                        "color": theme["text"],
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": theme["text"],
                    },
                    "bar": {
                        "color": health_status_color,
                    },
                    "bgcolor": theme["surface_alt"],
                    "bordercolor": theme["border"],
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": (
                                "rgba(255,92,108,0.20)"
                            ),
                        },
                        {
                            "range": [50, 70],
                            "color": (
                                "rgba(246,173,54,0.20)"
                            ),
                        },
                        {
                            "range": [70, 85],
                            "color": (
                                "rgba(105,185,255,0.20)"
                            ),
                        },
                        {
                            "range": [85, 100],
                            "color": (
                                "rgba(53,208,127,0.20)"
                            ),
                        },
                    ],
                    "threshold": {
                        "line": {
                            "color": theme["text"],
                            "width": 3,
                        },
                        "thickness": 0.75,
                        "value": health_score,
                    },
                },
            )
        )

        health_gauge.update_layout(
            title={
                "text": "",
            },
            height=320,
            margin={
                "l": 30,
                "r": 30,
                "t": 70,
                "b": 20,
            },
            showlegend=False,
        )

        show_chart(
            health_gauge
        )

    with health_detail_column:
        (
            health_metric_1,
            health_metric_2,
        ) = st.columns(2)

        (
            health_metric_3,
            health_metric_4,
        ) = st.columns(2)

        health_metric_1.metric(
            "Telemetry Stability",
            (
                f"{float(mission_health['telemetry_stability']):.1f}"
                "/100"
            ),
        )

        health_metric_2.metric(
            "Anomaly Control",
            (
                f"{float(mission_health['anomaly_control']):.1f}"
                "/100"
            ),
        )

        health_metric_3.metric(
            "Peak-Risk Resilience",
            (
                f"{float(mission_health['peak_resilience']):.1f}"
                "/100"
            ),
        )

        health_metric_4.metric(
            "Incident Readiness",
            (
                f"{float(mission_health['incident_readiness']):.1f}"
                "/100"
            ),
        )

        st.markdown(
            f"""
<div class="card">
<b>Operational interpretation:</b>
{html.escape(health_status)}
<br>
<b>Incidents considered:</b>
{int(mission_health['incidents_considered'])}
<br>
<b>Unresolved incidents:</b>
{int(mission_health['unresolved_incidents'])}
<br>
<b>Composite weights:</b>
30% telemetry stability, 20% anomaly control,
25% peak-risk resilience, and 25% incident readiness.
</div>
""",
            unsafe_allow_html=True,
        )

    st.caption(
        "This Mission Health Score is a transparent prototype "
        "decision-support index. It is not a certified flight "
        "health metric and does not confirm a spacecraft fault."
    )

    left, right = st.columns(2)
    with left:
        distribution = analysis["risk_level"].value_counts().reindex(
            ["Normal", "Watch", "Warning", "Critical"], fill_value=0
        ).reset_index()
        distribution.columns = ["Risk Level", "Segments"]
        fig = px.pie(
            distribution,
            names="Risk Level",
            values="Segments",
            hole=0.55,
            title="Predicted Risk Distribution",
            color="Risk Level",
            color_discrete_map={level: risk_color(level) for level in distribution["Risk Level"]},
        )
        show_chart(fig)
    with right:
        channel_summary = (
            analysis.groupby("channel", as_index=False)
            .agg(Segments=("segment", "count"), Anomalies=("prediction", "sum"), Mean_Risk=("hybrid_score", "mean"))
        )
        channel_summary["Channel Name"] = channel_summary["channel"].map(CHANNEL_NAMES).fillna(channel_summary["channel"])
        fig = px.bar(
            channel_summary,
            x="Channel Name",
            y="Anomalies",
            color="Mean_Risk",
            title="Predicted Anomalies by Real Telemetry Channel",
            color_continuous_scale="Reds",
            hover_data=["Segments", "Mean_Risk"],
        )
        show_chart(fig)

    st.subheader("Highest-Risk Segments")
    columns = [
        "segment", "channel", "prediction_label", "risk_level", "hybrid_score",
        "decision_margin", "top_feature", "top_feature_contribution", "anomaly", "train",
    ]
    st.dataframe(
        analysis[columns].sort_values("hybrid_score", ascending=False).head(30),
        **stretch_width_kwargs(st.dataframe),
        hide_index=True,
    )

elif page == "Telemetry Explorer":
    st.header("Real Telemetry Explorer")
    if raw_view is None or raw_view.empty:
        st.info("This feature upload contains engineered segment rows only. Upload a raw segments.csv-style file to view signal traces.")
    else:
        available = sorted(set(raw_view["segment"].astype(int)) & set(analysis["segment"].astype(int)))
        selected_segment = st.selectbox("Select a real telemetry segment", available)
        signal_frame = raw_view[raw_view["segment"].astype(int) == int(selected_segment)].copy()
        prediction_row = analysis[analysis["segment"].astype(int) == int(selected_segment)].iloc[0]
        channel = str(signal_frame["channel"].iloc[0])
        channel_name = CHANNEL_NAMES.get(channel, channel)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Segment", int(selected_segment))
        c2.metric("Channel", channel_name)
        c3.metric("Samples", len(signal_frame))
        c4.metric("Prediction", prediction_row["prediction_label"])
        c5.metric("Hybrid Risk", f"{prediction_row['hybrid_score']:.1f}/100")

        st.caption(f"Real OPS-SAT Telemetry — {channel_name} ({channel})")

        fig = px.line(
            signal_frame.sort_values("timestamp"),
            x="timestamp",
            y="value",
            labels={"value": "Telemetry Value", "timestamp": "Timestamp"},
        )
        fig.update_traces(line={"color": risk_color(str(prediction_row["risk_level"])), "width": 2.2})
        show_chart(fig)

        if pd.notna(prediction_row.get("anomaly")):
            truth = "Anomaly" if int(prediction_row["anomaly"]) == 1 else "Normal"
            st.caption(f"Ground-truth label: {truth} · Official split: {'Train' if int(prediction_row['train']) == 1 else 'Test'}")

        with st.expander("Engineered segment features"):
            feature_columns = artifact["numeric_features"]
            st.dataframe(
                pd.DataFrame({"Feature": feature_columns, "Value": [prediction_row[c] for c in feature_columns]}),
                **stretch_width_kwargs(st.dataframe),
                hide_index=True,
            )

elif page == "Incident Intelligence":
    st.header(
        "Incident Intelligence"
    )

    if (
        source_mode
        == "Latest PostgreSQL OPS-SAT Run"
    ):
        grouped_incidents = (
            load_current_grouped_incidents()
        )

        if not grouped_incidents:
            st.success(
                "No grouped incidents were found "
                "for the current PostgreSQL run."
            )

        else:
            incident_frame = (
                build_grouped_incident_frame(
                    grouped_incidents
                )
            )

            incident_lookup = {
                str(
                    incident[
                        "incident_code"
                    ]
                ): incident
                for incident in grouped_incidents
            }

            # Normalize workflow statuses for metrics and filtering.
            incident_frame["Status"] = (
                incident_frame["Status"]
                .fillna("open")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            status_order = [
                "open",
                "under_review",
                "confirmed",
                "rejected",
                "resolved",
            ]

            decision_recorded_statuses = {
                "confirmed",
                "rejected",
                "resolved",
            }

            total_incidents = len(
                incident_frame
            )

            critical_incidents = int(
                (
                    incident_frame[
                        "Severity"
                    ]
                    == "Critical"
                ).sum()
            )

            multi_segment_incidents = int(
                (
                    incident_frame[
                        "Duration"
                    ]
                    > 1
                ).sum()
            )

            pending_review_incidents = int(
                (
                    (
                        incident_frame[
                            "Review Required"
                        ]
                        == "Yes"
                    )
                    & (
                        incident_frame[
                            "Status"
                        ]
                        == "open"
                    )
                ).sum()
            )

            under_review_incidents = int(
                (
                    incident_frame[
                        "Status"
                    ]
                    == "under_review"
                ).sum()
            )

            confirmed_incidents = int(
                (
                    incident_frame[
                        "Status"
                    ]
                    == "confirmed"
                ).sum()
            )

            rejected_incidents = int(
                (
                    incident_frame[
                        "Status"
                    ]
                    == "rejected"
                ).sum()
            )

            resolved_incidents = int(
                (
                    incident_frame[
                        "Status"
                    ]
                    == "resolved"
                ).sum()
            )

            (
                metric_1,
                metric_2,
                metric_3,
                metric_4,
            ) = st.columns(4)

            metric_1.metric(
                "Grouped Incidents",
                total_incidents,
            )

            metric_2.metric(
                "Critical Incidents",
                critical_incidents,
            )

            metric_3.metric(
                "Multi-Segment Incidents",
                multi_segment_incidents,
            )

            metric_4.metric(
                "Pending Human Review",
                pending_review_incidents,
            )

            (
                workflow_metric_1,
                workflow_metric_2,
                workflow_metric_3,
                workflow_metric_4,
            ) = st.columns(4)

            workflow_metric_1.metric(
                "Under Review",
                under_review_incidents,
            )

            workflow_metric_2.metric(
                "Confirmed",
                confirmed_incidents,
            )

            workflow_metric_3.metric(
                "Rejected",
                rejected_incidents,
            )

            workflow_metric_4.metric(
                "Resolved",
                resolved_incidents,
            )

            st.caption(
                "Incidents are grouped using the same "
                "telemetry channel and nearby sample "
                "indexes. Pending Human Review counts "
                "only open incidents that have not entered "
                "the operator-review workflow."
            )

            st.subheader(
                "Incident Operations Filters"
            )

            available_statuses = [
                status
                for status in status_order
                if status
                in set(
                    incident_frame[
                        "Status"
                    ].tolist()
                )
            ]

            severity_order = [
                "Critical",
                "Warning",
                "Watch",
            ]

            available_severities = [
                severity
                for severity in severity_order
                if severity
                in set(
                    incident_frame[
                        "Severity"
                    ].tolist()
                )
            ]

            available_channels = sorted(
                incident_frame[
                    "Channel"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            (
                filter_column_1,
                filter_column_2,
                filter_column_3,
                filter_column_4,
            ) = st.columns(4)

            with filter_column_1:
                selected_statuses = st.multiselect(
                    "Status",
                    options=available_statuses,
                    default=[],
                    format_func=lambda value: (
                        value.replace(
                            "_",
                            " ",
                        ).title()
                    ),
                    help=(
                        "Leave empty to show "
                        "all workflow statuses."
                    ),
                )

            with filter_column_2:
                selected_severities = st.multiselect(
                    "Severity",
                    options=available_severities,
                    default=[],
                    help=(
                        "Leave empty to show "
                        "all severity levels."
                    ),
                )

            with filter_column_3:
                selected_channels = st.multiselect(
                    "Telemetry Channel",
                    options=available_channels,
                    default=[],
                    help=(
                        "Leave empty to show "
                        "all telemetry channels."
                    ),
                )

            with filter_column_4:
                selected_review_state = st.selectbox(
                    "Review State",
                    options=[
                        "All",
                        "Pending Human Review",
                        "Under Review",
                        "Decision Recorded",
                        "No Review Required",
                    ],
                )

            filtered_incident_frame = (
                incident_frame.copy()
            )

            if selected_statuses:
                filtered_incident_frame = (
                    filtered_incident_frame.loc[
                        filtered_incident_frame[
                            "Status"
                        ].isin(
                            selected_statuses
                        )
                    ]
                )

            if selected_severities:
                filtered_incident_frame = (
                    filtered_incident_frame.loc[
                        filtered_incident_frame[
                            "Severity"
                        ].isin(
                            selected_severities
                        )
                    ]
                )

            if selected_channels:
                filtered_incident_frame = (
                    filtered_incident_frame.loc[
                        filtered_incident_frame[
                            "Channel"
                        ].isin(
                            selected_channels
                        )
                    ]
                )

            if (
                selected_review_state
                == "Pending Human Review"
            ):
                filtered_incident_frame = (
                    filtered_incident_frame.loc[
                        (
                            filtered_incident_frame[
                                "Review Required"
                            ]
                            == "Yes"
                        )
                        & (
                            filtered_incident_frame[
                                "Status"
                            ]
                            == "open"
                        )
                    ]
                )

            elif (
                selected_review_state
                == "Under Review"
            ):
                filtered_incident_frame = (
                    filtered_incident_frame.loc[
                        filtered_incident_frame[
                            "Status"
                        ]
                        == "under_review"
                    ]
                )

            elif (
                selected_review_state
                == "Decision Recorded"
            ):
                filtered_incident_frame = (
                    filtered_incident_frame.loc[
                        filtered_incident_frame[
                            "Status"
                        ].isin(
                            decision_recorded_statuses
                        )
                    ]
                )

            elif (
                selected_review_state
                == "No Review Required"
            ):
                filtered_incident_frame = (
                    filtered_incident_frame.loc[
                        filtered_incident_frame[
                            "Review Required"
                        ]
                        == "No"
                    ]
                )

            filtered_incident_frame = (
                filtered_incident_frame.copy()
            )

            st.caption(
                "Showing "
                f"{len(filtered_incident_frame):,} "
                "of "
                f"{len(incident_frame):,} "
                "grouped incidents."
            )

            if filtered_incident_frame.empty:
                st.info(
                    "No incidents match the "
                    "selected operational filters."
                )
                st.stop()

            display_incident_frame = (
                filtered_incident_frame.copy()
            )

            display_incident_frame[
                "Status"
            ] = (
                display_incident_frame[
                    "Status"
                ]
                .str.replace(
                    "_",
                    " ",
                    regex=False,
                )
                .str.title()
            )

            st.subheader(
                "Grouped Incident Register"
            )

            display_columns = [
                "Incident Code",
                "Channel",
                "Severity",
                "Start Segment",
                "End Segment",
                "Duration",
                "Peak Risk",
                "Peak Confidence",
                "Top Feature",
                "Review Required",
                "Status",
            ]

            st.dataframe(
                display_incident_frame[
                    display_columns
                ].sort_values(
                    by=[
                        "Peak Risk",
                        "Duration",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                ),
                **stretch_width_kwargs(st.dataframe),
                hide_index=True,
                column_config={
                    "Peak Risk": (
                        st.column_config.ProgressColumn(
                            "Peak Risk",
                            min_value=0.0,
                            max_value=100.0,
                            format="%.2f",
                        )
                    ),
                    "Peak Confidence": (
                        st.column_config.ProgressColumn(
                            "Peak Confidence",
                            min_value=0.0,
                            max_value=100.0,
                            format="%.2f",
                        )
                    ),
                    "Duration": (
                        st.column_config.NumberColumn(
                            "Duration",
                            help=(
                                "Number of anomalous "
                                "segments grouped into "
                                "this incident."
                            ),
                        )
                    ),
                },
            )

            channel_summary = (
                filtered_incident_frame
                .groupby(
                    [
                        "Channel",
                        "Severity",
                    ],
                    as_index=False,
                )
                .agg(
                    Incidents=(
                        "Incident Code",
                        "count",
                    ),
                    Peak_Risk=(
                        "Peak Risk",
                        "max",
                    ),
                )
            )

            channel_chart = px.bar(
                channel_summary,
                x="Channel",
                y="Incidents",
                color="Severity",
                title=(
                    "Filtered Grouped Incidents "
                    "by Telemetry Channel"
                ),
                hover_data=[
                    "Peak_Risk",
                ],
                color_discrete_map={
                    "Watch": theme["info"],
                    "Warning": theme["warning"],
                    "Critical": theme["danger"],
                },
            )

            show_chart(
                channel_chart
            )

            st.subheader(
                "Inspect a Grouped Incident"
            )

            selected_incident_code = (
                st.selectbox(
                    "Incident",
                    filtered_incident_frame
                    .sort_values(
                        "Peak Risk",
                        ascending=False,
                    )[
                        "Incident Code"
                    ]
                    .tolist(),
                )
            )

            selected_incident = (
                incident_lookup[
                    selected_incident_code
                ]
            )

            selected_metadata = (
                safe_json_dictionary(
                    selected_incident.get(
                        "metadata"
                    )
                )
            )

            selected_segments = (
                safe_json_list(
                    selected_metadata.get(
                        "segment_ids"
                    )
                )
            )

            selected_channel = str(
                selected_metadata.get(
                    "channel",
                    "Unknown",
                )
            )

            selected_severity = str(
                selected_incident.get(
                    "severity",
                    "Watch",
                )
            )

            selected_duration = int(
                selected_incident.get(
                    "duration_samples"
                )
                or len(selected_segments)
                or 1
            )

            selected_peak_risk = float(
                selected_incident.get(
                    "peak_risk_score"
                )
                or 0.0
            )

            selected_peak_confidence = float(
                selected_incident.get(
                    "peak_confidence"
                )
                or 0.0
            )

            selected_status = str(
                selected_incident.get(
                    "status",
                    "open",
                )
            )

            detail_1, detail_2, detail_3, detail_4, detail_5 = (
                st.columns(5)
            )

            detail_1.metric(
                "Severity",
                selected_severity,
            )

            detail_2.metric(
                "Grouped Segments",
                selected_duration,
            )

            detail_3.metric(
                "Peak Risk",
                (
                    f"{selected_peak_risk:.2f}"
                    "/100"
                ),
            )

            detail_4.metric(
                "Peak Confidence",
                (
                    f"{selected_peak_confidence:.2f}"
                    "%"
                ),
            )

            detail_5.metric(
                "Status",
                selected_status.replace(
                    "_",
                    " ",
                ).title(),
            )

            start_segment = (
                selected_metadata.get(
                    "start_segment",
                    "Unknown",
                )
            )

            end_segment = (
                selected_metadata.get(
                    "end_segment",
                    "Unknown",
                )
            )

            top_feature = str(
                selected_incident.get(
                    "top_feature"
                )
                or "Unknown"
            )

            st.markdown(
                f"""
<div class="card">
<b>Incident code:</b>
{html.escape(selected_incident_code)}
<br>
<b>Channel:</b>
{html.escape(CHANNEL_NAMES.get(selected_channel, selected_channel))}
({html.escape(selected_channel)})
<br>
<b>Start segment:</b>
{html.escape(str(start_segment))}
<br>
<b>End segment:</b>
{html.escape(str(end_segment))}
<br>
<b>Top feature at peak:</b>
{html.escape(top_feature)}
<br>
<b>Grouping rule:</b>
Same channel with nearby sample indexes
<br>
<b>Interpretation:</b>
This incident groups related anomalous segments.
It is not a confirmed spacecraft hardware failure.
</div>
""",
                unsafe_allow_html=True,
            )

            incident_summary = str(
                selected_incident.get(
                    "summary"
                )
                or (
                    "No incident summary "
                    "was stored."
                )
            )

            st.subheader(
                "Incident Summary"
            )

            st.info(
                incident_summary
            )

            if bool(
                selected_incident.get(
                    "human_review_required"
                )
            ):
                st.warning(
                    "This grouped incident requires "
                    "human mission-control review."
                )

            st.subheader(
                "Operator Review Workflow"
            )

            current_incident_status = str(
                selected_incident.get(
                    "status",
                    "open",
                )
            ).strip().lower()

            incident_status_options = [
                "open",
                "under_review",
                "confirmed",
                "rejected",
                "resolved",
            ]

            try:
                current_status_index = (
                    incident_status_options.index(
                        current_incident_status
                    )
                )

            except ValueError:
                current_status_index = 0

            current_operator_name = str(
                selected_metadata.get(
                    "operator_name",
                    "",
                )
            )

            current_operator_note = str(
                selected_metadata.get(
                    "operator_note",
                    "",
                )
            )

            review_flash_key = (
                "incident_review_flash_"
                + selected_incident_code
            )

            review_flash_message = (
                st.session_state.pop(
                    review_flash_key,
                    None,
                )
            )

            if review_flash_message:
                st.success(
                    str(
                        review_flash_message
                    )
                )

            with st.form(
                key=(
                    "incident_review_form_"
                    + selected_incident_code
                )
            ):
                selected_review_status = (
                    st.selectbox(
                        "Incident Status",
                        options=(
                            incident_status_options
                        ),
                        index=(
                            current_status_index
                        ),
                        format_func=lambda value: (
                            value.replace(
                                "_",
                                " ",
                            ).title()
                        ),
                    )
                )

                operator_name = st.text_input(
                    "Operator Name",
                    value=current_operator_name,
                    placeholder=(
                        "Example: Mission Controller"
                    ),
                )

                operator_note = st.text_area(
                    "Operator Note",
                    value=current_operator_note,
                    height=130,
                    placeholder=(
                        "Describe the review decision, "
                        "observed evidence, and required action."
                    ),
                )

                save_operator_review = (
                    st.form_submit_button(
                        "Save Operator Review",
                        **stretch_width_kwargs(st.form_submit_button),
                    )
                )

            if save_operator_review:
                clean_operator_name = (
                    operator_name.strip()
                )

                clean_operator_note = (
                    operator_note.strip()
                )

                if not clean_operator_name:
                    st.error(
                        "Operator name is required."
                    )

                elif not clean_operator_note:
                    st.error(
                        "Operator note is required."
                    )

                else:
                    try:
                        incident_id = to_uuid(
                            selected_incident[
                                "id"
                            ]
                        )

                        updated_incident = update_incident_review(
                            incident_id=incident_id,
                            status=(
                                selected_review_status
                            ),
                            operator_name=(
                                clean_operator_name
                            ),
                            operator_note=(
                                clean_operator_note
                            ),
                        )

                        analysis_run_id = to_uuid(
                            updated_incident[
                                "analysis_run_id"
                            ]
                        )

                        refreshed_incidents = list_incidents(
                            analysis_run_id
                        )

                        refreshed_health = (
                            calculate_mission_health_score(
                                frame=analysis,
                                incidents=refreshed_incidents,
                            )
                        )

                        _ = update_analysis_run_mission_health(
                            analysis_run_id=analysis_run_id,
                            mission_health_score=float(
                                refreshed_health["score"]
                            ),
                            health_snapshot=refreshed_health,
                        )

                        st.session_state[
                            "opssat_database_incidents"
                        ] = refreshed_incidents

                        st.session_state[
                            review_flash_key
                        ] = (
                            "Incident review saved "
                            "successfully to PostgreSQL."
                        )

                        st.rerun()

                    except Exception as exc:
                        st.error(
                            "Incident review could not "
                            "be saved."
                        )

                        st.caption(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

            review_history = safe_json_list(
                selected_metadata.get(
                    "review_history"
                )
            )

            if review_history:
                st.subheader(
                    "Operator Review History"
                )

                review_history_frame = (
                    pd.DataFrame(
                        review_history
                    )
                )

                preferred_history_columns = [
                    "reviewed_at",
                    "operator_name",
                    "previous_status",
                    "new_status",
                    "operator_note",
                ]

                available_history_columns = [
                    column
                    for column
                    in preferred_history_columns
                    if column
                    in review_history_frame.columns
                ]

                if available_history_columns:
                    review_history_frame = (
                        review_history_frame[
                            available_history_columns
                        ]
                    )

                review_history_frame = (
                    review_history_frame.rename(
                        columns={
                            "reviewed_at": (
                                "Reviewed At"
                            ),
                            "operator_name": (
                                "Operator"
                            ),
                            "previous_status": (
                                "Previous Status"
                            ),
                            "new_status": (
                                "New Status"
                            ),
                            "operator_note": (
                                "Operator Note"
                            ),
                        }
                    )
                )

                st.dataframe(
                    review_history_frame.iloc[
                        ::-1
                    ],
                    **stretch_width_kwargs(st.dataframe),
                    hide_index=True,
                )

            else:
                st.info(
                    "No operator review has been "
                    "recorded for this incident yet."
                )

            if selected_segments:
                st.subheader(
                    "Grouped Segment IDs"
                )

                st.code(
                    ", ".join(
                        str(segment_id)
                        for segment_id
                        in selected_segments
                    ),
                    language="text",
                )

            peak_sample_index = (
                selected_metadata.get(
                    "peak_sample_index"
                )
            )

            peak_rows = pd.DataFrame()

            if (
                peak_sample_index is not None
                and "sample_index"
                in analysis.columns
            ):
                numeric_sample_indexes = (
                    pd.to_numeric(
                        analysis[
                            "sample_index"
                        ],
                        errors="coerce",
                    )
                )

                peak_rows = analysis.loc[
                    numeric_sample_indexes
                    == int(
                        peak_sample_index
                    )
                ]

            if (
                peak_rows.empty
                and start_segment
                != "Unknown"
            ):
                numeric_segments = (
                    pd.to_numeric(
                        analysis[
                            "segment"
                        ],
                        errors="coerce",
                    )
                )

                try:
                    start_segment_number = int(
                        start_segment
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    start_segment_number = None

                if start_segment_number is not None:
                    peak_rows = analysis.loc[
                        numeric_segments
                        == start_segment_number
                    ]

            if not peak_rows.empty:
                peak_row = (
                    peak_rows
                    .sort_values(
                        "hybrid_score",
                        ascending=False,
                    )
                    .iloc[0]
                )

                st.subheader(
                    "Peak Segment Evidence"
                )

                evidence_1, evidence_2, evidence_3, evidence_4 = (
                    st.columns(4)
                )

                evidence_1.metric(
                    "Segment",
                    int(
                        peak_row[
                            "segment"
                        ]
                    ),
                )

                evidence_2.metric(
                    "Hybrid Risk",
                    (
                        f"{float(peak_row['hybrid_score']):.2f}"
                        "/100"
                    ),
                )

                evidence_3.metric(
                    "Supervised Score",
                    (
                        f"{float(peak_row['supervised_score']):.2f}"
                        "%"
                    ),
                )

                evidence_4.metric(
                    "Decision Margin",
                    (
                        f"{float(peak_row['decision_margin']):.2f}"
                        "/100"
                    ),
                )

                st.info(
                    str(
                        peak_row[
                            "explanation"
                        ]
                    )
                )

                contributions = (
                    peak_row[
                        "feature_contributions"
                    ]
                )

                if isinstance(
                    contributions,
                    str,
                ):
                    try:
                        contributions = json.loads(
                            contributions.replace(
                                "'",
                                '"',
                            )
                        )

                    except json.JSONDecodeError:
                        contributions = {}

                if isinstance(
                    contributions,
                    dict,
                ) and contributions:
                    contribution_frame = (
                        pd.DataFrame(
                            {
                                "Feature": list(
                                    contributions.keys()
                                ),
                                "Contribution": list(
                                    contributions.values()
                                ),
                            }
                        )
                        .sort_values(
                            "Contribution",
                            ascending=True,
                        )
                        .tail(12)
                    )

                    contribution_chart = px.bar(
                        contribution_frame,
                        x="Contribution",
                        y="Feature",
                        orientation="h",
                        text="Contribution",
                        title=(
                            "Peak Segment Feature "
                            "Contribution"
                        ),
                    )

                    contribution_chart.update_traces(
                        texttemplate=(
                            "%{text:.1f}%"
                        ),
                        textposition="outside",
                        marker_color=(
                            theme["info"]
                        ),
                    )

                    show_chart(
                        contribution_chart
                    )

            else:
                st.info(
                    "The peak prediction row could "
                    "not be matched to the current "
                    "analysis DataFrame."
                )

    else:
        st.caption(
            "Select Latest PostgreSQL OPS-SAT Run "
            "to inspect grouped database incidents. "
            "The current source supports individual "
            "segment explanation."
        )

        candidates = (
            analysis
            .sort_values(
                "hybrid_score",
                ascending=False,
            )[
                "segment"
            ]
            .astype(int)
            .tolist()
        )

        selected_segment = st.selectbox(
            "Select a segment to explain",
            candidates,
        )

        row = analysis[
            analysis[
                "segment"
            ].astype(int)
            == int(
                selected_segment
            )
        ].iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Predicted Status",
            row["risk_level"],
        )

        c2.metric(
            "Hybrid Risk",
            f"{row['hybrid_score']:.1f}/100",
        )

        c3.metric(
            "Anomaly Probability",
            f"{row['supervised_score']:.1f}%",
        )

        c4.metric(
            "Decision Margin",
            f"{row['decision_margin']:.1f}/100",
        )

        st.subheader(
            "Observed Evidence"
        )

        st.info(
            str(
                row["explanation"]
            )
        )

        contributions = row[
            "feature_contributions"
        ]

        if isinstance(
            contributions,
            str,
        ):
            contributions = json.loads(
                contributions.replace(
                    "'",
                    '"',
                )
            )

        contribution_frame = (
            pd.DataFrame(
                {
                    "Feature": list(
                        contributions.keys()
                    ),
                    "Contribution": list(
                        contributions.values()
                    ),
                }
            )
            .sort_values(
                "Contribution",
                ascending=True,
            )
            .tail(12)
        )

        st.caption("Local Engineered-Feature Deviation Evidence")

        fig = px.bar(
            contribution_frame,
            x="Contribution",
            y="Feature",
            orientation="h",
            text="Contribution",
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_color=theme["info"],
        )

        show_chart(
            fig
        )

        st.subheader(
            "Decision Context"
        )

        st.markdown(
            f"""
<div class="card">
<b>Channel:</b>
{html.escape(CHANNEL_NAMES.get(str(row['channel']), str(row['channel'])))}
({html.escape(str(row['channel']))})
<br>
<b>Top feature:</b>
{html.escape(str(row['top_feature']))}
<br>
<b>Isolation score:</b>
{float(row['isolation_score']):.1f}/100
<br>
<b>Supervised score:</b>
{float(row['supervised_score']):.1f}/100
<br>
<b>Human interpretation:</b>
This is a segment-level anomaly decision.
It does not identify a confirmed hardware root cause.
</div>
""",
            unsafe_allow_html=True,
        )


elif page == "Upload & Test":
    st.write(
        "The website validates every upload before prediction. It accepts either the official raw "
        "`segments.csv` structure or the official engineered `dataset.csv` structure. Ground-truth "
        "labels, when present, are used only after prediction for evaluation."
    )

    if source_mode == "Upload Real OPSSAT CSV" and upload_validation is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Input Type", upload_validation.kind.replace("_", " ").title())
        c2.metric("Rows Accepted", f"{upload_validation.rows:,}")
        c3.metric("Segments", f"{upload_validation.segments:,}")
        c4.metric("Channels", f"{upload_validation.channels:,}")
        c5.metric("Label Coverage", f"{upload_validation.label_coverage:.0%}")

        if upload_validation.removed_rows:
            st.warning(f"Validation removed {upload_validation.removed_rows} invalid or duplicate row(s).")
        for message in upload_validation.messages:
            st.warning(message)
        for warning in compatibility["warnings"]:
            st.warning(warning)

        st.subheader("Upload Compatibility")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Schema & Channel Status", compatibility["status"])
        d2.metric("Known Channel Coverage", f"{compatibility['known_channel_ratio']:.0%}")
        d3.metric("Drift Compatibility", drift_summary["compatibility"])
        d4.metric("Drift Score", f"{drift_summary['overall_score']:.2f}")

        st.subheader(
            "PostgreSQL Persistence"
        )

        upload_database_result = (
            st.session_state[
                "opssat_upload_save_result"
            ]
        )

        upload_is_persisted = (
            isinstance(
                upload_database_result,
                dict,
            )
            and upload_signature is not None
            and upload_database_result.get(
                "upload_signature"
            )
            == upload_signature
        )

        if upload_is_persisted:
            st.success(
                "The original CSV, normalized features, "
                "telemetry session, predictions, and "
                "grouped incidents are persisted."
            )

            persistence_1, persistence_2, persistence_3 = (
                st.columns(3)
            )

            persistence_1.metric(
                "Saved Predictions",
                int(
                    upload_database_result[
                        "total_predictions"
                    ]
                ),
            )

            persistence_2.metric(
                "Detected Anomalies",
                int(
                    upload_database_result[
                        "total_anomalies"
                    ]
                ),
            )

            persistence_3.metric(
                "Grouped Incidents",
                int(
                    upload_database_result[
                        "total_incidents"
                    ]
                ),
            )

            st.caption(
                "Dataset ID: "
                + str(
                    upload_database_result[
                        "dataset_id"
                    ]
                )
                + " · Session ID: "
                + str(
                    upload_database_result[
                        "telemetry_session_id"
                    ]
                )
                + " · Run ID: "
                + str(
                    upload_database_result[
                        "analysis_run_id"
                    ]
                )
            )

        else:
            st.info(
                "Use **Save Upload & Analysis to PostgreSQL** "
                "in the sidebar to persist this validated upload."
            )

        st.subheader("Upload Predictions")
        display_columns = [
            "segment", "channel", "prediction_label", "risk_level", "hybrid_score",
            "decision_margin", "top_feature", "anomaly",
        ]
        st.dataframe(analysis[display_columns], **stretch_width_kwargs(st.dataframe), hide_index=True)

        if row_evaluation is not None:
            st.subheader("Ground-Truth Segment Evaluation")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Precision", f"{row_evaluation['Precision']:.3f}")
            r2.metric("Recall", f"{row_evaluation['Recall']:.3f}")
            r3.metric("F1", f"{row_evaluation['F1']:.3f}")
            r4.metric("False Alarms", int(row_evaluation["False Alarms"]))
            r5.metric("Missed Anomalies", int(row_evaluation["Missed Anomalies"]))

            matrix = np.array(
                [
                    [row_evaluation["TN"], row_evaluation["FP"]],
                    [row_evaluation["FN"], row_evaluation["TP"]],
                ],
                dtype=int,
            )
            fig = px.imshow(
                matrix,
                text_auto=True,
                x=["Predicted Normal", "Predicted Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                title="Uploaded Data Confusion Matrix",
                color_continuous_scale="Blues",
            )
            show_chart(fig)

            metric_table = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC",
                        "PR-AUC", "ROC-AUC", "False Alarms / 1000",
                    ],
                    "Value": [
                        row_evaluation["Accuracy"], row_evaluation["Balanced Accuracy"],
                        row_evaluation["Precision"], row_evaluation["Recall"], row_evaluation["F1"],
                        row_evaluation["MCC"], row_evaluation["PR-AUC"], row_evaluation["ROC-AUC"],
                        row_evaluation["False Alarms / 1000"],
                    ],
                }
            )
            st.dataframe(metric_table, **stretch_width_kwargs(st.dataframe), hide_index=True)
        else:
            st.info(
                "Ground-truth labels are unavailable. Predictions and anomaly scores are shown, "
                "but accuracy, confusion-matrix, and event-detection metrics cannot be calculated."
            )

        if event_evaluation is not None:
            st.subheader("Event-Based Evaluation")
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("True Events", event_evaluation["True Events"])
            e2.metric("Detected Events", event_evaluation["Detected Events"])
            e3.metric("Missed Events", event_evaluation["Missed Events"])
            e4.metric("False Alert Events", event_evaluation["False Alert Events"])
            e5.metric("Event F1", f"{event_evaluation['Event F1']:.3f}")
            if not event_ledger.empty:
                st.dataframe(event_ledger, **stretch_width_kwargs(st.dataframe), hide_index=True)

        with st.expander("Telemetry drift details"):
            for note in drift_summary["notes"]:
                st.caption(note)
            st.dataframe(drift_details, **stretch_width_kwargs(st.dataframe), hide_index=True)
    else:
        st.info("Select **Upload Real OPSSAT CSV** in the sidebar to test your own real data.")

    st.subheader("Ready-to-Upload Real Test Files")
    samples = [
        ("Normal real segment", "opssat_real_normal.csv"),
        ("Anomalous real segment", "opssat_real_anomaly.csv"),
        ("Normal → anomaly → recovery", "opssat_real_mixed.csv"),
        ("Magnetometer anomaly set", "opssat_real_magnetometer_anomalies.csv"),
        ("Photodiode anomaly set", "opssat_real_photodiode_anomalies.csv"),
    ]
    for row_start in range(0, len(samples), 3):
        sample_columns = st.columns(3)
        for column, (label, filename) in zip(
            sample_columns,
            samples[row_start: row_start + 3],
            strict=False,
        ):
            path = SAMPLES_DIR / filename
            with column:
                if path.exists():
                    st.download_button(
                        label=f"Download {label}",
                        data=path.read_bytes(),
                        file_name=filename,
                        mime="text/csv",
                        **stretch_width_kwargs(st.download_button),
                    )

elif page == "Model Validation":
    st.header("Model Validation on the Official Test Split")
    st.write(
        "Thresholds were selected only inside the official training partition. The packaged results below are then calculated once on the unseen official OPSSAT-AD test split."
    )
    metric_columns = ["Model", "Precision", "Recall", "F1", "MCC", "PR-AUC", "ROC-AUC", "False Alarms / 1000", "Threshold"]
    st.dataframe(official_metrics[metric_columns], **stretch_width_kwargs(st.dataframe), hide_index=True)

    chart_data = official_metrics.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1", "MCC", "PR-AUC", "ROC-AUC"],
        var_name="Metric",
        value_name="Value",
    )
    fig = px.bar(
        chart_data,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="OPSSAT-AD Official Test Metrics",
        range_y=[0, 1.05],
    )
    show_chart(fig)

    hybrid = official_metrics[official_metrics["Model"] == "OPSSAT Hybrid"].iloc[0]
    matrix = np.array([[hybrid["TN"], hybrid["FP"]], [hybrid["FN"], hybrid["TP"]]], dtype=int)
    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted Normal", "Predicted Anomaly"],
        y=["Actual Normal", "Actual Anomaly"],
        title="Hybrid Confusion Matrix",
        color_continuous_scale="Blues",
    )
    show_chart(fig)

    train_counts = official_dataset.groupby(["train", "anomaly"]).size().reset_index(name="Segments")
    train_counts["Split"] = train_counts["train"].map({1: "Official Train", 0: "Official Test"})
    train_counts["Label"] = train_counts["anomaly"].map({1: "Anomaly", 0: "Normal"})
    st.caption("Official Split and Label Distribution")

    fig = px.bar(
        train_counts,
        x="Split",
        y="Segments",
        color="Label",
        barmode="group",
        color_discrete_map={"Normal": theme["success"], "Anomaly": theme["danger"]},
    )
    show_chart(fig)

    st.subheader("Official Test Event-Based Evaluation")
    if event_evaluation is not None:
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("True Events", event_evaluation["True Events"])
        e2.metric("Detected Events", event_evaluation["Detected Events"])
        e3.metric("Missed Events", event_evaluation["Missed Events"])
        e4.metric("False Alert Events", event_evaluation["False Alert Events"])
        e5.metric("Event F1", f"{event_evaluation['Event F1']:.3f}")
        if not event_ledger.empty:
            st.dataframe(event_ledger, **stretch_width_kwargs(st.dataframe), hide_index=True)

    with st.expander("Reproducibility metadata"):
        metadata_text = json.dumps(
            official_metadata,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        # Use Streamlit's native code component so JSON newlines and
        # indentation are preserved reliably in both Light and Dark modes.
        st.code(
            metadata_text,
            language="text",
            line_numbers=False,
            wrap_lines=False,
        )

elif page == "Data Drift Monitor":
    st.write(
        "This monitor compares the current engineered telemetry distribution with the nominal "
        "official-training envelope. Drift indicates a change in data distribution; it is not, by itself, "
        "proof of a spacecraft anomaly or hardware fault."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compatibility", drift_summary["compatibility"])
    c2.metric("Overall Drift Score", f"{drift_summary['overall_score']:.2f}")
    c3.metric("High-Drift Features", drift_summary["high_features"])
    c4.metric("Moderate-Drift Features", drift_summary["moderate_features"])

    for note in drift_summary["notes"]:
        st.info(note)

    top_drift = drift_details.head(12).sort_values("Drift Score", ascending=True)
    fig = px.bar(
        top_drift,
        x="Drift Score",
        y="Feature",
        color="Level",
        orientation="h",
        title="Largest Distribution Shifts vs. Nominal Training Data",
        color_discrete_map={
            "Stable": theme["success"],
            "Moderate": theme["warning"],
            "High": theme["danger"],
        },
    )
    show_chart(fig)
    st.dataframe(drift_details, **stretch_width_kwargs(st.dataframe), hide_index=True)

    if drift_summary["affected_features"]:
        st.warning("Affected features: " + ", ".join(drift_summary["affected_features"][:12]))
    else:
        st.success("No engineered feature crossed the moderate-drift threshold.")

elif page == "Reports & Responsible AI":
    report_lines = [
        "MISSIONGUARD AI — REAL ESA OPS-SAT ANALYSIS",
        "=" * 58,
        f"Data source: {source_name}",
        f"Segments analyzed: {len(analysis)}",
        f"Predicted anomalies: {summary['anomaly_count']}",
        f"Anomaly rate: {summary['anomaly_rate']:.2f}%",
        f"Peak hybrid risk: {summary['peak_score']:.2f}/100",
        f"Overall status: {summary['status']}",
        f"Mission health score: {float(mission_health['score']):.2f}/100",
        f"Mission health status: {mission_health['status']}",
        f"Telemetry stability component: {float(mission_health['telemetry_stability']):.2f}/100",
        f"Anomaly control component: {float(mission_health['anomaly_control']):.2f}/100",
        f"Peak-risk resilience component: {float(mission_health['peak_resilience']):.2f}/100",
        f"Incident readiness component: {float(mission_health['incident_readiness']):.2f}/100",
        f"Incidents considered: {int(mission_health['incidents_considered'])}",
        f"Unresolved incidents: {int(mission_health['unresolved_incidents'])}",
        f"Data compatibility: {drift_summary['compatibility']}",
        f"Drift score: {drift_summary['overall_score']:.2f}",
        "",
        "HIGHEST-RISK SEGMENTS",
        "-" * 58,
    ]
    for _, row in analysis.sort_values("hybrid_score", ascending=False).head(15).iterrows():
        report_lines.extend(
            [
                f"Segment {int(row['segment'])} | {row['channel']} | {row['risk_level']}",
                f"Hybrid risk: {row['hybrid_score']:.2f}/100 | Decision margin: {row['decision_margin']:.1f}/100",
                f"Evidence: {row['explanation']}",
                "",
            ]
        )
    if row_evaluation is not None:
        report_lines.extend(
            [
                "GROUND-TRUTH SEGMENT EVALUATION",
                "-" * 58,
                f"Precision: {row_evaluation['Precision']:.4f}",
                f"Recall: {row_evaluation['Recall']:.4f}",
                f"F1: {row_evaluation['F1']:.4f}",
                f"False alarms: {row_evaluation['False Alarms']}",
                f"Missed anomalies: {row_evaluation['Missed Anomalies']}",
                "",
            ]
        )
    if event_evaluation is not None:
        report_lines.extend(
            [
                "EVENT-BASED EVALUATION",
                "-" * 58,
                f"True events: {event_evaluation['True Events']}",
                f"Detected events: {event_evaluation['Detected Events']}",
                f"Missed events: {event_evaluation['Missed Events']}",
                f"False alert events: {event_evaluation['False Alert Events']}",
                f"Event F1: {event_evaluation['Event F1']:.4f}",
                "",
            ]
        )
    report_lines.extend(
        [
            "RESPONSIBLE-AI NOTICE",
            "-" * 58,
            "This educational decision-support prototype predicts segment-level anomalies.",
            "The Mission Health Score is an internal weighted index, not a certified flight-health metric.",
            "It does not prove a hardware root cause and must not replace spacecraft operators.",
            "Ground-truth labels are used only for evaluation, not as model inputs.",
        ]
    )
    text_report = "\n".join(report_lines)
    html_report = "<html><body><pre>" + html.escape(text_report) + "</pre></body></html>"

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download TXT report", text_report, "missionguard_opssat_report.txt", "text/plain", **stretch_width_kwargs(st.download_button))
    c2.download_button("Download HTML report", html_report, "missionguard_opssat_report.html", "text/html", **stretch_width_kwargs(st.download_button))
    c3.download_button(
        "Download analyzed CSV",
        analysis.to_csv(index=False).encode("utf-8"),
        "missionguard_opssat_predictions.csv",
        "text/csv",
        **stretch_width_kwargs(st.download_button),
    )

    with st.expander("Preview text report"):
        st.code(text_report, language="text")

    st.subheader("Responsible-AI Guardrails")
    st.markdown(
        """
- The application reports **anomaly evidence**, not confirmed causal diagnoses.
- The Mission Health Score is a transparent prototype composite, not a certified spacecraft-health measurement.
- Unknown channels are handled explicitly and never renamed as battery, engine, or radiation sensors without documentation.
- Official test labels are held out from training and used only to measure performance.
- A human operator must review high-risk or low-margin decisions before any operational action.
- The dataset contains selected OPS-SAT channels and should not be treated as a complete spacecraft health model.
"""
    )

elif page == "Dataset & Attribution":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feature Segments", f"{len(official_dataset):,}")
    c2.metric("Raw Samples", f"{len(official_segments):,}")
    c3.metric("Telemetry Channels", official_dataset["channel"].nunique())
    c4.metric("Official Test Segments", int((official_dataset["train"] == 0).sum()))

    st.markdown(
        """
### OPSSAT-AD v2

MissionGuard AI uses the OPSSAT-AD benchmark containing real telemetry acquired aboard ESA's OPS-SAT CubeSat. The project includes both the original manually split telemetry segments and the corresponding engineered segment-feature table.

- **Dataset DOI:** `10.5281/zenodo.15108715`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Use in this project:** model training, official held-out testing, real CSV upload examples, and signal visualization
"""
    )

    channel_frame = pd.DataFrame(
        {
            "Channel ID": sorted(official_dataset["channel"].unique()),
            "Documented Signal": [CHANNEL_NAMES.get(channel, "Unknown") for channel in sorted(official_dataset["channel"].unique())],
            "Segments": [int((official_dataset["channel"] == channel).sum()) for channel in sorted(official_dataset["channel"].unique())],
        }
    )
    st.dataframe(channel_frame, **stretch_width_kwargs(st.dataframe), hide_index=True)

    dataset_card_path = PROJECT_ROOT / "data" / "DATASET_CARD.md"
    if dataset_card_path.exists():
        with st.expander("Full dataset card"):
            st.markdown(dataset_card_path.read_text(encoding="utf-8"))

    with st.expander("Required feature schema"):
        st.code(
            "segment, anomaly, train, channel, sampling, duration, len, mean, var, std, "
            "kurtosis, skew, n_peaks, smooth10_n_peaks, smooth20_n_peaks, diff_peaks, "
            "diff2_peaks, diff_var, diff2_var, gaps_squared, len_weighted, "
            "var_div_duration, var_div_len",
            language="text",
        )

elif page == "Team & Contact":
    youssef_photo = image_data_uri(YOUSSEF_PHOTO_PATH)
    shereen_photo = image_data_uri(SHEREEN_PHOTO_PATH)

    st.markdown(
        """
<div class="mg-team-intro">
  <div>
    <span class="mg-section-no">Mission team</span>
    <h3>Engineering, intelligence, and visual communication in one multidisciplinary team.</h3>
  </div>
  <div>
    <p>We combine AI and data science with front-end engineering and graphic design so advanced mission analytics remain accurate, understandable, and visually memorable.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="mg-team-grid">
  <article class="mg-team-card">
    <div class="mg-team-photo-wrap">
      <img class="mg-team-photo" src="{youssef_photo}" alt="Portrait of Youssef Osama Soliman">
    </div>
    <div class="mg-team-info">
      <span class="mg-team-role">Co-Creator / AI & Data Science</span>
      <h2>Youssef<br>Osama Soliman</h2>
      <p>AI & Data Science Engineer focused on machine learning, analytics, and building data-driven solutions. Also contributes front-end development and graphic design to the MissionGuard experience.</p>
      <div class="mg-skill-cloud">
        <span>Machine Learning</span><span>Analytics</span><span>Python</span><span>SQL</span><span>Power BI</span><span>Front-End</span><span>Graphic Design</span>
      </div>
      <a class="mg-contact-link" href="mailto:yousef.osama.salem@gmail.com">yousef.osama.salem@gmail.com ↗</a>
    </div>
  </article>

  <article class="mg-team-card">
    <div class="mg-team-photo-wrap">
      <img class="mg-team-photo shereen" src="{shereen_photo}" alt="Portrait of Shereen Ahmed Hazem">
    </div>
    <div class="mg-team-info">
      <span class="mg-team-role">Co-Creator / AI Research & Design</span>
      <h2>Shereen<br>Ahmed Hazem</h2>
      <p>AI & Data Science Engineer working across generative AI, AI research, data analysis, and front-end development. Leads the project's senior graphic-design direction and visual storytelling.</p>
      <div class="mg-skill-cloud">
        <span>Generative AI</span><span>AI Research</span><span>Python</span><span>SQL</span><span>Power BI</span><span>Front-End</span><span>Senior Graphic Design</span>
      </div>
      <a class="mg-contact-link" href="mailto:shereensoliman142@gmail.com">shereensoliman142@gmail.com ↗</a>
    </div>
  </article>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<section class="mg-proof-band">
  <div class="mg-proof-copy">
    <span class="mg-eyebrow">Shared mission</span>
    <h2>Make complex AI<br>clear enough to act on.</h2>
    <p>Our goal is not only to detect unusual spacecraft behavior, but to communicate the evidence, uncertainty, and next review step in a way mission teams can trust.</p>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

elif page == "IBM Bob Evidence":
    log_path = PROJECT_ROOT / "docs" / "ibm-bob-development-log.md"
    if log_path.exists():
        st.markdown(log_path.read_text(encoding="utf-8"))
    else:
        st.info("Add documented IBM Bob prompts and verified development outcomes to `docs/ibm-bob-development-log.md`.")

st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """
<div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;padding:.6rem 0 1.4rem;color:var(--mg-muted);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;">
  <span>MissionGuard AI / Real OPSSAT-AD telemetry</span>
  <span>Educational research prototype — not certified flight software</span>
</div>
""",
    unsafe_allow_html=True,
)