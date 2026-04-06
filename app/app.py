"""
Traffic Accident Severity Analysis Dashboard
Professional analytical dashboard + schema-safe inference + normalized severity labels

Severity convention used across the entire app:
- Displayed severity is ALWAYS 1 to 4
- If a model exposes internal classes 0 to 3 (common with XGBoost), they are mapped to 1 to 4
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Accident Severity Analysis Dashboard",
    page_icon="🚦",
    layout="wide",
)

MODEL_PATHS = {
    "XGBoost": "models/xgboost_tuned_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Decision Tree": "models/decision_tree_model.pkl",
}

FEATURE_COLUMNS_PATH = "test_train/feature_columns.pkl"

SEVERITY_LABELS = {
    1: "Minor",
    2: "Moderate",
    3: "Serious",
    4: "Severe",
}

SEVERITY_COLORS = {
    1: "#41b37d",
    2: "#f0a43a",
    3: "#f06b42",
    4: "#d6455d",
}

DAY_TO_NUM = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

SCENARIOS = {
    "Custom": None,
    "Low-risk daytime urban": {
        "Start_Lat": 34.05,
        "Start_Lng": -118.24,
        "Distance_mi": 0.2,
        "Hour": 13,
        "Month": 5,
        "Day_of_Week": "Tuesday",
        "Temperature_F": 72,
        "Humidity_pct": 50,
        "Pressure_in": 30.1,
        "Visibility_mi": 10.0,
        "Wind_Speed_mph": 4,
        "Weather_Condition": "Clear",
        "Sunrise_Sunset": "Day",
        "Traffic_Signal": True,
        "Junction": False,
        "Crossing": False,
        "Stop": False,
        "Amenity": False,
        "Bump": False,
        "Give_Way": False,
        "Railway": False,
        "Roundabout": False,
        "Station": False,
    },
    "Busy rush-hour conflict point": {
        "Start_Lat": 29.76,
        "Start_Lng": -95.37,
        "Distance_mi": 1.0,
        "Hour": 17,
        "Month": 11,
        "Day_of_Week": "Friday",
        "Temperature_F": 66,
        "Humidity_pct": 70,
        "Pressure_in": 29.8,
        "Visibility_mi": 6.0,
        "Wind_Speed_mph": 9,
        "Weather_Condition": "Rain",
        "Sunrise_Sunset": "Night",
        "Traffic_Signal": False,
        "Junction": True,
        "Crossing": True,
        "Stop": False,
        "Amenity": False,
        "Bump": False,
        "Give_Way": False,
        "Railway": False,
        "Roundabout": False,
        "Station": False,
    },
    "High-risk winter highway": {
        "Start_Lat": 45.00,
        "Start_Lng": -110.00,
        "Distance_mi": 3.8,
        "Hour": 22,
        "Month": 1,
        "Day_of_Week": "Saturday",
        "Temperature_F": 18,
        "Humidity_pct": 84,
        "Pressure_in": 29.1,
        "Visibility_mi": 1.2,
        "Wind_Speed_mph": 28,
        "Weather_Condition": "Snow",
        "Sunrise_Sunset": "Night",
        "Traffic_Signal": False,
        "Junction": True,
        "Crossing": False,
        "Stop": False,
        "Amenity": False,
        "Bump": False,
        "Give_Way": False,
        "Railway": False,
        "Roundabout": False,
        "Station": False,
    },
    "Foggy unsignalized junction": {
        "Start_Lat": 39.73,
        "Start_Lng": -104.99,
        "Distance_mi": 1.8,
        "Hour": 6,
        "Month": 12,
        "Day_of_Week": "Monday",
        "Temperature_F": 34,
        "Humidity_pct": 92,
        "Pressure_in": 29.5,
        "Visibility_mi": 0.7,
        "Wind_Speed_mph": 7,
        "Weather_Condition": "Fog",
        "Sunrise_Sunset": "Night",
        "Traffic_Signal": False,
        "Junction": True,
        "Crossing": True,
        "Stop": True,
        "Amenity": False,
        "Bump": False,
        "Give_Way": True,
        "Railway": False,
        "Roundabout": False,
        "Station": False,
    },
}

PERFORMANCE_DATA = pd.DataFrame(
    {
        "Model": ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"],
        "Accuracy": [0.813, 0.666, 0.653, 0.649],
        "F1-Macro": [0.479, 0.428, 0.403, 0.329],
        "ROC-AUC": [0.880, 0.849, 0.813, 0.717],
    }
)

TOP_FEATURES = pd.DataFrame(
    {
        "Feature": ["Distance(mi)", "Month", "Season_Winter", "Start_Lng", "Start_Lat"],
        "Importance": [0.8996, 0.4756, 0.2605, 0.2028, 0.1720],
        "Theme": ["Crash extent", "Seasonality", "Winter effect", "Geographic", "Geographic"],
    }
)

WEATHER_INSIGHT = pd.DataFrame(
    {
        "Weather": ["Clear/Fair", "Overcast", "Rain", "Fog"],
        "Severity 3+ Rate (%)": [30.0, 29.5, 21.0, 10.0],
    }
)


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(69, 104, 220, 0.18), transparent 24%),
            radial-gradient(circle at top left, rgba(176, 106, 179, 0.18), transparent 22%),
            linear-gradient(180deg, #0f172a 0%, #111827 35%, #0b1220 100%);
        color: #e5e7eb;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero {
        background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(168,85,247,0.20));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    }

    .glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        text-align: left;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .metric-label {
        font-size: 0.88rem;
        color: #cbd5e1;
    }

    .pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        color: white;
    }

    .result-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.11);
        border-radius: 22px;
        padding: 1.15rem;
        box-shadow: 0 16px 38px rgba(0,0,0,0.24);
    }

    .muted {
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px 16px;
        color: #e5e7eb;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,0.20) !important;
    }

    .stMarkdown, .stText, .stCaption {
        color: #e5e7eb;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.9rem;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def canonicalize(name: str) -> str:
    s = str(name).strip()
    s = s.replace("%", "pct").replace("°", "")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_").replace(".", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x in ("", None):
            return default
        return int(x)
    except Exception:
        return default


def to_binary(x: Any) -> int:
    return int(bool(x))


@st.cache_data
def load_feature_columns() -> List[str]:
    cols = joblib.load(FEATURE_COLUMNS_PATH)
    if isinstance(cols, (pd.Index, np.ndarray)):
        cols = cols.tolist()
    return list(cols)


@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODEL_PATHS[model_name])


def model_features(model, fallback_features: List[str]) -> List[str]:
    return list(getattr(model, "feature_names_in_", fallback_features))


def feature_lookup(columns: List[str]) -> Dict[str, str]:
    return {canonicalize(c): c for c in columns}


def first_present(lookup: Dict[str, str], *names: str) -> Optional[str]:
    for name in names:
        key = canonicalize(name)
        if key in lookup:
            return lookup[key]
    return None


def set_feature(row: Dict[str, float], lookup: Dict[str, str], names: List[str], value: Any) -> None:
    col = first_present(lookup, *names)
    if col is not None:
        row[col] = value


def infer_onehot_groups(columns: List[str], prefixes: List[str]) -> Dict[str, List[str]]:
    out = {p: [] for p in prefixes}
    canon_prefixes = {p: canonicalize(p) + "_" for p in prefixes}
    for col in columns:
        c = canonicalize(col)
        for p, pref in canon_prefixes.items():
            if c.startswith(pref):
                out[p].append(col)
    return {k: v for k, v in out.items() if v}


def season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    return "Fall"


def engineered_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    hour = safe_int(raw["Hour"], 12)
    month = safe_int(raw["Month"], 1)
    dow = DAY_TO_NUM[raw["Day_of_Week"]]
    season = season_from_month(month)

    traffic_signal = to_binary(raw["Traffic_Signal"])
    crossing = to_binary(raw["Crossing"])
    stop = to_binary(raw["Stop"])
    junction = to_binary(raw["Junction"])
    
    # Visibility categories
    visibility = safe_float(raw.get("Visibility_mi", 10.0))
    
    # Time period (Morning, Afternoon, Evening, Night)
    if 6 <= hour < 12:
        time_period = "Morning"
    elif 12 <= hour < 18:
        time_period = "Afternoon"
    elif 18 <= hour < 22:
        time_period = "Evening"
    else:
        time_period = "Night"

    return {
        "Day_of_Week_Num": dow,
        "Rush_Hour": int(hour in {7, 8, 15, 16, 17}),
        "Is_Weekend": int(dow >= 5),
        "Night_Driving": int(hour in {20, 21, 22, 23, 0, 1, 2, 3, 4, 5}),
        "Clear_Weather": int(raw["Weather_Condition"] in {"Clear", "Fair", "Partly Cloudy"}),
        "Season": season,
        "Season_Winter": int(season == "Winter"),
        "Season_Spring": int(season == "Spring"),
        "Season_Summer": int(season == "Summer"),
        "Infrastructure_Safety_Score": traffic_signal + crossing + stop,
        "High_Risk_Junction": int(junction == 1 and traffic_signal == 0),
        "Visibility_Poor": int(visibility < 1.0),
        "Visibility_Good": int(visibility >= 5.0),
        "Time_Period_Morning": int(time_period == "Morning"),
        "Time_Period_Afternoon": int(time_period == "Afternoon"),
        "Time_Period_Evening": int(time_period == "Evening"),
    }


def build_input_frame(raw: Dict[str, Any], expected_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lookup = feature_lookup(expected_cols)
    row: Dict[str, float] = {col: 0.0 for col in expected_cols}

    # Get raw numeric values
    distance_mi = safe_float(raw["Distance_mi"])
    visibility_mi = safe_float(raw["Visibility_mi"])
    wind_speed_mph = safe_float(raw["Wind_Speed_mph"])
    precipitation_in = safe_float(raw.get("Precipitation_in", 0.0))

    direct_map = {
        "Start_Lat": ["Start_Lat", "StartLat"],
        "Start_Lng": ["Start_Lng", "StartLng"],
        "Distance_mi": ["Distance_mi", "Distance(mi)", "Distance"],
        "Distance_log": ["Distance_log"],  # Log-transformed distance
        "Temperature_F": ["Temperature_F", "Temperature(F)", "Temperature"],
        "Humidity_pct": ["Humidity_pct", "Humidity(%)", "Humidity"],
        "Pressure_in": ["Pressure_in", "Pressure(in)", "Pressure"],
        "Visibility_mi": ["Visibility_mi", "Visibility(mi)", "Visibility"],
        "Visibility_log": ["Visibility_log"],  # Log-transformed visibility
        "Wind_Speed_mph": ["Wind_Speed_mph", "Wind_Speed(mph)", "Wind_Speed"],
        "Wind_Speed_log": ["Wind_Speed_log"],  # Log-transformed wind speed
        "Precipitation_in": ["Precipitation_in", "Precipitation(in)", "Precipitation"],
        "Precipitation_log": ["Precipitation_log"],  # Log-transformed precipitation
        "Hour": ["Hour"],
        "Month": ["Month"],
        "Day": ["Day"],
        "Year": ["Year"],
        "Traffic_Signal": ["Traffic_Signal"],
        "Junction": ["Junction"],
        "Crossing": ["Crossing"],
        "Stop": ["Stop"],
        "Amenity": ["Amenity"],
        "Bump": ["Bump"],
        "Give_Way": ["Give_Way", "GiveWay"],
        "Railway": ["Railway"],
        "Roundabout": ["Roundabout"],
        "Station": ["Station"],
    }

    for raw_key, aliases in direct_map.items():
        if raw_key in raw:
            value = raw[raw_key]
        elif raw_key == "Distance_log":
            value = np.log1p(distance_mi)
        elif raw_key == "Visibility_log":
            value = np.log1p(visibility_mi)
        elif raw_key == "Wind_Speed_log":
            value = np.log1p(wind_speed_mph)
        elif raw_key == "Precipitation_log":
            value = np.log1p(precipitation_in)
        elif raw_key == "Day":
            value = 1  # Default day
        elif raw_key == "Year":
            value = 2023  # Default year
        elif raw_key == "Precipitation_in":
            value = precipitation_in
        else:
            continue
            
        if isinstance(value, bool):
            value = to_binary(value)
        elif raw_key in {"Hour", "Month", "Day", "Year"}:
            value = safe_int(value)
        else:
            value = safe_float(value)
        set_feature(row, lookup, aliases, value)

    eng = engineered_features(raw)
    for k, v in eng.items():
        set_feature(row, lookup, [k], v)

    set_feature(row, lookup, ["Day_of_Week_Num", "Day_of_Week"], eng["Day_of_Week_Num"])
    set_feature(row, lookup, ["Sunrise_Sunset"], int(raw["Sunrise_Sunset"] == "Night"))
    
    # State encoding (if State column exists)
    state_value = 0  # Default to 0 or CA
    set_feature(row, lookup, ["State"], state_value)

    onehot_groups = infer_onehot_groups(
        expected_cols,
        ["Weather_Condition", "Sunrise_Sunset", "Season", "Day_of_Week", "Time_Period"]
    )

    active_categories = {
        "Weather_Condition": raw["Weather_Condition"],
        "Sunrise_Sunset": raw["Sunrise_Sunset"],
        "Season": eng["Season"],
        "Day_of_Week": raw["Day_of_Week"],
    }
    
    # Determine active time period from engineered features
    if eng.get("Time_Period_Morning"):
        active_categories["Time_Period"] = "Morning"
    elif eng.get("Time_Period_Afternoon"):
        active_categories["Time_Period"] = "Afternoon"
    elif eng.get("Time_Period_Evening"):
        active_categories["Time_Period"] = "Evening"
    else:
        active_categories["Time_Period"] = "Night"

    for group_name, cols in onehot_groups.items():
        if group_name not in active_categories:
            continue
        selected = canonicalize(active_categories[group_name])
        prefix = canonicalize(group_name) + "_"
        for col in cols:
            c = canonicalize(col)
            suffix = c[len(prefix):] if c.startswith(prefix) else c
            row[col] = float(suffix == selected)

    X = pd.DataFrame([row], columns=expected_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    debug = {
        "raw_input": raw,
        "engineered": eng,
        "log_transforms": {
            "Distance_log": np.log1p(distance_mi),
            "Visibility_log": np.log1p(visibility_mi),
            "Wind_Speed_log": np.log1p(wind_speed_mph),
            "Precipitation_log": np.log1p(precipitation_in),
        },
        "non_zero_features": {k: v for k, v in row.items() if safe_float(v) != 0.0},
    }
    return X, debug


def normalize_display_class(cls: Any) -> Any:
    """
    Normalize internal model classes to app display classes.

    0 -> 1
    1 -> 2
    2 -> 3
    3 -> 4

    If model already uses 1..4, keep it as-is.
    """
    try:
        value = int(cls)
    except Exception:
        return cls

    if value in {1, 2, 3, 4}:
        return value
    if value in {0, 1, 2, 3}:
        return value + 1
    return value


def normalize_probability_map(classes: List[Any], probs: np.ndarray) -> Dict[int, float]:
    """
    Convert model probability outputs into the app's 1..4 severity system.

    Handles:
    - XGBoost internal classes [0,1,2,3]
    - Models already using [1,2,3,4]

    Returns a complete dict for displayed classes 1..4 where possible.
    """
    mapped: Dict[int, float] = {}
    for cls, prob in zip(classes, probs):
        disp = normalize_display_class(cls)
        try:
            disp = int(disp)
        except Exception:
            continue
        mapped[disp] = float(prob)

    # Keep ordering consistent
    return {k: mapped.get(k, 0.0) for k in sorted(mapped.keys())}


def predict(model, X: pd.DataFrame) -> Dict[str, Any]:
    raw_pred = model.predict(X)[0]
    display_pred = normalize_display_class(raw_pred)

    proba_map = None
    confidence = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [])) or list(range(len(probs)))
        proba_map = normalize_probability_map(classes, probs)

        try:
            display_pred = int(display_pred)
        except Exception:
            pass

        confidence = proba_map.get(display_pred)

        if confidence is None and proba_map:
            display_pred = max(proba_map, key=proba_map.get)
            confidence = proba_map[display_pred]

    return {
        "predicted_class": display_pred,
        "probabilities": proba_map,
        "confidence": confidence,
    }


def severity_text(display_class: Any) -> Tuple[Any, str]:
    try:
        c = int(display_class)
        return c, SEVERITY_LABELS.get(c, f"Severity {c}")
    except Exception:
        return display_class, str(display_class)


def plot_probabilities(prob_map: Dict[int, float]):
    """
    Always plot probabilities using Severity 1..4 labels.
    """
    ordered_classes = [1, 2, 3, 4]
    df = pd.DataFrame(
        {
            "Severity": [f"Severity {c}" for c in ordered_classes],
            "Probability": [100 * prob_map.get(c, 0.0) for c in ordered_classes],
        }
    )

    fig = px.bar(
        df,
        x="Severity",
        y="Probability",
        text="Probability",
        template="plotly_dark",
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        yaxis_title="Probability (%)",
        xaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def scenario_flags(raw: Dict[str, Any]) -> List[str]:
    flags = []
    if raw["Hour"] in {7, 8, 15, 16, 17}:
        flags.append("Rush hour")
    if raw["Sunrise_Sunset"] == "Night":
        flags.append("Night conditions")
    if raw["Weather_Condition"] in {"Rain", "Snow", "Fog", "Thunderstorm"}:
        flags.append(raw["Weather_Condition"])
    if raw["Junction"] and not raw["Traffic_Signal"]:
        flags.append("Unsignalized junction")
    if raw["Distance_mi"] >= 2.0:
        flags.append("Long impact distance")
    if raw["Month"] in {11, 12, 1, 2}:
        flags.append("Winter season")
    return flags


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

artifact_error = None
saved_feature_columns: List[str] = []

try:
    saved_feature_columns = load_feature_columns()
except Exception as e:
    artifact_error = str(e)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🚦 Severity Dashboard")
page = st.sidebar.radio("Navigation", ["Overview", "Predict", "Findings", "Model Results"])
model_name = st.sidebar.selectbox("Prediction model", list(MODEL_PATHS.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.caption(
    "This dashboard presents the results of a traffic accident severity analysis built on large-scale crash data."
)
st.sidebar.caption(
    "It combines analytical findings, model comparison, and scenario-based prediction in one interface."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Summary")
st.sidebar.markdown(
    """
- **Study focus:** factors associated with accident severity  
- **Data scope:** large-scale historical crash records  
- **Severity scale:** standardized to **1–4** across the app  
- **Main use:** interpret patterns and test scenario risk  
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Why It Matters")
st.sidebar.markdown(
    """
The project helps identify where severe crashes are more likely to emerge and how it affects traffic flow and safety. 
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Snapshot")
best_row = PERFORMANCE_DATA.sort_values("Accuracy", ascending=False).iloc[0]
st.sidebar.markdown(
    f"""
- **Best overall model:** {best_row['Model']}
- **Accuracy:** {best_row['Accuracy']:.1%}
- **ROC-AUC:** {best_row['ROC-AUC']:.3f}
- **Alternative for comparison:** Random Forest
"""
)

st.sidebar.markdown("---")
st.sidebar.caption("All displayed severity labels use the 1–4 system.")


# =============================================================================
# OVERVIEW
# =============================================================================

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:1.95rem;font-weight:800;color:white;">Crash Severity Analysis Dashboard</div>
            <div class="muted" style="margin-top:0.35rem;">
                This project examines how accident severity changes across time, location, weather, and roadway context, and how well machine learning models can capture those patterns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='metric-card'><div class='metric-value'>500k</div><div class='metric-label'>Accident records studied</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='metric-card'><div class='metric-value'>2016–2023</div><div class='metric-label'>Observation period</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='metric-card'><div class='metric-value'>4</div><div class='metric-label'>Severity levels analyzed</div></div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.08, 0.92])

    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### What this study examined")
        st.markdown(
            """
            This analysis explored which factors are most associated with traffic accident severity. The study focused on how severity changes across location, timing, weather, roadway setting, and infrastructure context, and then tested machine learning models to evaluate how well those patterns could be predicted.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### What the analysis revealed")
        st.markdown(
            """
            The results showed that severity is shaped by a combination of crash extent, seasonality, geographic variation, and road environment. Longer impact distance, winter-related conditions, and weaker traffic control around conflict points stood out as important signals of elevated severity risk.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Key result snapshot")
        st.metric("Best overall model", "XGBoost")
        st.metric("Best accuracy", "81.3%")
        st.metric("Reference ROC-AUC", "0.880")
        st.markdown(
            """
            **Interpretation:**  
            XGBoost achieved the strongest overall predictive performance, while the broader comparison showed that model choice still depends on whether the priority is general accuracy or more balanced handling of rarer severe crashes.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Recommended focus areas")
        st.markdown(
            """
- Prioritize unsignalized conflict points for signal installation or redesign  
- Strengthen interventions on winter-sensitive and low-visibility corridors  
- Use geographic risk mapping to identify persistent severity hotspots  
- Compare overall accuracy with severe-case detection when selecting models  
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if artifact_error:
        st.error(f"Artifact loading error: {artifact_error}")


# =============================================================================
# PREDICT
# =============================================================================

elif page == "Predict":
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:1.8rem;font-weight:800;color:white;">Scenario Prediction</div>
            <div class="muted" style="margin-top:0.3rem;">
                Test the trained models on specific accident conditions and compare how different scenarios shift predicted severity within the app’s standardized 1–4 severity system.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if artifact_error:
        st.error(f"Cannot continue: {artifact_error}")
        st.stop()

    try:
        model = load_model(model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    expected_cols = model_features(model, saved_feature_columns)
    preset_name = st.selectbox("Scenario preset", list(SCENARIOS.keys()), index=0)

    base_defaults = SCENARIOS["Low-risk daytime urban"]
    defaults = base_defaults if SCENARIOS[preset_name] is None else SCENARIOS[preset_name]

    t1, t2, t3 = st.tabs(["Location & Time", "Weather", "Road Context"])

    with t1:
        a, b, c = st.columns(3)
        with a:
            start_lat = st.number_input("Latitude", -90.0, 90.0, float(defaults["Start_Lat"]), 0.01)
            start_lng = st.number_input("Longitude", -180.0, 180.0, float(defaults["Start_Lng"]), 0.01)
        with b:
            distance_mi = st.slider("Distance (mi)", 0.0, 10.0, float(defaults["Distance_mi"]), 0.1)
            hour = st.slider("Hour", 0, 23, int(defaults["Hour"]))
        with c:
            month = st.selectbox("Month", list(range(1, 13)), index=int(defaults["Month"]) - 1)
            day_of_week = st.selectbox("Day of week", list(DAY_TO_NUM.keys()), index=list(DAY_TO_NUM.keys()).index(defaults["Day_of_Week"]))

    with t2:
        a, b, c = st.columns(3)
        with a:
            temperature_f = st.slider("Temperature (°F)", -30, 120, int(defaults["Temperature_F"]))
            humidity_pct = st.slider("Humidity (%)", 0, 100, int(defaults["Humidity_pct"]))
        with b:
            pressure_in = st.slider("Pressure (inHg)", 26.0, 33.0, float(defaults["Pressure_in"]), 0.1)
            visibility_mi = st.slider("Visibility (mi)", 0.0, 25.0, float(defaults["Visibility_mi"]), 0.1)
        with c:
            wind_speed_mph = st.slider("Wind speed (mph)", 0, 80, int(defaults["Wind_Speed_mph"]))
            weather_condition = st.selectbox(
                "Weather condition",
                ["Clear", "Fair", "Cloudy", "Overcast", "Fog", "Rain", "Snow", "Thunderstorm"],
                index=["Clear", "Fair", "Cloudy", "Overcast", "Fog", "Rain", "Snow", "Thunderstorm"].index(defaults["Weather_Condition"])
            )
            sunrise_sunset = st.selectbox("Day / Night", ["Day", "Night"], index=["Day", "Night"].index(defaults["Sunrise_Sunset"]))

    with t3:
        a, b, c = st.columns(3)
        with a:
            traffic_signal = st.checkbox("Traffic signal", value=defaults["Traffic_Signal"])
            junction = st.checkbox("Junction", value=defaults["Junction"])
            crossing = st.checkbox("Crossing", value=defaults["Crossing"])
            stop = st.checkbox("Stop sign", value=defaults["Stop"])
        with b:
            amenity = st.checkbox("Amenity nearby", value=defaults["Amenity"])
            bump = st.checkbox("Bump", value=defaults["Bump"])
            give_way = st.checkbox("Give way", value=defaults["Give_Way"])
        with c:
            railway = st.checkbox("Railway", value=defaults["Railway"])
            roundabout = st.checkbox("Roundabout", value=defaults["Roundabout"])
            station = st.checkbox("Station", value=defaults["Station"])

    raw_input = {
        "Start_Lat": start_lat,
        "Start_Lng": start_lng,
        "Distance_mi": distance_mi,
        "Hour": hour,
        "Month": month,
        "Day_of_Week": day_of_week,
        "Temperature_F": temperature_f,
        "Humidity_pct": humidity_pct,
        "Pressure_in": pressure_in,
        "Visibility_mi": visibility_mi,
        "Wind_Speed_mph": wind_speed_mph,
        "Weather_Condition": weather_condition,
        "Sunrise_Sunset": sunrise_sunset,
        "Traffic_Signal": traffic_signal,
        "Junction": junction,
        "Crossing": crossing,
        "Stop": stop,
        "Amenity": amenity,
        "Bump": bump,
        "Give_Way": give_way,
        "Railway": railway,
        "Roundabout": roundabout,
        "Station": station,
    }

    flags = scenario_flags(raw_input)
    if flags:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("**Scenario flags**")
        st.markdown("  ".join([f"<span class='pill' style='background:rgba(255,255,255,0.12);'>{f}</span>" for f in flags]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Prediction setup")
        st.caption(f"Model: {model_name}")
        st.caption(f"Expected training features: {len(expected_cols)}")
        st.caption("Severity display is standardized to 1–4 for all models.")
        predict_clicked = st.button("Run prediction", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        try:
            X_input, debug = build_input_frame(raw_input, expected_cols)
            result = predict(model, X_input)
            sev_class, sev_label = severity_text(result["predicted_class"])
            sev_color = SEVERITY_COLORS.get(sev_class, "#64748b")

            with right:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(f"<span class='pill' style='background:{sev_color};'>Severity {sev_class}</span>", unsafe_allow_html=True)
                st.markdown(f"## {sev_label}")
                if result["confidence"] is not None:
                    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                else:
                    st.caption("Probability output is unavailable for this model.")
                st.markdown("</div>", unsafe_allow_html=True)

            c1, c2 = st.columns([1.12, 0.88])

            with c1:
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.markdown("### Probability distribution")
                if result["probabilities"] is not None:
                    plot_probabilities(result["probabilities"])
                else:
                    st.info("This model does not expose class probabilities.")
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.markdown("### Interpretation")
                st.markdown(
                    f"""
                    This scenario is assessed as **Severity {sev_class} ({sev_label})** under the selected model.

                    The prediction reflects the combined influence of timing, roadway setting, weather, and accident extent. Comparing presets helps show how calmer urban daytime conditions differ from winter, low-visibility, or unsignalized conflict settings.
                    """
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Inspect prepared model input"):
                nz = pd.DataFrame([{"feature": k, "value": v} for k, v in debug["non_zero_features"].items()])
                st.dataframe(nz.sort_values("feature"), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("### Suggested comparison flow")
    st.markdown(  """
        If one model still leans heavily toward one severity class, that usually reflects class imbalance in training.
        This app fixes the inference pipeline so the selected conditions actually flow into the model correctly.
        For testing variation, try the preset scenarios and compare XGBoost against Random Forest.
                

- Start with **Low-risk daytime urban**  
- Compare against **Busy rush-hour conflict point**  
- Then test **High-risk winter highway**  
- Switch between **XGBoost** and **Random Forest** to compare model behavior          
        """    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# FINDINGS
# =============================================================================

elif page == "Findings":
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:1.8rem;font-weight:800;color:white;">Findings & Patterns</div>
            <div class="muted" style="margin-top:0.3rem;">
                The analysis highlighted recurring patterns in when, where, and under what conditions severe accidents become more likely.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)

    with a:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Top drivers of severity")
        fig = px.bar(
            TOP_FEATURES.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            color="Theme",
            orientation="h",
            template="plotly_dark",
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Weather pattern")
        fig = px.bar(
            WEATHER_INSIGHT,
            x="Weather",
            y="Severity 3+ Rate (%)",
            template="plotly_dark",
            text="Severity 3+ Rate (%)",
        )
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Interpretation of the patterns")
        st.markdown(
            """
                Several factors combined to increase severity. Worst outcomes happened when multiple risks overlapped, such as worse crashes,
                  bad weather, low visibility, nighttime, weak infrastructure, or specific dangerous locations
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)



# =============================================================================
# MODEL RESULTS
# =============================================================================

elif page == "Model Results":
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:1.8rem;font-weight:800;color:white;">Model Results</div>
            <div class="muted" style="margin-top:0.3rem;">
                The modeling stage compared several classifiers to understand both predictive strength and trade-offs across severity classes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)

    with a:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Accuracy comparison")
        fig = px.bar(PERFORMANCE_DATA, x="Model", y="Accuracy", text="Accuracy", template="plotly_dark")
        fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### ROC-AUC comparison")
        fig = px.bar(PERFORMANCE_DATA, x="Model", y="ROC-AUC", text="ROC-AUC", template="plotly_dark")
        fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Comparative reading")
        st.markdown(
            """
            **XGBoost** achieved the strongest overall scores and serves as the best general-purpose model in the dashboard.

            **Random Forest** remains important as a comparison model, especially where more balanced behavior across severity classes is desirable.

            **Decision Tree** and **Logistic Regression** provide simpler baselines that help contextualize the gains from stronger nonlinear models.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Practical conclusion")
        st.markdown(
            """
            The comparison shows that “best” depends on the operational aim:

- Use **XGBoost** when overall predictive performance is the priority  
- Compare with **Random Forest** when balanced severity handling matters more  
- Keep **Logistic Regression** and **Decision Tree** as reference baselines  
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.dataframe(PERFORMANCE_DATA, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)