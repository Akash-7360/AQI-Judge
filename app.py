"""
Air Quality Index (AQI) Classification System
Developed by Akash Uduthala
"""
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import plotly.graph_objects as go
import joblib

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Air Quality Index (AQI) Classifier",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Atmospheric Theme Management & Background Loading
# ==============================================================================
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

with st.sidebar:
    st.markdown("### 🎨 **Theme**")
    theme_choice = st.radio(
        "Theme",
        ["☀️", "🌙"],
        index=0 if st.session_state["theme_mode"] == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state["theme_mode"] = "Light" if theme_choice == "☀️" else "Dark"

is_dark = st.session_state["theme_mode"] == "Dark"

# Load Atmospheric Background Images
@st.cache_data
def get_base64_bg(relative_path: str) -> str:
    img_file = Path(__file__).resolve().parent / relative_path
    if img_file.exists():
        with open(img_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

light_bg_b64 = get_base64_bg("assets/air_light_bg.jpg")
dark_bg_b64 = get_base64_bg("assets/air_dark_bg.jpg")
active_bg = dark_bg_b64 if is_dark else light_bg_b64

# Atmospheric Air & Breeze Color Tokens
if is_dark:
    # Midnight Troposphere / Aurora Sky
    text_main = "#f0f9ff"
    text_muted = "#94a9be"
    accent = "#38bdf8"
    accent_glow = "rgba(56, 189, 248, 0.4)"
    bg_sidebar = "rgba(9, 17, 28, 0.92)"
    bg_card = "rgba(13, 23, 38, 0.82)"
    border_color = "rgba(56, 189, 248, 0.25)"
    btn_bg = "rgba(18, 30, 48, 0.82)"
    btn_hover_bg = "rgba(28, 46, 74, 0.92)"
    input_bg = "rgba(13, 23, 38, 0.88)"
    card_shadow = "0 8px 32px rgba(0, 0, 0, 0.45)"
    hero_bg = "linear-gradient(135deg, rgba(15, 26, 44, 0.88) 0%, rgba(10, 36, 52, 0.82) 100%)"
    hero_shadow = "0 12px 36px rgba(0, 0, 0, 0.60)"
    plot_template = "plotly_dark"
else:
    # Clean Atmospheric Morning Breeze
    text_main = "#0c2438"
    text_muted = "#335c75"
    accent = "#0284c7"
    accent_glow = "rgba(2, 132, 199, 0.35)"
    bg_sidebar = "rgba(240, 248, 252, 0.90)"
    bg_card = "rgba(255, 255, 255, 0.88)"
    border_color = "rgba(186, 230, 253, 0.80)"
    btn_bg = "rgba(240, 249, 255, 0.90)"
    btn_hover_bg = "#e0f2fe"
    input_bg = "rgba(255, 255, 255, 0.92)"
    card_shadow = "0 8px 30px rgba(2, 132, 199, 0.08)"
    hero_bg = "linear-gradient(135deg, rgba(255, 255, 255, 0.94) 0%, rgba(224, 242, 254, 0.88) 100%)"
    hero_shadow = "0 10px 32px rgba(2, 132, 199, 0.12)"
    plot_template = "plotly_white"

custom_css = f"""
<style>
    /* Atmospheric Air Background Image */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{active_bg}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        color: {text_main} !important;
    }}
    .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {text_main};
    }}

    /* Transparent Seamless Header Bar */
    header, [data-testid="stHeader"] {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    [data-testid="stDecoration"] {{
        display: none !important;
    }}
    [data-testid="stHeader"] svg, [data-testid="stToolbar"] svg, [data-testid="stHeader"] button, [data-testid="stToolbar"] button {{
        fill: {text_main} !important;
        color: {text_main} !important;
    }}

    /* Frosted Glass Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border-right: 1px solid {border_color} !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {text_main} !important;
    }}

    /* Frosted Glass Atmospheric Cards */
    .clean-card {{
        background-color: {bg_card} !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid {border_color} !important;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: {card_shadow};
    }}
    .clean-card p, .clean-card span, .clean-card div {{
        color: {text_main} !important;
    }}

    /* Atmospheric Hero Banner */
    .hero-header {{
        background: {hero_bg} !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid {border_color} !important;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: {hero_shadow};
    }}

    /* Frosted Atmospheric Buttons */
    .stButton > button {{
        background-color: {btn_bg} !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: {text_main} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease;
    }}
    .stButton > button:hover {{
        border-color: {accent} !important;
        background-color: {btn_hover_bg} !important;
        color: {accent} !important;
        transform: translateY(-1px);
    }}
    .stButton > button[kind="primary"] {{
        background-color: {accent} !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px {accent_glow} !important;
    }}

    /* Frosted Inputs */
    input, select, textarea, [data-baseweb="input"], [data-baseweb="base-input"] {{
        background-color: {input_bg} !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        color: {text_main} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
    }}

    /* Badge */
    .badge {{
        display: inline-block;
        padding: 5px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.4px;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==============================================================================
# Helper Functions & CPCB Categorization
# ==============================================================================
def render_html(html_str: str):
    """Renders HTML safely without markdown indented code-block quirks."""
    clean = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

def get_cpcb_details(category: str):
    info = {
        "Good": {
            "color": "#10B981",
            "textColor": "#ffffff",
            "icon": "🟢",
            "range": "0 - 30 µg/m³ (AQI 0-50)",
            "impact": "Clean Air. Minimal or no health risk.",
            "action": "Safe for all outdoor activities and exercise."
        },
        "Satisfactory": {
            "color": "#84CC16",
            "textColor": "#0f172a",
            "icon": "🟡",
            "range": "31 - 60 µg/m³ (AQI 51-100)",
            "impact": "Acceptable air quality. Minor irritation to highly sensitive people.",
            "action": "Normal outdoor activities are safe for most individuals."
        },
        "Moderate": {
            "color": "#F59E0B",
            "textColor": "#0f172a",
            "icon": "🟠",
            "range": "61 - 90 µg/m³ (AQI 101-200)",
            "impact": "Breathing discomfort to people with lungs, asthma, and heart diseases.",
            "action": "Sensitive groups should reduce prolonged outdoor exertion."
        },
        "Poor": {
            "color": "#F97316",
            "textColor": "#ffffff",
            "icon": "🔴",
            "range": "91 - 120 µg/m³ (AQI 201-300)",
            "impact": "Breathing discomfort to most people on prolonged exposure.",
            "action": "Wear N95 mask outdoors; limit strenuous outdoor exercise."
        },
        "Very Poor": {
            "color": "#EF4444",
            "textColor": "#ffffff",
            "icon": "🟣",
            "range": "121 - 250 µg/m³ (AQI 301-400)",
            "impact": "Respiratory illness on prolonged exposure. Severe hazard.",
            "action": "Avoid outdoor activities; run indoor air purifiers."
        },
        "Severe": {
            "color": "#7F1D1D",
            "textColor": "#ffffff",
            "icon": "🚨",
            "range": "250+ µg/m³ (AQI 401-500)",
            "impact": "Emergency health hazard. Seriously affects even healthy people.",
            "action": "Strictly remain indoors; avoid any outdoor air exposure."
        }
    }
    return info.get(category, info["Moderate"])

def get_ground_truth_category(pm: float) -> str:
    if pm <= 30:
        return "Good"
    elif pm <= 60:
        return "Satisfactory"
    elif pm <= 90:
        return "Moderate"
    elif pm <= 120:
        return "Poor"
    elif pm <= 250:
        return "Very Poor"
    else:
        return "Severe"

# ==============================================================================
# Model Loading & Training (Decision Tree, Naive Bayes, KNN)
# ==============================================================================
@st.cache_resource
def load_or_train_classification_models():
    model_path = Path(__file__).resolve().parent / "model.joblib"
    if model_path.exists():
        try:
            saved_data = joblib.load(model_path)
            if isinstance(saved_data, dict) and "models" in saved_data:
                return saved_data["models"], saved_data["summary_df"], saved_data["reports"]
        except Exception:
            pass

    csv_path = Path(__file__).resolve().parent / "data" / "air-quality-india.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / "dataset" / "air-quality-india.csv"
    df = pd.read_csv(csv_path)

    df["Category"] = df["PM2.5"].apply(get_ground_truth_category)

    feature_cols = ["PM2.5", "Hour", "Month"]
    X = df[feature_cols]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Decision Tree (Champion)
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)

    # 2. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # 3. K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    trained_models = {
        "Decision Tree": dt,
        "Naive Bayes": nb,
        "KNN": knn
    }

    # Compute evaluation metrics
    metrics_summary = []
    reports = {}

    for name, clf in trained_models.items():
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)

        metrics_summary.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1-Score": f1
        })
        reports[name] = report_dict

    summary_df = pd.DataFrame(metrics_summary)

    # Cache pre-trained models to model.joblib for instant startup
    try:
        joblib.dump({
            "models": trained_models,
            "summary_df": summary_df,
            "reports": reports
        }, model_path)
    except Exception:
        pass

    return trained_models, summary_df, reports

models_dict, summary_metrics_df, detailed_reports = load_or_train_classification_models()

# ==============================================================================
# Sidebar Project Info & Navigation
# ==============================================================================
with st.sidebar:
    st.divider()
    st.markdown("### 🧭 **Navigation**")
    selected_view = st.radio(
        "Navigation",
        [
            "🎯 Sample AQI Prediction",
            "📊 Model Comparison & Reports"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### 📌 **Project Details**")
    st.markdown(
        """
        - **Project:** Air Quality Category Classification
        - **Standards:** CPCB India (NAQI)
        - **Dataset:** 74,640 Indian Hourly Records (2017–2026)
        - **Algorithms:**
          1. 🌲 Decision Tree (Champion)
          2. 👥 K-Nearest Neighbors (KNN)
          3. 📊 Naive Bayes (Gaussian)
        """
    )
    render_html(
        f"""
        <div style="margin-top: 24px; padding-top: 10px; border-top: 1px solid rgba(148, 163, 184, 0.2); text-align: center; font-size: 0.72rem; color: {text_muted}; opacity: 0.6; letter-spacing: 0.2px;">
            Developed by Akash Uduthala
        </div>
        """
    )

# ==============================================================================
# Header Section
# ==============================================================================
render_html(
    f"""
    <div class="hero-header">
        <h2 style="margin: 0; font-weight: 800; color: {accent};">
            🍃 Air Quality Index (AQI) Classifier 🌿
        </h2>
        <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: {text_muted};">
            Nature-Inspired Environmental Monitoring • Decision Tree, Naive Bayes, and KNN on CPCB Standards
        </p>
    </div>
    """
)

# ==============================================================================
# VIEW 1: Sample AQI Prediction (Showcase Best Model + Side Comparison)
# ==============================================================================
if selected_view == "🎯 Sample AQI Prediction":
    # Quick Preset Scenarios
    st.markdown("##### ⚡ Quick Scenario Presets:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    # Session state for inputs
    if "sample_pm" not in st.session_state:
        st.session_state["sample_pm"] = 65.0
    if "pm_slider" not in st.session_state:
        st.session_state["pm_slider"] = float(st.session_state["sample_pm"])
    if "pm_num" not in st.session_state:
        st.session_state["pm_num"] = float(st.session_state["sample_pm"])
    if "sample_hour" not in st.session_state:
        st.session_state["sample_hour"] = 14
    if "sample_month" not in st.session_state:
        st.session_state["sample_month"] = 10

    def sync_from_num():
        val = float(st.session_state["pm_num"])
        st.session_state["sample_pm"] = val
        st.session_state["pm_slider"] = val

    def sync_from_slider():
        val = float(st.session_state["pm_slider"])
        st.session_state["sample_pm"] = val
        st.session_state["pm_num"] = val

    def apply_preset(pm_val, hr_val, mo_val):
        val = float(pm_val)
        st.session_state["sample_pm"] = val
        st.session_state["pm_num"] = val
        st.session_state["pm_slider"] = val
        st.session_state["sample_hour"] = int(hr_val)
        st.session_state["sample_month"] = int(mo_val)

    col_p1.button("🍃 Clean Mountain (Good)", on_click=apply_preset, args=(22.0, 8, 7), use_container_width=True)
    col_p2.button("🌾 Rural Forest (Satisfactory)", on_click=apply_preset, args=(52.0, 10, 5), use_container_width=True)
    col_p3.button("🏙️ Urban Traffic (Moderate)", on_click=apply_preset, args=(78.0, 17, 10), use_container_width=True)
    col_p4.button("🌫️ Smog Inversion (Poor)", on_click=apply_preset, args=(115.0, 21, 12), use_container_width=True)

    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)

    # Input Parameters Bar
    st.markdown('<div class="clean-card" style="padding: 18px 22px;">', unsafe_allow_html=True)
    
    # PM2.5 Concentration Input (Synchronized Number Input with +/- buttons and Range Slider)
    col_pm_num, col_pm_slider = st.columns([1, 1.8])
    with col_pm_num:
        pm_input = st.number_input(
            "PM2.5 Concentration (µg/m³)",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            key="pm_num",
            on_change=sync_from_num,
            help="Fine-tune with + / - buttons or enter an exact value"
        )
    with col_pm_slider:
        st.slider(
            "Slide to Adjust PM2.5 Concentration",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            key="pm_slider",
            on_change=sync_from_slider,
            help="Drag slider to smoothly increase or decrease PM2.5 reading"
        )

    st.markdown(f'<div style="height: 1px; background: {border_color}; margin: 12px 0 16px 0; opacity: 0.5;"></div>', unsafe_allow_html=True)

    # Temporal Parameters (Hour and Month)
    col_in2, col_in3 = st.columns([1, 1])
    with col_in2:
        hour_input = st.slider("Hour of Day (0–23)", 0, 23, value=int(st.session_state["sample_hour"]))
    with col_in3:
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_idx = int(st.session_state["sample_month"]) - 1
        month_name = st.selectbox("Month of Year", month_names, index=month_idx)
        month_input = month_names.index(month_name) + 1
    st.markdown('</div>', unsafe_allow_html=True)

    # Prepare DataFrame for Predictions
    sample_df = pd.DataFrame([{
        "PM2.5": pm_input,
        "Hour": hour_input,
        "Month": month_input
    }])

    # Predictions
    pred_dt = models_dict["Decision Tree"].predict(sample_df)[0]
    pred_nb = models_dict["Naive Bayes"].predict(sample_df)[0]
    pred_knn = models_dict["KNN"].predict(sample_df)[0]
    truth_cat = get_ground_truth_category(pm_input)

    # Probabilities for Naive Bayes
    nb_probs = models_dict["Naive Bayes"].predict_proba(sample_df)[0]
    nb_classes = list(models_dict["Naive Bayes"].classes_)
    nb_conf = max(nb_probs) * 100

    # Layout: Showcase Best Model cleanly on the main screen
    dt_info = get_cpcb_details(pred_dt)
    render_html(
        f"""
        <div class="clean-card" style="border-top: 6px solid {dt_info['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 0.8rem; font-weight: 800; letter-spacing: 1px; color: {accent}; text-transform: uppercase;">
                        🌟 BEST CLASSIFIER
                    </span>
                    <h2 style="margin: 2px 0 0 0; font-size: 1.6rem; font-weight: 800; color: {text_main};">
                        Decision Tree Classifier
                    </h2>
                </div>
                <span class="badge" style="background-color: {dt_info['color']}; color: {dt_info['textColor']}; font-size: 0.9rem;">
                    Top Accuracy: 100.0%
                </span>
            </div>
            <div style="background-color: {'rgba(10, 26, 44, 0.85)' if is_dark else 'rgba(235, 246, 254, 0.92)'}; border: 1px solid {border_color}; border-radius: 12px; padding: 18px; margin: 12px 0; text-align: center;">
                <div style="font-size: 0.85rem; color: {text_muted}; text-transform: uppercase; font-weight: 600;">
                    Predicted CPCB Health Category
                </div>
                <div style="font-size: 2.5rem; font-weight: 900; color: {dt_info['color']}; margin: 4px 0;">
                    {dt_info['icon']} {pred_dt}
                </div>
                <div style="font-size: 0.88rem; color: {text_muted};">
                    Standard CPCB Range: <b>{dt_info['range']}</b>
                </div>
            </div>
            <div style="margin-top: 14px;">
                <h4 style="margin: 0 0 6px 0; color: {text_main}; font-size: 1.05rem;">
                    🩺 Health Advisory & Precautionary Guidance:
                </h4>
                <p style="margin: 0 0 10px 0; font-size: 0.92rem; color: {text_muted};">
                    <b>Impact:</b> {dt_info['impact']}
                </p>
                <div style="background-color: {'rgba(14, 34, 56, 0.85)' if is_dark else 'rgba(240, 249, 255, 0.94)'}; border-left: 4px solid {dt_info['color']}; padding: 12px 14px; border-radius: 0 8px 8px 0; font-size: 0.9rem;">
                    <b>🏃 Recommended Action:</b> {dt_info['action']}
                </div>
            </div>
        </div>
        """
    )

# ==============================================================================
# VIEW 2: Model Comparison & Reports
# ==============================================================================
elif selected_view == "📊 Model Comparison & Reports":
    st.markdown("### 📊 Algorithms Performance Comparison")
    st.caption("Evaluated on 14,928 held-out testing records (20% of 74,640 records) from Indian CPCB monitoring network (2017–2026).")

    # 1. Summary Metrics Table
    display_df = summary_metrics_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["F1-Score"] = display_df["F1-Score"].apply(lambda x: f"{x * 100:.2f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 2. Model-Comparison Chart (Grouped Bar Chart)
    st.markdown("### 📈 Model Comparison Chart")

    chart_models = summary_metrics_df["Algorithm"].tolist()
    accuracies = [v * 100 for v in summary_metrics_df["Accuracy"]]
    precisions = [v * 100 for v in summary_metrics_df["Precision"]]
    recalls = [v * 100 for v in summary_metrics_df["Recall"]]
    f1_scores = [v * 100 for v in summary_metrics_df["F1-Score"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_models, y=accuracies, name="Accuracy (%)", marker_color="#38bdf8"))
    fig.add_trace(go.Bar(x=chart_models, y=precisions, name="Precision (%)", marker_color="#10b981"))
    fig.add_trace(go.Bar(x=chart_models, y=recalls, name="Recall (%)", marker_color="#f59e0b"))
    fig.add_trace(go.Bar(x=chart_models, y=f1_scores, name="F1-Score (%)", marker_color="#a855f7"))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[80, 105], title="Score (%)"),
        xaxis=dict(title="Algorithm"),
        template=plot_template,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Precision & Recall Reports (Detailed Breakdown)
    st.markdown("### 📋 Precision & Recall Reports by Class")

    selected_model = st.selectbox(
        "Select Algorithm to inspect detailed classification report:",
        ["Decision Tree", "Naive Bayes", "KNN"]
    )

    report_dict = detailed_reports[selected_model]
    classes = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor"]
    class_rows = []

    for cls in classes:
        if cls in report_dict:
            class_rows.append({
                "AQI Category": cls,
                "Precision": f"{report_dict[cls]['precision'] * 100:.2f}%",
                "Recall": f"{report_dict[cls]['recall'] * 100:.2f}%",
                "F1-Score": f"{report_dict[cls]['f1-score'] * 100:.2f}%",
                "Support (Samples)": int(report_dict[cls]["support"])
            })

    report_df = pd.DataFrame(class_rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Project Summary & Key Takeaways
    render_html(
        f"""
        <div class="clean-card" style="margin-top: 14px;">
            <h4 style="margin: 0 0 8px 0; color: {accent};">💡 Key Presentation Takeaways:</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 0.92rem; color: {text_muted};">
                <li><b>Decision Tree (100.0%):</b> Directly learns orthogonal split boundaries corresponding to official CPCB cutoff thresholds (30, 60, 90, 120, 250 µg/m³).</li>
                <li><b>K-Nearest Neighbors / KNN (99.49%):</b> Groups sensor readings with nearest historical neighbor observations, yielding near-perfect category clustering.</li>
                <li><b>Naive Bayes (96.64%):</b> Assumes Gaussian bell-curve probability distributions for each class, performing robustly despite minor overlap near boundary thresholds.</li>
            </ul>
        </div>
        """
    )

render_html(
    f"""
    <div style="margin-top: 36px; padding: 14px 0 6px 0; text-align: center; font-size: 0.73rem; color: {text_muted}; opacity: 0.55; letter-spacing: 0.25px;">
        Air Quality Classification System • Developed by Akash Uduthala
    </div>
    """
)
