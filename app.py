# ==========================================================
# Diabetes Prediction Dashboard
# Home Page
# ==========================================================
# Professional Edition
# Author: Diya Mohan
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# Constants
# ==========================================================

DATA_PATH = "diabetes.csv"
LOGO_PATH = "images/logo.png"
BANNER_PATH = "images/Diabetes_banner.png"
PREVIEW_PATH = "images/Prediction.png"

PRIMARY_COLOR = "#0F766E"       # teal - medical / trustworthy
ACCENT_COLOR = "#F97316"        # warm accent for highlights
BG_CARD = "#FFFFFF"


# ==========================================================
# Global Styling
# ==========================================================

def inject_custom_css() -> None:
    """Apply a consistent, professional visual theme across the app."""
    st.markdown(
        f"""
        <style>
            /* Overall app background */
            .stApp {{
                background-color: #F8FAFC;
            }}

            /* Sidebar styling */
            section[data-testid="stSidebar"] {{
                background-color: #0B3B36;
            }}
            section[data-testid="stSidebar"] * {{
                color: #E6F4F1 !important;
            }}

            /* KPI card container */
            .kpi-card {{
                background-color: {BG_CARD};
                border-radius: 14px;
                padding: 1.25rem 1rem;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
                border-left: 5px solid {PRIMARY_COLOR};
                text-align: center;
            }}
            .kpi-label {{
                font-size: 0.85rem;
                color: #64748B;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.35rem;
            }}
            .kpi-value {{
                font-size: 1.7rem;
                font-weight: 700;
                color: #0F172A;
            }}

            /* Section headers */
            .section-title {{
                font-size: 1.15rem;
                font-weight: 700;
                color: #0F172A;
                margin-top: 0.25rem;
                margin-bottom: 0.75rem;
            }}

            /* Nav / feature cards */
            .nav-card {{
                background-color: {BG_CARD};
                border-radius: 14px;
                padding: 1.25rem;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
                border-top: 4px solid {ACCENT_COLOR};
                height: 100%;
            }}
            .nav-card h4 {{
                margin: 0 0 0.4rem 0;
                color: #0F172A;
            }}
            .nav-card p {{
                margin: 0;
                color: #475569;
                font-size: 0.92rem;
            }}

            .app-footer {{
                text-align: center;
                padding: 1.5rem 0 0.5rem 0;
                color: #64748B;
                font-size: 0.85rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> str:
    """Return HTML markup for a single KPI card."""
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """


def nav_card(icon: str, title: str, description: str) -> str:
    """Return HTML markup for a navigation / feature card."""
    return f"""
        <div class="nav-card">
            <h4>{icon} {title}</h4>
            <p>{description}</p>
        </div>
    """


# ==========================================================
# Data Loading (with graceful error handling)
# ==========================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    """Load the diabetes dataset from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    return pd.read_csv(path)


def safe_image(path: str, **kwargs) -> None:
    """Display an image if it exists, otherwise show a subtle placeholder."""
    if os.path.exists(path):
        st.image(path, **kwargs)
    else:
        st.info(f"Image not found: `{path}`")


# ==========================================================
# App Entry Point
# ==========================================================

def main():
    inject_custom_css()

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        safe_image(LOGO_PATH, width=160)
        st.title("🩺 Diabetes Dashboard")
        st.divider()
        with st.sidebar:
    

        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/Data Analysis.py", label="📊 Data Analysis", icon="📊")
        st.page_link("pages/Diabetes Prediction.py", label="🔍 Prediction", icon="🔍")
        st.page_link("pages/Model Performance.py", label="📈 Model Performance", icon="📈")
        st.page_link("pages/About.py", label="ℹ️ About", icon="ℹ️")
        st.info("⬅️ Select a page from the navigation above.")

    # Everything else...
        

    # ------------------------------------------------------
    # Banner + Title
    # ------------------------------------------------------
    safe_image(BANNER_PATH, use_container_width=True)

    st.title("🩺 Diabetes Prediction Dashboard")
    st.caption("AI-Powered Diabetes Risk Prediction using Logistic Regression")
    st.markdown("---")

    # ------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(kpi_card("📁 Records", f"{len(df):,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("📊 Features", f"{df.shape[1] - 1}"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("🤖 Model", "Logistic Regression"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card("🎯 Target", "Binary"), unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Charts
    # ------------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">📊 Diabetes Distribution</div>', unsafe_allow_html=True)

        if "Outcome" in df.columns:
            outcome = df["Outcome"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = [PRIMARY_COLOR, ACCENT_COLOR]
            ax.pie(
                outcome,
                labels=["Non-Diabetic", "Diabetic"],
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                textprops={"fontsize": 10},
            )
            ax.axis("equal")
            fig.patch.set_alpha(0.0)
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Column 'Outcome' not found in dataset.")

    with right:
        st.markdown('<div class="section-title">🩺 Prediction Preview</div>', unsafe_allow_html=True)
        safe_image(PREVIEW_PATH, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Dataset Snapshot
    # ------------------------------------------------------
    st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Feature Summary
    # ------------------------------------------------------
    st.markdown('<div class="section-title">📈 Feature Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Quick Navigation
    # ------------------------------------------------------
    st.markdown('<div class="section-title">🚀 Explore the Dashboard</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            nav_card(
                "📊", "Dataset Analytics",
                "Explore the complete dataset, visualizations, and feature correlations."
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            nav_card(
                "🔍", "Diabetes Prediction",
                "Predict diabetes risk using patient health parameters."
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            nav_card(
                "📈", "Model Performance",
                "View accuracy, ROC curve, confusion matrix, and classification report."
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ------------------------------------------------------
    # Footer
    # ------------------------------------------------------
    st.markdown(
        """
        <div class="app-footer">
            <strong>🩺 Diabetes Prediction Dashboard</strong><br>
            Developed by <b>Diya Mohan</b><br>
            Python • Streamlit • Machine Learning • Logistic Regression
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
