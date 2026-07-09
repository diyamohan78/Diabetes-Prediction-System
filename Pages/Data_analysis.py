# ==========================================================
# Diabetes Prediction System
# Data Analysis Page
# ==========================================================
# Professional Edition
# Author: Diya Mohan
# ==========================================================

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Data Analysis",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = "diabetes.csv"
PRIMARY_COLOR = "#0F766E"
ACCENT_COLOR = "#F97316"

sns.set_style("whitegrid")


# ==========================================================
# Global Styling
# ==========================================================

def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: #F8FAFC;
            }}
            .section-title {{
                font-size: 1.15rem;
                font-weight: 700;
                color: #0F172A;
                margin-top: 0.5rem;
                margin-bottom: 0.75rem;
            }}
            .kpi-card {{
                background-color: #FFFFFF;
                border-radius: 14px;
                padding: 1.1rem 1rem;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
                border-left: 5px solid {PRIMARY_COLOR};
                text-align: center;
            }}
            .kpi-label {{
                font-size: 0.82rem;
                color: #64748B;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.3rem;
            }}
            .kpi-value {{
                font-size: 1.55rem;
                font-weight: 700;
                color: #0F172A;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> str:
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """


# ==========================================================
# Data Loading
# ==========================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    return pd.read_csv(path)


# ==========================================================
# Page Content
# ==========================================================

def main() -> None:
    inject_custom_css()

    st.title("📊 Diabetes Dataset Analysis")
    st.caption("Explore the diabetes dataset through summary statistics and visualizations.")
    st.markdown("---")

    try:
        dataset = load_data(DATA_PATH)
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}\n\nPlease place `diabetes.csv` in the app's root directory.")
        st.stop()

    # ------------------------------------------------------
    # Dataset Shape (KPIs)
    # ------------------------------------------------------
    st.markdown('<div class="section-title">📐 Dataset Shape</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(kpi_card("Rows", f"{dataset.shape[0]:,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Columns", f"{dataset.shape[1]}"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Missing Values", f"{int(dataset.isnull().sum().sum())}"), unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Dataset Preview
    # ------------------------------------------------------
    st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(dataset.head(20), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Data Types & Missing Values side by side
    # ------------------------------------------------------
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">📝 Data Types</div>', unsafe_allow_html=True)
        dtypes_df = dataset.dtypes.astype(str).rename("Data Type").to_frame()
        st.dataframe(dtypes_df, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">❓ Missing Values</div>', unsafe_allow_html=True)
        missing_df = dataset.isnull().sum().rename("Missing Count").to_frame()
        st.dataframe(missing_df, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Statistical Summary
    # ------------------------------------------------------
    st.markdown('<div class="section-title">📈 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(dataset.describe(), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Outcome Distribution & Correlation Heatmap
    # ------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">🩺 Diabetes Outcome Distribution</div>', unsafe_allow_html=True)

        if "Outcome" in dataset.columns:
            counts = dataset["Outcome"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(
                ["Non-Diabetic", "Diabetic"],
                counts.values,
                color=[PRIMARY_COLOR, ACCENT_COLOR],
                edgecolor="white",
            )
            ax.set_xlabel("Outcome")
            ax.set_ylabel("Count")
            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_alpha(0.0)
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Column 'Outcome' not found in dataset.")

    with col_right:
        st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)

        numeric_df = dataset.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(6, 4.6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="Blues",
            fmt=".2f",
            annot_kws={"size": 7},
            ax=ax,
        )
        fig.patch.set_alpha(0.0)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Feature Distribution Explorer
    # ------------------------------------------------------
    st.markdown('<div class="section-title">📊 Feature Distributions</div>', unsafe_allow_html=True)

    feature_columns = [c for c in dataset.columns if c != "Outcome"]
    selected = st.selectbox("Select a feature to explore", feature_columns)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dataset[selected], bins=20, color=PRIMARY_COLOR, edgecolor="white")
    ax.set_title(f"Distribution of {selected}")
    ax.set_xlabel(selected)
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_alpha(0.0)
    st.pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    main()