# ==========================================================
# Diabetes Prediction System
# About Page
# ==========================================================
# Professional Edition
# Author: Diya Mohan
# ==========================================================

import streamlit as st




# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide",
)

PRIMARY_COLOR = "#0F766E"
ACCENT_COLOR = "#F97316"


# ==========================================================
# Global Styling (kept consistent with the rest of the app)
# ==========================================================

def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: #F8FAFC;
            }}

            .info-card {{
                background-color: #FFFFFF;
                border-radius: 14px;
                padding: 1.5rem 1.75rem;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
                border-left: 5px solid {PRIMARY_COLOR};
                margin-bottom: 1rem;
            }}
            .info-card h3 {{
                margin-top: 0;
                color: #0F172A;
            }}
            .info-card p, .info-card li {{
                color: #334155;
                font-size: 0.97rem;
                line-height: 1.55;
            }}

            .pill {{
                display: inline-block;
                background-color: #E6F4F1;
                color: {PRIMARY_COLOR};
                font-weight: 600;
                font-size: 0.82rem;
                padding: 0.3rem 0.75rem;
                border-radius: 999px;
                margin: 0.2rem 0.35rem 0.2rem 0;
            }}
SS
            .step-row {{
                display: flex;
                align-items: flex-start;
                gap: 0.75rem;
                padding: 0.55rem 0;
                border-bottom: 1px solid #EEF2F6;
            }}
            .step-num {{
                background-color: {PRIMARY_COLOR};
                color: white;
                font-weight: 700;
                font-size: 0.8rem;
                width: 1.6rem;
                height: 1.6rem;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                margin-top: 0.1rem;
            }}
            .step-text {{
                color: #334155;
                font-size: 0.95rem;
                padding-top: 0.1rem;
            }}

            .dev-card {{
                background: linear-gradient(135deg, {PRIMARY_COLOR}, #134E4A);
                color: white;
                border-radius: 16px;
                padding: 1.75rem 2rem;
            }}
            .dev-card h3, .dev-card p {{
                color: white !important;
                margin: 0.2rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def pills(items: list[str]) -> str:
    """Render a list of strings as rounded pill badges."""
    return "".join(f'<span class="pill">{item}</span>' for item in items)


def numbered_steps(steps: list[str]) -> str:
    """Render an ordered list as clean numbered rows instead of a plain <ol>."""
    rows = "".join(
        f'<div class="step-row"><div class="step-num">{i}</div>'
        f'<div class="step-text">{step}</div></div>'
        for i, step in enumerate(steps, start=1)
    )
    return f'<div>{rows}</div>'


# ==========================================================
# Page Content
# ==========================================================

def main() -> None:
    inject_custom_css()

    st.title("ℹ️ About This Project")
    st.caption("Everything you need to know about the Diabetes Prediction System")
    st.markdown("---")

    # ------------------------------------------------------
    # Project Overview
    # ------------------------------------------------------
    st.markdown(
        """
        <div class="info-card">
            <h3>🩺 Project Overview</h3>
            <p>
                The <b>Diabetes Prediction System</b> is a machine learning web application
                built to estimate the likelihood that a patient is diabetic, based on a
                small set of routinely collected medical measurements.
            </p>
            <p>
                It is powered by a <b>Logistic Regression</b> classifier trained on the
                widely used <b>Pima Indians Diabetes Dataset</b>. Users enter patient
                information and receive an instant, probability-backed prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------
    # Dataset Information + Input Features
    # ------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📂 Dataset Information</h3>
                <p><b>Dataset:</b> Pima Indians Diabetes Dataset</p>
                <p><b>Records:</b> 768 &nbsp;|&nbsp; <b>Features:</b> 8</p>
                <p><b>Target Variable:</b> Outcome</p>
                <ul>
                    <li><b>0</b> → Non-Diabetic</li>
                    <li><b>1</b> → Diabetic</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>📊 Input Features</h3>
                <p>{pills([
                    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                    "Insulin", "BMI", "Diabetes Pedigree Function", "Age",
                ])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------
    # ML Workflow
    # ------------------------------------------------------
    st.markdown(
        f"""
        <div class="info-card">
            <h3>🤖 Machine Learning Workflow</h3>
            {numbered_steps([
                "Load the dataset",
                "Clean and preprocess the data",
                "Scale features using StandardScaler",
                "Split data into train and test sets",
                "Train the Logistic Regression model",
                "Evaluate model performance",
                "Persist the model with Joblib",
                "Serve real-time predictions via Streamlit",
            ])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------
    # Technologies + Features
    # ------------------------------------------------------
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>🛠️ Technologies Used</h3>
                <p>{pills([
                    "Python", "Streamlit", "Pandas", "NumPy",
                    "Scikit-learn", "Matplotlib", "Seaborn", "Joblib",
                ])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="info-card">
                <h3>✨ Project Features</h3>
                <ul>
                    <li>Interactive Streamlit dashboard</li>
                    <li>Real-time diabetes prediction</li>
                    <li>In-depth exploratory data analysis</li>
                    <li>Model performance evaluation</li>
                    <li>Confusion matrix &amp; ROC curve</li>
                    <li>Classification report</li>
                    <li>Prediction probability breakdown</li>
                    <li>Clean, professional UI</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------
    # Future Enhancements
    # ------------------------------------------------------
    st.markdown(
        """
        <div class="info-card">
            <h3>🚀 Future Enhancements</h3>
            <ul>
                <li>Hyperparameter tuning</li>
                <li>Compare multiple ML models</li>
                <li>Cloud deployment</li>
                <li>Patient report generation (PDF)</li>
                <li>User authentication</li>
                <li>Database integration</li>
                <li>Real-time prediction API</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------
    # Developer
    # ------------------------------------------------------
    st.markdown(
        """
        <div class="dev-card">
            <h3>👩‍💻 Developer</h3>
            <p><b>Diya Mohan</b></p>
            <p>Machine Learning • Data Analytics • Python • Power BI</p>
            <p style="margin-top:0.75rem; opacity:0.9;">
                This project demonstrates the application of machine learning and
                Streamlit for building practical healthcare prediction systems.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()