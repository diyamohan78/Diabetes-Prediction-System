# ==========================================================
# Diabetes Prediction System
# Prediction Page
# ==========================================================
# Professional Edition
# Author: Diya Mohan
# ==========================================================

import os

import numpy as np
import joblib
import streamlit as st
import db_utils

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

PRIMARY_COLOR = "#0F766E"
DANGER_COLOR = "#DC2626"


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

            .result-card {{
                border-radius: 16px;
                padding: 1.5rem 1.75rem;
                margin-top: 0.5rem;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
            }}
            .result-card.negative {{
                background-color: #ECFDF5;
                border-left: 6px solid #059669;
            }}
            .result-card.positive {{
                background-color: #FEF2F2;
                border-left: 6px solid {DANGER_COLOR};
            }}
            .result-title {{
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }}
            .result-title.negative {{ color: #065F46; }}
            .result-title.positive {{ color: #991B1B; }}
            .result-sub {{
                color: #475569;
                font-size: 0.92rem;
            }}

            .patient-card {{
                background-color: #FFFFFF;
                border-radius: 14px;
                padding: 1.25rem 1.5rem;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
                border-left: 5px solid {PRIMARY_COLOR};
                margin-bottom: 1rem;
            }}
            .patient-card h4 {{
                margin-top: 0;
                color: #0F172A;
            }}

            .saved-banner {{
                background-color: #ECFDF5;
                border-left: 5px solid #059669;
                border-radius: 10px;
                padding: 0.75rem 1rem;
                color: #065F46;
                font-weight: 600;
                margin-top: 0.5rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Model Loading (cached, with graceful error handling)
# ==========================================================

@st.cache_resource(show_spinner="Loading model...")
def load_artifacts(model_path: str, scaler_path: str):
    missing = [p for p in (model_path, scaler_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(f"`{m}`" for m in missing)
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ==========================================================
# Page Content
# ==========================================================

def main() -> None:
    inject_custom_css()
    db_utils.init_db()

    st.title("🩺 Diabetes Prediction")
    st.write("Enter the patient's details below to predict whether they are diabetic or not.")
    st.divider()

    try:
        model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
    except FileNotFoundError as e:
        st.error(
            f"⚠️ {e}\n\n"
            "Please make sure `model.pkl` and `scaler.pkl` are present in the "
            "app's root directory before using this page."
        )
        st.stop()

    # ------------------------------------------------------
    # Patient Details
    # ------------------------------------------------------
    st.markdown('<div class="patient-card">', unsafe_allow_html=True)
    st.markdown("#### 🧾 Patient Details")

    name_col, contact_col = st.columns(2)
    with name_col:
        full_name = st.text_input("Full Name *", placeholder="e.g. Priya Sharma")
    with contact_col:
        contact_number = st.text_input("Contact Number", placeholder="e.g. 98765 43210")

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Input Fields
    # ------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.5)
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.50
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    st.markdown("")

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------
    if st.button("🔍 Predict Diabetes", use_container_width=True, type="primary"):

        if not full_name.strip():
            st.error("⚠️ Please enter the patient's full name before predicting.")
            st.stop()

        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
        ]])

        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
            st.stop()

        st.divider()
        st.subheader("Prediction Result")

        is_diabetic = prediction[0] == 1
        diabetic_prob = float(probability[0][1])
        non_diabetic_prob = float(probability[0][0])

        # Persist this patient + prediction to the local database
        try:
            record_id = db_utils.insert_record(
                full_name=full_name,
                contact_number=contact_number,
                pregnancies=int(pregnancies),
                glucose=float(glucose),
                blood_pressure=float(blood_pressure),
                skin_thickness=float(skin_thickness),
                insulin=float(insulin),
                bmi=float(bmi),
                diabetes_pedigree=float(diabetes_pedigree),
                age=int(age),
                prediction="Diabetic" if is_diabetic else "Non-Diabetic",
                probability_diabetic=diabetic_prob,
                probability_non_diabetic=non_diabetic_prob,
            )
            st.markdown(
                f"""
                <div class="saved-banner">
                    💾 Record saved to database (ID #{record_id}) for
                    <b>{full_name}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"⚠️ Prediction succeeded but saving to the database failed: {e}")

        if is_diabetic:
            st.markdown(
                f"""
                <div class="result-card positive">
                    <div class="result-title positive">⚠️ Likely Diabetic</div>
                    <div class="result-sub">
                        The model estimates a {diabetic_prob*100:.1f}% probability of diabetes
                        based on the entered parameters. This is not a medical diagnosis —
                        please consult a healthcare professional.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card negative">
                    <div class="result-title negative">✅ Likely Non-Diabetic</div>
                    <div class="result-sub">
                        The model estimates a {non_diabetic_prob*100:.1f}% probability of no diabetes
                        based on the entered parameters. This is not a medical diagnosis —
                        please consult a healthcare professional.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.subheader("Prediction Probability")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Non-Diabetic", f"{non_diabetic_prob*100:.2f}%")
        with col_b:
            st.metric("Diabetic", f"{diabetic_prob*100:.2f}%")

        st.progress(diabetic_prob)
        st.caption(
            "Progress bar reflects the model's predicted probability of a diabetic outcome."
        )

    # ------------------------------------------------------
    # Saved Patient Records
    # ------------------------------------------------------
    st.divider()
    with st.expander("📜 View Saved Patient Records"):
        records = db_utils.fetch_all_records()

        if records.empty:
            st.info("No patient records saved yet. Run a prediction above to add one.")
        else:
            st.dataframe(records, use_container_width=True, hide_index=True)

            col_dl, col_clear = st.columns([3, 1])
            with col_dl:
                st.download_button(
                    "⬇️ Download as CSV",
                    data=records.to_csv(index=False).encode("utf-8"),
                    file_name="patient_records.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_clear:
                if st.button("🗑️ Clear History", use_container_width=True):
                    db_utils.delete_all_records()
                    st.rerun()


if __name__ == "__main__":
    main()