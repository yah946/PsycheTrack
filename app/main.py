
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Mental Health Risk Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Mental Health Risk Prediction")
st.write("Enter the student's information to predict their mental health risk.")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/compare/model/classifier.joblib")

model = load_model()

# -----------------------------
# Input form
# -----------------------------
with st.form("prediction_form"):

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 17, 29, 21)
        gender = st.selectbox(
            "Gender", ["Female", "Male", "Non-binary"]
        )

    with col2:
        year = st.selectbox(
            "Year of Study", [1, 2, 3, 4, 5]
        )

        previous_history = st.selectbox(
            "Previous Mental Health History", ["No", "Yes"]
        )

    st.divider()
    st.subheader("Academic Information")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Study Hours Per Week", 2.4, 38.3, 20.0
        )

        gpa = st.number_input(
            "GPA", 1.5, 3.66, 2.56
        )

    with col2:
        attendance = st.number_input(
            "Attendance Rate (%)", 54.4, 100.0, 85.0
        )

        part_time_job = st.selectbox(
            "Part-Time Job", ["No", "Yes"]
        )

    st.divider()
    st.subheader("Lifestyle Information")

    col1, col2 = st.columns(2)

    with col1:
        sleep = st.number_input(
            "Sleep Hours", 3.3, 10.0, 7.0
        )

        social_media = st.number_input(
            "Social Media Hours", 2.1, 10.0, 5.0
        )

        physical_activity = st.selectbox(
            "Physical Activity", ["High", "Moderate", "Low"]
        )

    with col2:
        screen_time = st.number_input(
            "Screen Time Hours", 3.3, 14.0, 8.0
        )

        diet = st.selectbox(
            "Diet Quality", ["Good", "Average", "Poor"]
        )

    st.divider()
    st.subheader("Psychological and Social Factors")

    col1, col2 = st.columns(2)

    with col1:
        financial_stress = st.number_input(
            "Financial Stress", 1.4, 10.0, 5.0
        )

    with col2:
        family_support = st.number_input(
            "Family Support", 1.7, 9.9, 7.0
        )

    submitted = st.form_submit_button(
        "Predict Mental Health Risk",
        use_container_width=True
    )

# -----------------------------
# Validation and prediction
# -----------------------------
if submitted:

    errors = []

    # Business validation
    if social_media >= screen_time:
        errors.append(
            "Social Media Hours must be less than Screen Time Hours."
        )

    if errors:
        for error in errors:
            st.error(error)

    else:

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year,
            "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study_hours,
            "GPA": gpa,
            "Social_Media_Hours": social_media,
            "Screen_Time_Hours": screen_time,
            "Physical_Activity": physical_activity,
            "Financial_Stress": financial_stress,
            "Family_Support": family_support,
            "Attendance_Rate": attendance,
            "Part_Time_Job": part_time_job,
            "Diet_Quality": diet,
            "Previous_Mental_Health_History": previous_history
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "High":
            st.error("High Mental Health Risk")

        elif prediction == "Medium":
            st.warning("Medium Mental Health Risk")

        else:
            st.success("Low Mental Health Risk")
