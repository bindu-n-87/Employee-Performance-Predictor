import streamlit as st
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/performance_model.pkl")
dept_encoder = joblib.load("models/department_encoder.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

st.title("Employee Performance Prediction System")
st.markdown("AI-powered HR decision support tool")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Employee Input Form")

age = st.sidebar.number_input("Age", 18, 65, 30)
experience = st.sidebar.number_input("Experience", 0, 40, 5)

department = st.sidebar.selectbox(
    "Department",
    ["Sales", "IT", "HR", "Finance", "Marketing"]
)

salary = st.sidebar.number_input("Salary", 20000, 200000, 50000)
training = st.sidebar.slider("Training Hours", 0, 100, 20)
attendance = st.sidebar.slider("Attendance %", 0, 100, 85)
projects = st.sidebar.slider("Projects Completed", 0, 30, 5)
overtime = st.sidebar.slider("Overtime Hours", 0, 100, 10)

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("Predict Performance"):

    dept_encoded = dept_encoder.transform([department])[0]

    input_data = np.array([[
        age,
        experience,
        dept_encoded,
        salary,
        training,
        attendance,
        projects,
        overtime
    ]])

    prediction = model.predict(input_data)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    # -----------------------------
    # DISPLAY RESULT
    # -----------------------------
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if result == "High":
            st.success("High Performer")
        elif result == "Medium":
            st.warning("Medium Performer")
        else:
            st.error("Low Performer")

    with col2:
        st.info("HR Recommendation")

        if result == "High":
            st.write("✔ Promotion Recommended")
        elif result == "Medium":
            st.write("✔ Training Recommended")
        else:
            st.write("✔ Performance Improvement Plan Required")

# -----------------------------
# FOOTER INSIGHT
# -----------------------------
st.markdown("---")
st.markdown("Built using Machine Learning + Streamlit")