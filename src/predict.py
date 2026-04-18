import numpy as np
import joblib

# -----------------------------
# LOAD MODEL + ENCODERS
# -----------------------------
model = joblib.load("models/performance_model.pkl")
dept_encoder = joblib.load("models/department_encoder.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

def predict_performance():

    print("\n--- HR PERFORMANCE PREDICTOR ---")

    age = int(input("Age: "))
    experience = int(input("Experience (Years): "))
    department = input("Department (Sales/IT/HR/Finance/Marketing): ")
    salary = int(input("Salary: "))
    training = int(input("Training Hours: "))
    attendance = int(input("Attendance %: "))
    projects = int(input("Projects Completed: "))
    overtime = int(input("Overtime Hours: "))

    # encode department
    try:
        dept_encoded = dept_encoder.transform([department])[0]
    except:
        print("Unknown department → defaulting to 0")
        dept_encoded = 0

    # -----------------------------
    # MATCH TRAINING FEATURE ORDER EXACTLY
    # -----------------------------
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

    print("\n======================")
    print("PREDICTION:", result)
    print("======================")

    if result == "High":
        print("Recommendation: Promote Employee")
    elif result == "Medium":
        print("Recommendation: Training Recommended")
    else:
        print("Recommendation: Performance Improvement Plan")

if __name__ == "__main__":
    predict_performance()