import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Employee_ID": range(1, n+1),
    "Age": np.random.randint(22, 60, n),
    "Experience_Years": np.random.randint(0, 35, n),
    "Department": np.random.choice(
        ["Sales", "IT", "HR", "Finance", "Marketing"], n
    ),
    "Salary": np.random.randint(20000, 150000, n),
    "Training_Hours": np.random.randint(0, 100, n),
    "Attendance_Percentage": np.random.randint(50, 100, n),
    "Projects_Completed": np.random.randint(1, 20, n),
    "Overtime_Hours": np.random.randint(0, 60, n)
})

def calculate_performance(row):

    score = (
        row["Training_Hours"] * 0.4 +
        row["Attendance_Percentage"] * 0.5 +
        row["Projects_Completed"] * 4 -
        row["Overtime_Hours"] * 0.3 +
        (row["Experience_Years"] * 0.2)
    )

    # STRONGER CLASS SEPARATION
    if score > 140:
        return "High"
    elif score > 90:
        return "Medium"
    else:
        return "Low"

data["Performance"] = data.apply(calculate_performance, axis=1)

file_path = "data/employee_data.csv"
data.to_csv(file_path, index=False)

print("Dataset created successfully!")
print("Saved at:", file_path)
print(data.head())
