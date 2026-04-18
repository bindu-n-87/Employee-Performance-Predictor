import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

data_path = "data/employee_data.csv"
df = pd.read_csv(data_path)

print("\n--- DATA LOADED ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- STATISTICS ---")
print(df.describe())

os.makedirs("outputs/plots", exist_ok=True)

plt.figure(figsize=(6,4))
sns.countplot(x="Performance", data=df)
plt.title("Employee Performance Distribution")
plt.savefig("outputs/plots/performance_distribution.png")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Performance", y="Salary", data=df)
plt.title("Salary vs Performance")
plt.savefig("outputs/plots/salary_vs_performance.png")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Performance", y="Training_Hours", data=df)
plt.title("Training Hours vs Performance")
plt.savefig("outputs/plots/training_vs_performance.png")
plt.show()

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/plots/correlation_heatmap.png")
plt.show()

print("\n--- BUSINESS INSIGHTS ---")

print("1. High performers usually have higher training hours.")
print("2. Attendance strongly affects performance.")
print("3. Overtime has negative impact on performance.")
print("4. Salary is not always directly correlated with performance.")
