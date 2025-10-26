# ==============================================================
# Smart Manufacturing Output Estimator - Model Training Script
# Enhanced Version with Outlier Removal, Residual Plot, and EDA
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import os

# ====================================================
# Step 1: Load Dataset
# ====================================================
df = pd.read_csv("data/manufacturing_dataset_1000_samples.csv")
df.columns = df.columns.str.strip()  # remove trailing spaces

df = df.rename(columns={
    "Injection_Temperature": "Mold_Temp",
    "Injection_Pressure": "Line_Pressure",
    "Cycle_Time": "Cycle_Duration",
    "Cooling_Time": "Cool_Period",
    "Material_Viscosity": "Viscosity_Index",
    "Ambient_Temperature": "Env_Temp",
    "Machine_Age": "Machine_Years",
    "Operator_Experience": "Operator_Skill",
    "Maintenance_Hours": "Maintenance_Time",
    "Parts_Per_Hour": "Output_Rate"
})

df = df.dropna()
print(f"Dataset Loaded: {df.shape[0]} rows and {df.shape[1]} columns.")

# ====================================================
# Step 2: Outlier Detection and Removal
# ====================================================
print("\nPerforming Outlier Analysis and Removal...")
for col in ["Mold_Temp", "Line_Pressure", "Cycle_Duration", "Cool_Period"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = df.shape[0]
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    after = df.shape[0]
    print(f" - {col}: removed {before - after} outliers")

# ====================================================
# Step 3: Feature Engineering
# ====================================================
df["Pressure_Temp_Ratio"] = df["Line_Pressure"] / df["Mold_Temp"]
df["Efficiency_Index"] = (df["Operator_Skill"] / (df["Cycle_Duration"] + df["Cool_Period"])) * 100
df["Cooling_Efficiency"] = df["Cool_Period"] / df["Cycle_Duration"]
print("\nFeature engineering completed.")

# ====================================================
# Step 4: Define Features and Target
# ====================================================
X = df[[
    "Mold_Temp", "Line_Pressure", "Cycle_Duration", "Cool_Period",
    "Viscosity_Index", "Env_Temp", "Machine_Years", "Operator_Skill",
    "Maintenance_Time", "Pressure_Temp_Ratio", "Efficiency_Index", "Cooling_Efficiency"
]]
y = df["Output_Rate"]

# ====================================================
# Step 5: Data Scaling and Splitting
# ====================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)
print("\nData split into training and testing sets.")

# ====================================================
# Step 6: Model Training (Ridge Regression)
# ====================================================
model = Ridge(alpha=0.5)
model.fit(X_train, y_train)
print("Model trained successfully.\n")

# ====================================================
# Step 7: Evaluation Metrics
# ====================================================
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score  : {r2:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")

# ====================================================
# Step 8: Model and Scaler Export
# ====================================================
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/output_predictor.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
print("\nModel and scaler saved successfully to 'models/' folder.")

# ====================================================
# Step 9: Visualizations
# ====================================================
os.makedirs("visuals", exist_ok=True)

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, color="orchid")
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.title("Actual vs Predicted Output")
plt.tight_layout()
plt.savefig("visuals/prediction_graph.png", dpi=150)
plt.close()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_pred, y=residuals, color="teal")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Model Validation)")
plt.tight_layout()
plt.savefig("visuals/residuals.png", dpi=150)
plt.close()

# Feature Coefficient Importance Plot
features = X.columns
coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
coeff_df = coeff_df.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Coefficient", y="Feature", data=coeff_df, palette="coolwarm")
plt.title("Feature Importance (Linear Coefficients)")
plt.tight_layout()
plt.savefig("visuals/feature_importance.png", dpi=150)
plt.close()

print("All visualizations saved successfully to 'visuals/' folder.")

# ====================================================
# Step 10: Print Completion Summary
# ====================================================
print("\n================ Training Summary ================")
print(f"Records Used: {len(df)}")
print(f"Final Model: Ridge Regression (α=0.5)")
print(f"Performance Metrics: R2={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
print("==================================================")
print("\nModel Training Complete & Files Saved.")
