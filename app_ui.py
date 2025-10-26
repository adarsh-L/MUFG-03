# ==================================================
# Smart Manufacturing Output Estimator (Streamlit UI)
# Version 4.0 - Enhanced for Robustness & Presentation
# ==================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# Streamlit Page Settings
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Manufacturing Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------
model = pickle.load(open("models/output_predictor.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_output(Mold_Temp, Line_Pressure, Cycle_Duration, Cool_Period,
                   Viscosity_Index, Env_Temp, Machine_Years,
                   Operator_Skill, Maintenance_Time):
    # Create derived features based on input parameters
    ratio = Line_Pressure / Mold_Temp
    efficiency = (Operator_Skill / (Cycle_Duration + Cool_Period)) * 100
    cooling_eff = Cool_Period / Cycle_Duration
    features = np.array([[Mold_Temp, Line_Pressure, Cycle_Duration, Cool_Period,
                          Viscosity_Index, Env_Temp, Machine_Years,
                          Operator_Skill, Maintenance_Time,
                          ratio, efficiency, cooling_eff]])
    # Scale features before prediction
    scaled = scaler.transform(features)
    # Predict output
    prediction = model.predict(scaled)[0]
    return prediction


# --------------------------------------------------
# Sidebar Inputs for User to Enter Machine Parameters
# --------------------------------------------------
with st.sidebar:
    st.header("Enter Machine Parameters 🔧")
    Mold_Temp = st.slider("Mold Temperature (°C)", 150, 300, 230)
    Line_Pressure = st.slider("Line Pressure (bar)", 50, 180, 120)
    Cycle_Duration = st.slider("Cycle Duration (s)", 10, 50, 25)
    Cool_Period = st.slider("Cooling Period (s)", 5, 25, 12)
    Viscosity_Index = st.number_input("Material Viscosity", 100.0, 400.0, 250.0)
    Env_Temp = st.slider("Ambient Temperature (°C)", 15, 35, 25)
    Machine_Years = st.slider("Machine Age (Years)", 1, 15, 7)
    Operator_Skill = st.slider("Operator Skill (months)", 1, 120, 48)
    Maintenance_Time = st.slider("Maintenance Hours", 0, 200, 50)


# --------------------------------------------------
# Main App Title and Description
# --------------------------------------------------
st.title("🏭 Smart Manufacturing Output Estimator")
st.write("This tool predicts the hourly production (parts/hour) of manufacturing equipment "
         "based on operational parameters like temperature, pressure, cooling time, and operator experience.")


# --------------------------------------------------
# Prediction and Visualization Logic
# --------------------------------------------------
if st.button("Estimate Output"):
    result = predict_output(Mold_Temp, Line_Pressure, Cycle_Duration, Cool_Period,
                            Viscosity_Index, Env_Temp, Machine_Years, Operator_Skill, Maintenance_Time)
    
    # Determine performance category and suggest action
    performance = "Optimal ✅" if result > 300 else "Below Average ⚠️"
    tip = "Maintain steady pressure and reduce cooling time." if result < 300 else "Maintain current operational settings."
    
    st.success(f"**Predicted Parts Per Hour:** {round(result, 2)}")
    st.info(f"Machine Performance: {performance}")
    st.write("💡 Suggestion:", tip)
    
    # Log prediction results to a text file
    os.makedirs("logs", exist_ok=True)
    safe_perf = performance.encode("ascii", "ignore").decode()
    with open("logs/prediction_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | Predicted={result:.2f} | Performance={safe_perf}\n")

    # Visualize entered machine parameters in a bar chart
    st.subheader("Feature Impact Visualization")
    feature_values = [Mold_Temp, Line_Pressure, Cycle_Duration, Cool_Period, Viscosity_Index]
    labels = ["Temperature", "Pressure", "Cycle", "Cooling", "Viscosity"]

    fig, ax = plt.subplots()
    ax.bar(labels, feature_values, color="steelblue")
    ax.set_ylabel("Parameter Values")
    ax.set_title("Entered Machine Parameter Values")
    st.pyplot(fig)


# --------------------------------------------------
# Real-Time Prediction History Viewer
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 Prediction History Tracker")

if os.path.exists("logs/prediction_log.txt"):
    with open("logs/prediction_log.txt", encoding="utf-8") as f:
        logs = [line.strip().split("|") for line in f.readlines()]
    df_log = pd.DataFrame(logs, columns=["Timestamp", "Predicted", "Performance"])
    df_log = df_log.tail(10)  # Show last 10 entries
    st.table(df_log)

    # Downloadable CSV history
    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Full History", data=csv, file_name="prediction_history.csv")
else:
    st.warning("No prediction logs found. Run predictions first!")


# --------------------------------------------------
# Interactive Quick EDA Section
# --------------------------------------------------
st.markdown("---")
st.subheader("🧠 Quick Data Analysis (Upload CSV)")

uploaded = st.file_uploader("Upload Your Manufacturing Dataset (CSV)", type=["csv"])
if uploaded:
    user_df = pd.read_csv(uploaded)
    st.write("Dataset Preview:")
    st.dataframe(user_df.head())

    numeric_cols = user_df.select_dtypes(include='number').columns.tolist()

    st.write("**Summary Statistics**")
    st.write(user_df.describe())

    st.write("**Correlation Heatmap**")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(user_df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    st.write("**Boxplots For Outlier Detection**")
    for col in numeric_cols:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=user_df[col], color='skyblue', ax=ax_box)
        ax_box.set_title(f"Boxplot for {col}")
        st.pyplot(fig_box)

    st.write("**Histograms**")
    fig_hist, axs_hist = plt.subplots(nrows=(len(numeric_cols)+2)//3, ncols=3, figsize=(15, 8))
    axs_hist = axs_hist.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(user_df[col], bins=20, ax=axs_hist[i])
        axs_hist[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig_hist)

    st.write("**Pairplot (Feature Relationships)**")
    pairplot_fig = sns.pairplot(user_df[numeric_cols].sample(min(200, len(user_df))))
    st.pyplot(pairplot_fig)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Developed by Adarsh L | Capstone Project 2025 | Smart Manufacturing Analytics Dashboard")
st.caption("© 2025 Smart Manufacturing Solutions. All rights reserved.")