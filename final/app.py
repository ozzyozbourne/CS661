import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('diabetes.csv')

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Streamlit app layout
st.title("Diabetes Risk Prediction App")

# User input
bmi = st.slider("BMI", 20.0, 40.0, 25.0)
glucose = st.slider("Glucose Level", 70, 200, 100)
age = st.slider("Age", 18, 100, 30)
blood_pressure = st.slider("Blood Pressure", 80, 180, 120)

# Prepare input for prediction
user_input = np.array([[bmi, glucose, age, blood_pressure]])
user_input_scaled = scaler.transform(user_input)

# Predictions
log_reg_pred = log_reg.predict_proba(user_input_scaled)[:, 1][0]
rf_pred = rf.predict_proba(user_input_scaled)[:, 1][0]

# Display results
st.write("### Prediction Results")
st.write(f"*Logistic Regression Probability of Diabetes:* {log_reg_pred:.2f}")
st.write(f"*Random Forest Probability of Diabetes:* {rf_pred:.2f}")

# Recommendations
if log_reg_pred > 0.5 or rf_pred > 0.5:
    st.write("*Recommendation:* Consult a healthcare provider.")
else:
    st.write("*Recommendation:* Your risk of diabetes seems low. Maintain a healthy lifestyle.")
