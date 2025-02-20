import streamlit as st
import pickle
import numpy as np
import os

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scaler
model_path = os.path.join(current_dir, "model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Title
st.title("Titanic Survival Prediction")

# User inputs
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare Paid", min_value=0.0, max_value=500.0, value=10.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical values
Sex = 1 if Sex == "Male" else 0
Embarked_dict = {"C": 0, "Q": 1, "S": 2}
Embarked = Embarked_dict[Embarked]

# Prediction button
if st.button("Predict"):
    # Prepare input data
    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    features = scaler.transform(features)
    prediction = model.predict(features)[0]

    # Display result
    result = "Survived" if prediction == 1 else "Not Survived"
    st.success(f"Prediction: {result}")
