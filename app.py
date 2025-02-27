from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    Pclass = int(request.form['Pclass'])
    Sex = 1 if request.form['Sex'] == "Male" else 0
    Age = float(request.form['Age'])
    SibSp = int(request.form['SibSp'])
    Parch = int(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked_dict = {"C": 0, "Q": 1, "S": 2}
    Embarked = Embarked_dict[request.form['Embarked']]

    # Prepare input array
    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    
    # Transform input
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    result = "Survived" if prediction == 1 else "Not Survived"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
