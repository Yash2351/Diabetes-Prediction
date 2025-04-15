from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[key]) for key in request.form]
    prediction = model.predict([features])[0]
    result = "The person is likely to have diabetes." if prediction == 1 else "The person is unlikely to have diabetes."
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
