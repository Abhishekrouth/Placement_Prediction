from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import joblib

app= Flask(__name__)
model = joblib.load('placement.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['10th %'], data['10th Board'],data['12th %'],data['12th Board'],data['12th Stream'],
                data['Degree %'], data['Degree Type'],data['Work Experience'],data['Employability Test %'],
                data['Specialisation'],data['MBA %']]
    
    prediction = model.predict(np.array([features]))
    pred= int(prediction[0])
    if pred==1:
        return jsonify({'prediction': 'Placed'})
    else:
        return jsonify({'prediction': 'Not Placed'})
if __name__ == '__main__':
    app.run(debug=True)

    
