import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'
COLUMNS_PATH = 'models/columns.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
else:
    model, scaler, model_columns = None, None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model non-existent. Run train.py first.'}), 500

    try:
        data = request.get_json()

        # Initialize base dataframe with zero values matching training columns
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Set numeric inputs
        input_df['sqft'] = float(data['sqft'])
        input_df['bedrooms'] = float(data['bedrooms'])
        input_df['bathrooms'] = float(data['bathrooms'])
        input_df['age'] = float(data['age'])

        # Set categorical location dummy column
        loc_col = f"location_{data['location']}"
        if loc_col in input_df.columns:
            input_df[loc_col] = 1

        # Transform and Predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)