from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the trained Decision Tree model
with open('dt_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store the uploaded data
uploaded_data = None

@app.route('/')
def index():
    return render_template('preview.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_data  # Use the global variable to store uploaded data
    if 'file' not in request.files:
        return redirect(url_for('index', error="No file part in the request."))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error="No file selected for uploading."))

    if file:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load CSV
        data = pd.read_csv(filepath)
        uploaded_data = data  # Store the uploaded data in the global variable

        # Validate feature columns
        expected_features = model.feature_names_in_  # Fetch directly from the model
        missing_features = [col for col in expected_features if col not in data.columns]

        if missing_features:
            return redirect(url_for('index', error=f"Missing required columns: {', '.join(missing_features)}"))

        # Ensure categorical features are encoded
        le = LabelEncoder()
        if 'type' in data.columns:
            data['type'] = le.fit_transform(data['type'])
        if 'nameOrig' in data.columns:
            data['nameOrig'] = le.fit_transform(data['nameOrig'])
        if 'nameDest' in data.columns:
            data['nameDest'] = le.fit_transform(data['nameDest'])

        # Store the features for later prediction
        global features
        features = data[expected_features].values

        return render_template('preview.html', uploaded=True)

    return redirect(url_for('index', error="Unexpected error occurred."))

@app.route('/predict', methods=['POST'])
def predict():
    global uploaded_data, features
    # Check if the uploaded data is None or empty
    if uploaded_data is None or uploaded_data.empty:
        return redirect(url_for('index', error="No file uploaded yet or the file is empty."))

    # Check which prediction mode the user selected
    mode = request.form.get('mode')

    if mode == 'single':
        # Get user input from form for single prediction
        single_transaction = {
            'step': int(request.form.get('step')),
            'amount': float(request.form.get('amount')),
            'oldbalanceOrg': float(request.form.get('oldbalanceOrg')),
            'newbalanceOrg': float(request.form.get('newbalanceOrg')),
            'oldbalanceDest': float(request.form.get('oldbalanceDest')),
            'newbalanceDest': float(request.form.get('newbalanceDest')),
            'type_CASH_OUT': int(request.form.get('type_CASH_OUT')),
            'type_DEBIT': int(request.form.get('type_DEBIT')),
            'type_PAYMENT': int(request.form.get('type_PAYMENT')),
            'type_TRANSFER': int(request.form.get('type_TRANSFER')),
            # Add missing columns
            'nameOrig': 0,  # You can encode this or use an appropriate value
            'nameDest': 0,  # You can encode this or use an appropriate value
            'isFlaggedFraud': 0  # Typically 0 or 1, adjust accordingly
        }

        # Convert to DataFrame
        single_df = pd.DataFrame([single_transaction])

        # Ensure categorical features are encoded (similar to how you handle other features)
        le = LabelEncoder()
        single_df['nameOrig'] = le.fit_transform(single_df['nameOrig'])
        single_df['nameDest'] = le.fit_transform(single_df['nameDest'])

        # Predict
        try:
            prediction = model.predict(single_df.values)
            prediction_label = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'
        except Exception as e:
            return redirect(url_for('index', error=f"Prediction error: {str(e)}"))

        return render_template('preview.html', single_prediction=prediction_label)

    elif mode == 'whole':
        # Predict for the entire dataset
        predictions = model.predict(features)
        uploaded_data['Prediction'] = np.where(predictions == 0, 'Not Fraudulent', 'Fraudulent')

        # Generate prediction table
        result_columns = list(uploaded_data.columns) + ['Prediction']
        prediction_table = uploaded_data[result_columns].to_html(index=False)

        return render_template('preview.html', prediction_table=prediction_table)

    return redirect(url_for('index', error="Invalid prediction mode selected."))


if __name__ == '__main__':
    app.run(debug=True)
