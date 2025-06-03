from flask import Flask, render_template, request
import os
import logging
import numpy as np
import pandas as pd
from src.components.model_evaluation import ModelEvaluation
from src.utils import load_config, load_object

app = Flask(__name__)


# Load the configuration
try:
    config = load_config('xyz.yaml')
    logging.info("Configuration loaded successfully")
    
    # Load the trained model and preprocessor
    model_path = config['model_trainer']['trained_model_file_path']
    preprocessor_path = config['data_transformation']['preprocessor_ob_file_path']
    model = load_object(model_path)
    preprocessor = load_object(preprocessor_path)
    logging.info("Model and preprocessor loaded successfully")
    
except Exception as e:
    logging.error(f"Error during initialization: {str(e)}")
    raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input = {
            'Company': request.form['Company'],
            'TypeName': request.form['TypeName'],
            'Ram': int(request.form['Ram']),
            'Weight': float(request.form['Weight']),
            'TouchScreen': int(request.form.get('TouchScreen', 0)),
            'IPS': int(request.form.get('IPS', 0)),
            'PPI': float(request.form['PPI']),
            'HDD': int(request.form['HDD']),
            'SSD': int(request.form['SSD']),
            'Cpu_Brand': request.form['Cpu_Brand'],
            'Gpu_brand': request.form['Gpu_brand'],
            'os': request.form['os']
        }

        # Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Preprocess the input data
        input_features = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_features)
        predicted_price = np.exp(prediction[0])
        
        # Format the price nicely
        formatted_price = "{:,.2f}".format(predicted_price)

        # Return the prediction result
        return render_template('result.html', 
                             prediction=formatted_price,
                             user_input=user_input)

    except Exception as e:
        logging.error(f"An error occurred during prediction: {str(e)}")
        return render_template('error.html', error_message=str(e))
    
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port = 5000)