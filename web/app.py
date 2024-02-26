import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
from joblib import load

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('your_model_file.pkl')  # Replace 'your_model_file.pkl' with the actual filename

# Create a label encoder for categorical variables
label_encoders = {}  # Store label encoders for later use

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area']
        }

        # Convert categorical variables using label encoders
        # for feature, value in input_data.items():
        #     if isinstance(value, str):
        #         if feature not in label_encoders:
        #             label_encoders[feature] = LabelEncoder()
        #             label_encoders[feature].fit(['Unknown', value])
        #         input_data[feature] = label_encoders[feature].transform([value])[0]

        # input_array = input_array.values()
        # Prepare the input data as a NumPy array
        input_array = pd.DataFrame(columns = list(input_data.keys()), data = np.array(list(input_data.values())).reshape(1, -1))


        categorical_columns = [ 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
            
        encoder = load('./encoder.joblib')
        scaler = load('./scaler.joblib')
        model = load('./model.joblib')
    
        # Make prediction
        for i in categorical_columns:
            input_array[i] = encoder.transform(input_array[i])
        for i in input_array.columns:
            input_array[i] = scaler.transform(input_array[i])
            
        prediction = model.predict(input_array)

        # Convert the prediction back to a meaningful result
        result = 'Yes' if prediction[0] == 1 else 'No'

        return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)