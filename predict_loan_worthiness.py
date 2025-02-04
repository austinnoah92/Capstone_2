import requests

# URL of your prediction API (adjust host and port as needed)
url = 'http://localhost/predict'


data = {
    "ID": 101,                        # Example integer ID (if needed; your feature_engineering may ignore/drop it)
    "Loan_ID": "LP001002",             # Example loan identifier (usually dropped)
    "Gender": 1,                       # 1 might represent 'Male' and 0 'Female' (as per your dataset)
    "Married": 1,                      # 1 for married, 0 for not married
    "Dependents": "0",                 # As stored as an object (could be "0", "1", "2", or "3+")
    "Education": 1,                    # 1 for Graduate, 0 for Non-Graduate
    "Self_Employed": 0,                # 1 if self-employed, 0 otherwise
    "ApplicantIncome": 5000,           # Example income value
    "CoapplicantIncome": 1500.0,       # Example coapplicant income (float)
    "LoanAmount": 150,                 # Example loan amount (in thousands, for instance)
    "Loan_Amount_Term": 360,           # Example loan term in months
    "Credit_History": 1,               # 1 indicates good credit history, 0 indicates bad
    "Property_Area": 1,                 # Depending on your encoding, e.g., 1 might represent 'Urban'
    "Total_Income": 6500               # Example total income (if needed)
}

# Send the POST request with the raw data payload
response = requests.post(url, json=data).json()

# Print out the API response
print("API Response:")
print(response)
