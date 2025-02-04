from flask import Flask, request, jsonify
import joblib
import pandas as pd
from feature_engineering import feature_engineering


model = joblib.load('best_model.pkl')              
dv = joblib.load('best_dv.pkl')                      
optimal_threshold = joblib.load('optimal_threshold.pkl') 

app = Flask('loan_credit_predict')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data. Expect a JSON object with feature name: value pairs.
        data = request.get_json(force=True)
        
        df = pd.DataFrame([data])
        
        del df['Total_Income']
        
        # Optionally, apply feature engineering if your model was trained on engineered features.
        df = feature_engineering(df)
        
        # Transform the data using the loaded DictVectorizer.
        X_new = dv.transform(df.to_dict(orient='records'))
        
        # Obtain the predicted probabilities for the positive class.
        proba = model.predict_proba(X_new)[:, 1]
        
        # Apply the optimal threshold to obtain a binary prediction.
        prediction = int(proba[0] >= optimal_threshold)
        
        # Return both the binary prediction and the probability.
        return jsonify({'prediction': prediction, 'probability': proba[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app. The host is set to 0.0.0.0 so that it is externally visible.
    app.run(debug=True, host='0.0.0.0', port=9990)
