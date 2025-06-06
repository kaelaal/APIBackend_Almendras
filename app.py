from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line

# Load model and features
cls_model = joblib.load('trained_data/model_cls.pkl')
features = joblib.load('trained_data/model_features.pkl')
label_encoder = joblib.load('trained_data/label_encoder.pkl')

@app.route('/gradestudent', methods=['POST'])
def gradestudent():

  
    data = request.json

    # Remove 'Parental_Education_Level' if accidentally included
    data.pop('Parental_Education_Level', None)

    # Prepare input DataFrame
    input_df = pd.DataFrame([data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    # Predict
    result = cls_model.predict(input_encoded)[0]
    label = label_encoder.inverse_transform([result])[0]

    return jsonify({"Prediction": label})

if __name__ == '__main__':
    app.run(debug=True)
