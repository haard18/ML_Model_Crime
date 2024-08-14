from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
with open('trained_model.sav', 'rb') as file:
    model = pickle.load(file)

# Assuming the scaler was saved along with the model
scaler = StandardScaler()

# List of features to be provided as input (the original 16)
features = ['arrest', 'domestic', 'beat', 'district', 'ward', 'community_area',
            'year', 'latitude', 'longitude', 'day_of_week', 'month', 'time', 
            'zone', 'season', 'loc_grouped']
@app.route('/', methods=['GET'])
def hello():
    return 'Hello from model!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Ensure all 16 inputs are provided
        input_data = [data[feature] for feature in features]
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data], columns=features)

        # Apply any preprocessing steps (e.g., one-hot encoding, scaling)
        df_encoded = pd.get_dummies(df)  # Example: You might need more complex preprocessing

        # Ensure the input data matches the training columns (reindex)


        # Scaling
        df_scaled = scaler.transform(df_encoded)

        # Prediction
        prediction = model.predict(df_scaled)

        return jsonify({'prediction': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
