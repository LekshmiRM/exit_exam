import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from pickle file
with open("eurovision_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to DataFrame (keys must match your selected features)
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Return result as JSON
        return jsonify({"predicted_points": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
