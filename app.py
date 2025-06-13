from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load model and scalers
model = pickle.load(open("model.pkl", "rb"))
ms = pickle.load(open("minmaxscaler.pkl", "rb"))
sc = pickle.load(open("standscaler.pkl", "rb"))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route("/")
def index():
    return "Crop Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Get input values
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Apply scaling (very important)
        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled_data = ms.transform(input_data)
        final_data = sc.transform(scaled_data)

        # Predict and map to crop name
        pred = model.predict(final_data)
        crop = crop_dict.get(pred[0], "Unknown Crop")

        return jsonify({"crop": crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
