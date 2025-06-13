from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load only the model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def index():
    return "Crop Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        pred = model.predict(input_data)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        crop = crop_dict.get(pred[0], "Unknown Crop")
        return jsonify({"crop": crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
