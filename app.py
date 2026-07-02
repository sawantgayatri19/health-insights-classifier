from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from llm_explainer import HealthExplainer
import numpy as np

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("models/classifier.pkl")
except:
    model = None

explainer = HealthExplainer(model=model)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Health Insights Classifier API", "version": "1.0.0"}), 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not model:
            return jsonify({"error": "Model not loaded"}), 500
        features = np.array([[data.get("age", 0), data.get("bmi", 0)]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max() * 100
        return jsonify({"risk_score": round(probability, 2), "prediction": int(prediction)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
