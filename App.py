from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Load episodic model (your custom model class)
# ---------------------------------------------------
from episodic_model import EpisodicImprovementModel   # <-- ملف الموديل (هقولك تعمليه تحت)

app = Flask(__name__)

# Load saved model
model_path = "episodic_model.pkl"
model = EpisodicImprovementModel.load(model_path)


@app.route("/", methods=["GET"])
def home():
    return {"message": "Episodic Model is running successfully on Render!"}


# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert data to dataframe
        df = pd.DataFrame([data])

        # Predict improvement
        prediction = model.predict(df)

        return jsonify({
            "improvement_prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
