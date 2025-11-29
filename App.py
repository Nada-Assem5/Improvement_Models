from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np


from episodic_model import EpisodicImprovementModel  

app = Flask(__name__)

model_path = "episodic_model.pkl"
model = EpisodicImprovementModel.load(model_path)


@app.route("/", methods=["GET"])
def home():
    return {"message": "Episodic Model is running successfully on Render!"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()


        df = pd.DataFrame([data])

        prediction = model.predict(df)

        return jsonify({
            "improvement_prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
