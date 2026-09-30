
import joblib
import pandas as pd
from flask import Flask, render_template, request

from training.preprocessing import (
    FEATURE_COLS,
    FINAL_FEATURE_COLS,
    feature_engineering,
)

app = Flask(__name__)

model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/standard_scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = {
                col: float(request.form[col])
                for col in FEATURE_COLS
            }

            data = pd.DataFrame([values])
            data = feature_engineering(data)
            X = data[FINAL_FEATURE_COLS]

            X_scaled = scaler.transform(X)
            prediction = str(model.predict(X_scaled)[0])

        except (KeyError, ValueError):
            error = "Input tidak valid. Periksa kembali semua nilai."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        feature_cols=FEATURE_COLS,
        example_values=pd.read_csv("data/test.csv").iloc[0][FEATURE_COLS].to_dict(),
    )


if __name__ == "__main__":
    app.run(debug=True)