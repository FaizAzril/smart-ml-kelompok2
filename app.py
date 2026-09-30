
import base64
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request

from training.preprocessing import (
    FEATURE_COLS,
    FINAL_FEATURE_COLS,
    feature_engineering,
)

app = Flask(__name__)

# Model klasifikasi
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/standard_scaler.pkl")

# Model clustering
clustering_model = joblib.load("models/kmeans_model.pkl")
clustering_scaler = joblib.load("models/unsupervised_scaler.pkl")
pca_model = joblib.load("models/pca_2d.pkl")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "steel_plates_clean.csv"


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

    example_values = (
        pd.read_csv(BASE_DIR / "data" / "test.csv")
        .iloc[0][FEATURE_COLS]
        .to_dict()
    )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        feature_cols=FEATURE_COLS,
        example_values=example_values,
    )


@app.route("/clustering", methods=["GET"])
def clustering():
    # Muat dataset yang telah diproses
    df = pd.read_csv(DATA_PATH)

    # Pastikan semua fitur yang dibutuhkan tersedia
    missing_features = [
        col for col in FINAL_FEATURE_COLS
        if col not in df.columns
    ]

    if missing_features:
        return (
            "Kolom dataset tidak lengkap: "
            + ", ".join(missing_features),
            500,
        )

    # Scaling menggunakan scaler khusus clustering
    X = df[FINAL_FEATURE_COLS]
    X_scaled = clustering_scaler.transform(X)

    # Prediksi cluster dan koordinat PCA
    clusters = clustering_model.predict(X_scaled)
    coordinates = pca_model.transform(X_scaled)

    # Buat visualisasi PCA 2D
    fig, ax = plt.subplots(figsize=(9, 6))

    for cluster_id in sorted(set(clusters)):
        mask = clusters == cluster_id

        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.65,
            s=24,
        )

    ax.set_title("Visualisasi Clustering PCA 2D")
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    ax.legend()
    ax.grid(alpha=0.2)

    # Konversi gambar menjadi Base64 agar bisa ditampilkan di HTML
    image_buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(
        image_buffer,
        format="png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

    plot_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    # Ringkasan jumlah data per cluster
    cluster_counts = (
        pd.Series(clusters)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return render_template(
        "clustering.html",
        plot_image=plot_image,
        cluster_counts=cluster_counts,
        total_data=len(df),
    )


if __name__ == "__main__":
    app.run(debug=True)