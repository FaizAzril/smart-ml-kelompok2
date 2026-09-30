"""Reusable preprocessing utilities for the Steel Plates Faults dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

FEATURE_COLS = [
    "X_Minimum",
    "X_Maximum",
    "Y_Minimum",
    "Y_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "Outside_Global_Index",
    "LogOfAreas",
    "Log_X_Index",
    "Log_Y_Index",
    "Orientation_Index",
    "Luminosity_Index",
    "SigmoidOfAreas",
]

ENGINEERED_FEATURE_COLS = [
    "Defect_Width",
    "Defect_Height",
]

FINAL_FEATURE_COLS = FEATURE_COLS + ENGINEERED_FEATURE_COLS


def load_data(file_path):
    """Load the Steel Plates Faults dataset from a CSV file."""
    return pd.read_csv(file_path)


def create_target(df):
    """Convert the seven one-hot fault columns into Defect_Type."""
    data = df.copy()

    missing_targets = [
        col for col in TARGET_COLS
        if col not in data.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    active_labels = data[TARGET_COLS].sum(axis=1)
    invalid_rows = (active_labels != 1).sum()

    if invalid_rows > 0:
        raise ValueError(
            f"Found {invalid_rows} rows with a number of "
            "active labels different from 1."
        )

    data["Defect_Type"] = data[TARGET_COLS].idxmax(axis=1)

    return data


def feature_engineering(df):
    """Add engineered defect size features."""
    data = df.copy()

    required_cols = [
        "X_Minimum",
        "X_Maximum",
        "Y_Minimum",
        "Y_Maximum",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in data.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns for feature engineering: {missing_cols}"
        )

    data["Defect_Width"] = (
        data["X_Maximum"] - data["X_Minimum"]
    )

    data["Defect_Height"] = (
        data["Y_Maximum"] - data["Y_Minimum"]
    )

    return data


def prepare_dataset(df):
    """Create the final feature matrix X and target vector y."""
    data = create_target(df)
    data = feature_engineering(data)

    missing_features = [
        col for col in FINAL_FEATURE_COLS
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    X = data[FINAL_FEATURE_COLS].copy()
    y = data["Defect_Type"].copy()

    return data, X, y


def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """Split the dataset while preserving class proportions."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_features(X_train, X_test):
    """Fit StandardScaler on training data and transform both datasets."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def save_scaler(scaler, file_path):
    """Save a fitted scaler to a .pkl file."""
    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        output_path
    )


if __name__ == "__main__":
    print(
        "Steel Plates Faults preprocessing "
        "module loaded successfully."
    )

    print(
        f"Base features : {len(FEATURE_COLS)}"
    )

    print(
        f"Final features: {len(FINAL_FEATURE_COLS)}"
    )

    print(
        f"Target classes: {len(TARGET_COLS)}"
    )