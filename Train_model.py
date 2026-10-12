
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DATA_FILE = "car_data.csv"
MODEL_FILE = "car_model.pkl"
PROCESSED_FILE = "car_data_processed.csv"

# Common brands: longer names are checked first.
KNOWN_BRANDS = [
    "Mercedes-Benz", "Land Rover", "Maruti Suzuki",
    "Rolls-Royce", "Force", "Volkswagen", "Mitsubishi",
    "Chevrolet", "Mahindra", "Renault", "Hyundai",
    "Maruti", "Toyota", "Honda", "Tata", "Ford",
    "Nissan", "Datsun", "Skoda", "Fiat", "BMW",
    "Audi", "Jaguar", "Volvo", "Jeep", "Kia",
    "MG", "Lexus", "Porsche", "Daewoo", "Isuzu",
    "Ambassador", "OpelCorsa"
]

KNOWN_BRANDS.sort(key=len, reverse=True)


def split_car_name(name):
    """Separate a listing name into brand and remaining model name."""
    name = str(name).strip()

    for brand in KNOWN_BRANDS:
        if name.lower().startswith(brand.lower() + " "):
            return brand, name[len(brand):].strip()

        if name.lower() == brand.lower():
            return brand, "Other"

    parts = name.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return name, "Other"


def main():
    print("Loading real car dataset...")

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "name", "year", "selling_price", "km_driven",
        "fuel", "transmission", "owner"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Columns found in CSV: {list(df.columns)}"
        )

    # Keep only useful columns and clean the data.
    if "seller_type" not in df.columns:
        df["seller_type"] = "Unknown"

    df = df[
        [
            "name", "year", "selling_price", "km_driven",
            "fuel", "seller_type", "transmission", "owner"
        ]
    ].copy()

    df["name"] = df["name"].astype("string").str.strip()

    for col in ["year", "selling_price", "km_driven"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows that cannot be used for price prediction.
    df = df.dropna(
        subset=["name", "year", "selling_price", "km_driven"]
    )

    df = df[
        (df["selling_price"] > 0)
        & (df["year"] >= 1980)
        & (df["year"] <= 2035)
        & (df["km_driven"] >= 0)
    ].copy()

    # Split full listing name into brand and model.
    df[["brand", "model"]] = df["name"].apply(
        lambda value: pd.Series(split_car_name(value))
    )

    categorical_features = [
        "brand", "model", "fuel", "seller_type",
        "transmission", "owner"
    ]
    numeric_features = ["year", "km_driven"]
    features = categorical_features + numeric_features
    target = "selling_price"

    for col in categorical_features:
        df[col] = df[col].astype("string").fillna("Unknown").astype(str)

    # Save cleaned data for the Streamlit app.
    df.to_csv(PROCESSED_FILE, index=False)

    X = df[features]
    y = df[target]

    if len(df) < 10:
        raise ValueError(
            "Not enough valid rows found. Please check car_data.csv."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1
                )
            )
        ]
    )

    print(f"Training with {len(df)} cleaned car listings...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Test rows: {len(X_test)}")
    print(f"Mean Absolute Error: Rs. {mae:,.0f}")

    if len(y_test) > 1:
        r2 = r2_score(y_test, predictions)
        print(f"R² score: {r2:.3f}")

    # Train the final model using all available cleaned rows.
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features
        },
        MODEL_FILE
    )

    print("\nTraining completed successfully!")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved cleaned dataset: {PROCESSED_FILE}")
    print(f"Unique brands: {df['brand'].nunique()}")
    print(f"Cleaned rows: {len(df)}")


if __name__ == "__main__":
    main()
