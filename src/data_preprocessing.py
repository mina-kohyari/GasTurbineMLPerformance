import pandas as pd
from pathlib import Path


def process_data(DATA_PATH):
    from pathlib import Path
    import pandas as pd

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "turbine_data.csv"
    CLEAN_PATH = BASE_DIR / "data" / "turbine_data_clean.csv"

    # Load data
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully")
    print("Original shape:", df.shape)

    # Strip column names
    df.columns = df.columns.str.strip()

    # Convert numeric columns
    for col in df.columns:
        if col.lower() not in ["hour_time", "time", "date"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    df.dropna(inplace=True)
    print("Cleaned shape:", df.shape)

    # Save cleaned dataset
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned dataset saved to: {CLEAN_PATH}")

    # Features & target
    TARGET = "Real power"
    DROP_COLS = ["Hour_Time"]

    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]

    return X, y