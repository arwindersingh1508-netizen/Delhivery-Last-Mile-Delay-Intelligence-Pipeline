import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend",
    "dispatch_hour", "hub_tier", "hub_daily_capacity", "label_print_batch",
    "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no"
]

def clean_shipments(raw, medians=None):
    """Takes a raw shipments DataFrame and returns a tuple (clean_df, medians)."""
    df = raw.copy()

    # 1. Drop exact duplicate rows
    df = df.drop_duplicates()

    # 2. service_type: strip leading/trailing spaces and convert to title case
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. payment_mode: replace missing value with "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value: remove rupee symbol and commas, convert to numeric
    if "declared_value" in df.columns:
        df["declared_value_inr"] = (
            df["declared_value"]
            .astype(str)
            .str.replace("\u20b9", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["declared_value_inr"] = pd.to_numeric(df["declared_value_inr"], errors="coerce")
        df = df.drop(columns=["declared_value"])

    # 5. dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")

    # 6. parcel_weight_kg: values above 100 are scanner faults -> treat as missing
    if "parcel_weight_kg" in df.columns:
        df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. Fill missing values with medians
    cols_to_fill = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    computed_medians = {}

    if medians is None:
        for col in cols_to_fill:
            if col in df.columns:
                computed_medians[col] = float(df[col].median())
            else:
                computed_medians[col] = 0.0
    else:
        computed_medians = medians

    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(computed_medians[col])

    return df, computed_medians

def build_features(clean, hubs):
    """Returns a DataFrame with hub columns attached and new features engineered."""
    df = clean.copy()

    # Merge with hubs
    df = df.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left"
    )

    # Promised hours based on service type
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    # is_cod
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # is_weekend (1 when dispatch day is Saturday or Sunday)
    df["is_weekend"] = df["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)

    # dispatch_hour
    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    # log_declared_value
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
