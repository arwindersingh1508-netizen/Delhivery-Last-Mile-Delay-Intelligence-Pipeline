import pandas as pd
import numpy as np

from src.shipment_prep import clean_shipments, build_features, PROMISED_HOURS, FEATURES

# 1. Read data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# 2. Clean shipments
clean_df, computed_medians = clean_shipments(raw)

# Save impute values
impute_df = pd.DataFrame(list(computed_medians.items()), columns=["metric", "value"])
impute_df.to_csv("models/impute_values.csv", index=False)

# 3. Build features
model_df = build_features(clean_df, hubs)

# 4. Target: is_delayed (1 when delivery_hours > promised_hours, else 0)
model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

# Save modelling table
model_df.to_csv("data/modelling_table.csv", index=False)

# 5. Checks file
checks_data = [
    {"metric": "total_shipments_modeled", "value": len(model_df)}
]
checks_df = pd.DataFrame(checks_data, columns=["metric", "value"])
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# 6. Delay evidence table
bins = [0, 200, 400, 600, float("inf")]
labels = ["0-200", "200-400", "400-600", "600+"]
model_df["distance_band"] = pd.cut(model_df["distance_km"], bins=bins, labels=labels, right=False)

delay_rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    grouped = model_df.groupby(dim, observed=False).agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()
    for _, row in grouped.iterrows():
        delay_rows.append({
            "dimension": dim,
            "group": str(row[dim]),
            "shipments": row["shipments"],
            "delayed": row["delayed"],
            "delay_rate": round(row["delayed"] / row["shipments"], 4),
            "mean_hours": round(row["mean_hours"], 4)
        })

delay_group = pd.DataFrame(delay_rows)
delay_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# 7. Correlations
numeric_cols = model_df.select_dtypes(include=[np.number]).columns
corr_matrix = model_df[numeric_cols].corr()
corr_matrix.to_csv("outputs/part_a_correlations.csv")
#os.makedirs("data", exist_ok=True); open("data/modelling_table.csv", "w").close()
print("ok")
