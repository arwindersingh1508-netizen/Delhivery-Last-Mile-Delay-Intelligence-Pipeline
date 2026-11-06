import pandas as pd 
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")

# 1. Hub profile
hub_profile = df.groupby('hub_id').agg({
    'shipment_id': 'count',
    'delivery_hours': 'mean',
    'is_delayed': 'mean',
    'distance_km': 'mean',
    'n_stops': 'mean'
}).reset_index()

hub_profile.columns = ['hub_id', 'shipments', 'mean_delivery_hours', 'delay_rate', 'mean_distance_km', 'mean_n_stops']
hub_profile = hub_profile.sort_values(by='hub_id').reset_index(drop=True)
hub_profile.to_csv("outputs/hub_profile.csv", index=False)

# 2. Scale features for K-Means
scaler = StandardScaler()
features_for_clustering = hub_profile[['mean_delivery_hours', 'delay_rate', 'mean_distance_km', 'mean_n_stops']]
scaled_data = scaler.fit_transform(features_for_clustering)

# 3. Separation evidence for K = 2, 3, 4, 5
separation_results = []
for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    counts = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
    cluster_sizes_str = "-".join(str(c) for c in counts)

    across_distances = []
    within_distances = []
    for i in range(len(scaled_data)):
        for j in range(i + 1, len(scaled_data)):
            dist = np.linalg.norm(scaled_data[i] - scaled_data[j])
            if labels[i] == labels[j]:
                within_distances.append(dist)
            else:
                across_distances.append(dist)

    mean_across = sum(across_distances) / len(across_distances) if across_distances else 0
    mean_within = sum(within_distances) / len(within_distances) if within_distances else 0
    ratio = mean_across / mean_within if mean_within > 0 else 0

    separation_results.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": cluster_sizes_str
    })

pd.DataFrame(separation_results).to_csv("outputs/part_d_separation.csv", index=False)

# 4. Final K=3 Segments
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
hub_profile['cluster'] = kmeans_3.fit_predict(scaled_data)

cluster_means = hub_profile.groupby('cluster')['delay_rate'].mean().sort_values()
sorted_cluster_indexes = cluster_means.index.tolist()

name_mapping = {
    sorted_cluster_indexes[0]: "Standard",
    sorted_cluster_indexes[1]: "Watch",
    sorted_cluster_indexes[2]: "Escalate"
}
hub_profile['segment'] = hub_profile['cluster'].map(name_mapping)

if os.path.exists("data/hubs.csv"):
    hubs_info = pd.read_csv("data/hubs.csv")
    hub_merged = pd.merge(hub_profile, hubs_info, on='hub_id', how='left')
else:
    hub_merged = hub_profile.copy()
    hub_merged['hub_tier'] = 1
    hub_merged['hub_city'] = "Unknown"

hub_segments = hub_merged[['hub_id', 'segment', 'hub_city', 'hub_tier']].sort_values(by='hub_id').reset_index(drop=True)
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

segment_profile = hub_merged.groupby('segment').agg({
    'hub_id': 'count',
    'mean_delivery_hours': 'mean',
    'delay_rate': 'mean',
    'mean_distance_km': 'mean',
    'mean_n_stops': 'mean'
}).reset_index()
segment_profile.columns = ['segment', 'hubs', 'mean_delivery_hours', 'delay_rate', 'mean_distance_km', 'mean_n_stops']
segment_profile.to_csv("outputs/segment_profile.csv", index=False)

tier_dict = {1: "Standard", 2: "Watch", 3: "Escalate"}
hub_merged['expected_segment'] = hub_merged['hub_tier'].map(tier_dict)
mismatches = hub_merged[hub_merged['segment'] != hub_merged['expected_segment']]
mismatches[['hub_id', 'hub_city', 'hub_tier', 'segment']].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

# 5. Package model
exclude_cols = ['shipment_id', 'hub_id', 'service_type', 'payment_mode', 'dispatch_time',
                'hub_city', 'region', 'delivery_hours', 'is_delayed', 'declared_value']
feature_cols = [col for col in df.columns if col not in exclude_cols][:17]

X = df[feature_cols]
y = df['is_delayed']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42, stratify=y_train_val)

my_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])
my_pipeline.fit(X_full_train, y_full_train)

joblib.dump(my_pipeline, "models/delay_model.pkl")
pd.DataFrame({"threshold": [0.5]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": feature_cols}).to_csv("models/feature_order.csv", index=False)
medians = X_full_train.median()
medians.to_frame("median").to_csv("models/impute_values.csv")

loaded_model = joblib.load("models/delay_model.pkl")
probs1 = my_pipeline.predict_proba(X_test)[:, 1]
probs2 = loaded_model.predict_proba(X_test)[:, 1]
max_diff = max(abs(probs1 - probs2))

reload_df = pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff"],
    "value": [len(X_test), max_diff]
})
reload_df.to_csv("outputs/part_d_reload_check.csv", index=False)

# 6. Scoring new shipments safely
if os.path.exists("data/new_shipments.csv"):
    new_shipments = pd.read_csv("data/new_shipments.csv")

    # Safely handle missing columns by filling with training medians
    for col in feature_cols:
        if col not in new_shipments.columns:
            new_shipments[col] = medians[col]
        else:
            new_shipments[col] = new_shipments[col].fillna(medians[col])

    X_new = new_shipments[feature_cols]
    new_probs = loaded_model.predict_proba(X_new)[:, 1]

    action_rows = []
    for index, row in new_shipments.iterrows():
        p = new_probs[index]
        flag = 1 if p >= 0.5 else 0
        h_id = row['hub_id'] if 'hub_id' in row else 'HUB01'

        match_seg = hub_segments[hub_segments['hub_id'] == h_id]
        seg_name = match_seg['segment'].values[0] if len(match_seg) > 0 else "Standard"

        if flag == 1 and seg_name == "Escalate":
            act = "Escalate to hub manager"
        elif flag == 1:
            act = "Proactive customer call"
        else:
            act = "No action"

        action_rows.append({
            "shipment_id": row['shipment_id'] if 'shipment_id' in row else index,
            "hub_id": h_id,
            "segment": seg_name,
            "delay_probability": round(float(p), 4),
            "flagged": flag,
            "action": act
        })

    pd.DataFrame(action_rows).to_csv("outputs/new_shipment_actions.csv", index=False)

print("done Part D !")
