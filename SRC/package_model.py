
import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# 1. Load data
df = pd.read_csv("data/modelling_table.csv")
feature_cols = [c for c in df.columns if c not in ['delivery_hours', 'shipment_id']]
X = df[feature_cols]
y = df['delivery_hours']

# 2. Train final model on all data
final_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
final_model.fit(X, y)

# 3. Save the model 
model_path = "outputs/best_model.pkl"
joblib.dump(final_model, model_path)
print(f"Model packaged Done {model_path}!")
