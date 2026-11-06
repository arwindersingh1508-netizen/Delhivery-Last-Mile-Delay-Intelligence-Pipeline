from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

# Load data and select the 17 features
df = pd.read_csv("data/modelling_table.csv")
exclude_cols = {'shipment_id', 'hub_id', 'service_type', 'payment_mode', 'dispatch_time',
                'hub_city', 'region', 'delivery_hours', 'is_delayed', 'declared_value'}
feature_cols = [col for col in df.columns if col not in exclude_cols][:17]

X = df[feature_cols]
y = df['delivery_hours']

# Task 1: Train, Validation, Test Split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)

# Task 2: Scaling & Checks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

checks_df = pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [
        len(X_train), len(X_val), len(X_test),
        np.max(np.abs(np.mean(X_train_scaled, axis=0))),
        np.max(np.abs(np.mean(X_test_scaled, axis=0)))
    ]
})
checks_df.to_csv("outputs/part_b_checks.csv", index=False)

# Task 3: Baseline
val_baseline_pred = np.full_like(y_val, y_train.mean())
baseline_metrics = {
    "model": "mean_baseline", "split": "val",
    "MAE": mean_absolute_error(y_val, val_baseline_pred),
    "MSE": mean_squared_error(y_val, val_baseline_pred),
    "RMSE": np.sqrt(mean_squared_error(y_val, val_baseline_pred)),
    "R2": r2_score(y_val, val_baseline_pred)
}

# Task 4: Tree-Depth Diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_records = []
best_val_rmse, best_depth = float('inf'), None

for d in depths:
    dt = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    t_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
    v_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))

    label = str(d) if d is not None else "unlimited"
    tree_records.append({"max_depth": label, "train_rmse": t_rmse, "val_rmse": v_rmse})

    if v_rmse < best_val_rmse:
        best_val_rmse = v_rmse
        best_depth = d

pd.DataFrame(tree_records).to_csv("outputs/part_b_tree.csv", index=False)

# Task 5: Four Candidate Models
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

metrics_rows = [baseline_metrics]
val_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    val_scores[name] = rmse

    metrics_rows.append({
        "model": name, "split": "val",
        "MAE": mean_absolute_error(y_val, preds),
        "MSE": mean_squared_error(y_val, preds),
        "RMSE": rmse,
        "R2": r2_score(y_val, preds)
    })

# Task 6: Learning Rate Experiment
gb_lr_records = []
for lr in [0.01, 0.1, 1.0]:
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gbr.fit(X_train, y_train)
    preds = gbr.predict(X_val)
    gb_lr_records.append({
        "learning_rate": lr,
        "val_rmse": np.sqrt(mean_squared_error(y_val, preds)),
        "val_r2": r2_score(y_val, preds)
    })
pd.DataFrame(gb_lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# Task 7: Cross-Validation & Final Model Test
top_two = sorted(val_scores, key=val_scores.get)[:2]
winning_model = top_two[0]

X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for name in top_two:
    scores = cross_val_score(models[name], X_full, y_full, cv=kf, scoring='r2')
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

# Test evaluation for winning model
final_model = models[winning_model]
final_model.fit(X_full, y_full)
test_preds = final_model.predict(X_test)

metrics_rows.append({
    "model": winning_model, "split": "test",
    "MAE": mean_absolute_error(y_test, test_preds),
    "MSE": mean_squared_error(y_test, test_preds),
    "RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
    "R2": r2_score(y_test, test_preds)
})

pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
print("Done Part B!")
