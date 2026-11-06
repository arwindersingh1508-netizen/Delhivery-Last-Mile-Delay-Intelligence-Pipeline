import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Make sure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load data
df = pd.read_csv("data/modelling_table.csv")

# Select 17 features
exclude_cols = ['shipment_id', 'hub_id', 'service_type', 'payment_mode', 'dispatch_time',
                'hub_city', 'region', 'delivery_hours', 'is_delayed', 'declared_value']
feature_cols = [col for col in df.columns if col not in exclude_cols][:17]

X = df[feature_cols]
y = df['is_delayed']

# 1. Stratified Split (840 / 180 / 180)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.5, random_state=42, stratify=y_train_val
)

# Helper function to get model evaluation metrics
def get_metrics(y_true, y_pred, model_name, split="val"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "split": split,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

# 2. Blind Baseline and Three Classifiers
models_results = []

preds_blind = np.zeros_like(y_val)
models_results.append(get_metrics(y_val, preds_blind, "blind_all_on_time"))

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])
lr_pipe.fit(X_train, y_train)
models_results.append(get_metrics(y_val, lr_pipe.predict(X_val), "logistic_regression"))

best_f1 = 0
best_depth = 2
for d in [2, 3, 4, 5, 6, 8, 10]:
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    score = f1_score(y_val, dt.predict(X_val), zero_division=0)
    if score > best_f1:
        best_f1 = score
        best_depth = d

dt_best = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42)
dt_best.fit(X_train, y_train)
models_results.append(get_metrics(y_val, dt_best.predict(X_val), f"decision_tree_depth_{best_depth}"))

rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)
models_results.append(get_metrics(y_val, rf_default.predict(X_val), "random_forest_default"))

pd.DataFrame(models_results)[["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]].to_csv("outputs/part_c_models.csv", index=False)

# 3. Feature Scores & SelectKBest
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)

selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)

feat_df = pd.DataFrame({
    "feature": feature_cols,
    "f_score": selector_all.scores_,
    "rf_importance": rf_full.feature_importances_
})
feat_df["f_rank"] = feat_df["f_score"].rank(ascending=False, method="min").astype(int)
feat_df = feat_df.sort_values(by="f_rank").reset_index(drop=True)
feat_df["is_noise"] = feat_df["f_rank"] >= 8
feat_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_results = []
for k in [5, 7, 10, 12, 17]:
    pipe_k = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe_k.fit(X_train, y_train)
    score = f1_score(y_val, pipe_k.predict(X_val), zero_division=0)
    selectk_results.append({"k": k, "val_f1": score})
pd.DataFrame(selectk_results).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. Tuning with SelectKBest (k=7)
selector_7 = SelectKBest(f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [1, 5, 10]
}

rf_base = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5)
grid_search.fit(X_train_7, y_train)
grid_f1 = f1_score(y_val, grid_search.predict(X_val_7), zero_division=0)

random_search = RandomizedSearchCV(estimator=rf_base, param_distributions=param_grid, n_iter=9, cv=5, random_state=42)
random_search.fit(X_train_7, y_train)
random_f1 = f1_score(y_val, random_search.predict(X_val_7), zero_division=0)

rf_default_7 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default_7.fit(X_train_7, y_train)
default_f1 = f1_score(y_val, rf_default_7.predict(X_val_7), zero_division=0)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": len(grid_search.cv_results_['params']),
        "cv_folds": 5,
        "total_fits": len(grid_search.cv_results_['params']) * 5,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_f1
    },
    {
        "search": "randomized_search",
        "candidates": len(random_search.cv_results_['params']),
        "cv_folds": 5,
        "total_fits": len(random_search.cv_results_['params']) * 5,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 0,
        "best_params": "defaults",
        "best_cv_accuracy": "",
        "val_f1": default_f1
    }
]
pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)

best_row = pd.DataFrame(tuning_rows).sort_values(by="val_f1", ascending=False).iloc[0]
chosen_search = best_row["search"]

if chosen_search == "grid_search":
    best_model = grid_search.best_estimator_
    p = grid_search.best_params_
    n_est, max_d, min_leaf = p.get('n_estimators', 100), p.get('max_depth', None), p.get('min_samples_leaf', 1)
elif chosen_search == "randomized_search":
    best_model = random_search.best_estimator_
    p = random_search.best_params_
    n_est, max_d, min_leaf = p.get('n_estimators', 100), p.get('max_depth', None), p.get('min_samples_leaf', 1)
else:
    best_model = rf_default_7
    n_est, max_d, min_leaf = 100, "unlimited", 1

# 5. Alert Thresholds
final_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", best_model)
])
final_pipeline.fit(X_train, y_train)

val_probs = final_pipeline.predict_proba(X_val)[:, 1]
threshold_rows = []
valid_thresholds = []

for t in np.arange(0.1, 1.0, 0.1):
    preds = (val_probs >= t).astype(int)
    m = get_metrics(y_val, preds, f"thresh_{t:.1f}")
    threshold_rows.append({
        "threshold": round(t, 1),
        "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
        "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]
    })
    if m["recall"] >= 0.80:
        valid_thresholds.append(t)

pd.DataFrame(threshold_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

chosen_threshold = max(valid_thresholds) if valid_thresholds else 0.1

pd.DataFrame([{
    "selector": "SelectKBest(f_classif, k=7)",
    "n_estimators": n_est,
    "max_depth": max_d,
    "min_samples_leaf": min_leaf,
    "threshold": chosen_threshold,
    "chosen_from": chosen_search
}]).to_csv("outputs/part_c_final_config.csv", index=False)

# 6. Final Test Evaluation
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_pipeline.fit(X_full, y_full)
test_probs = final_pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= chosen_threshold).astype(int)

test_m = get_metrics(y_test, test_preds, "final_model", split="test")
pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": test_m["TN"], "FP": test_m["FP"], "FN": test_m["FN"], "TP": test_m["TP"],
    "accuracy": test_m["accuracy"], "precision": test_m["precision"], "recall": test_m["recall"], "f1": test_m["f1"]
}]).to_csv("outputs/part_c_final_test.csv", index=False)

print("done Part c!")
