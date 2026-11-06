# Delhivery-Last-Mile-Delay-Intelligence-Pipeline
An end-to-end Python Machine Learning project for predicting last-mile delivery times, identifying delayed shipments, and segmenting logistics hubs using synthetic shipment data.

# Delhivery Last-Mile Delay Intelligence Pipeline

Hey there! This is a machine learning project that predicts shipping delivery times, catches delayed packages, and groups different delivery hubs based on their performance.

---

## How to Run

If you want to run this project from a fresh download, open your terminal and type these commands in this exact order:

```bash
git clone [https://github.com/arwindersh1508-netizen/Delhivery-Last-Mile-Delay-Intelligence-Pipeline.git](https://github.com/arwindersh1508-netizen/Delhivery-Last-Mile-Delay-Intelligence-Pipeline.git)
cd Delhivery-Last-Mile-Delay-Intelligence-Pipeline
pip install -r requirements.txt
python generate_data.py
python shipment_prep.py
python src/part_a_clean_eda.py
python src/part_b_regression.py
python src/part_c_classification.py
python src/part_d_segments.py
python src/package_model.py
python src/predict_delay.py

Note on Model File: The model file (outputs/best_model.pkl) is created automatically when you run package_model.py. If python library versions are different on the grader's computer, it will just re-run that script to make sure everything works smoothly.

#Data Findings
(i) Which service gets delayed most? Economy shipments get delayed the most often, with a delay rate of 42.50% (compared to Standard at 24.10% and Express at 8.30%).

(ii) How hub tiers affect delays: Delays get worse as you move from big cities to remote areas—going from 15.20% in Tier 1 hubs up to 24.80% in Tier 2, and peaking at 41.50% in Tier 3 remote hubs.

(iii) How distance affects delays: Longer trips mean way more delays—going from 12.10% for short trips (<200 km) to 28.40% for medium trips (200–500 km), and 52.70% for long trips (>500 km).

(iv) Top things related to delivery time: Excluding delivery time itself, the three things most closely tied to how long a delivery takes (from part_a_correlations.csv) are:

distance_km: 0.8452

n_stops: 0.6214

sale_event_flag: 0.4120

(v) Why declared value was log-transformed: We transformed declared_value_inr using a log because the money amounts were super lopsided (the average was ₹12,543.50 while the middle value/median was only ₹1,120.00). Doing this makes the data much more normal and stable for the model (new mean: 6.95, median: 7.02).

#Model Results (Regression)
Baseline model: The simple linear regression baseline got an R-squared score of 0.6824, meaning it explains about 68.24% of what causes delivery time changes.

The winning model: The Random Forest Regressor won the test and got the best score with a Root Mean Squared Error (RMSE) of 3.4215 hours.

Tree depth (Bias vs Variance): Shallow trees (like depth 2 to 4) are too simple and miss things (high bias), while super deep trees (depth 10+) memorize the training practice test too much and mess up on new data (high variance).

Learning rates: A learning rate of 0.1 was the sweet spot. If it's too low (like 0.01), it takes forever and doesn't finish learning. If it's too high (like 0.5), it jumps past the right answer and gets worse.

If cross-validation and single split disagree: We trust the cross-validation model because it tests across multiple data splits instead of just guessing based on one lucky or unlucky split.

#Model Results (Classification). The blind baseline: Just blindly guessing that nothing is delayed gives an accuracy of 75.50%, but it is totally useless because it catches 0% of actual delays.Fake/Noise columns: The 5 fake random columns we added only made up 1.8% of the random forest's importance, and all of them correctly stayed out of the top 7 features.The $k$ table: Testing different numbers of features showed that keeping $k=7$ gave the best and most stable results before things started getting messy with extra noise.Tuning vs. Untuned: Tuning the hyper-parameters didn't really beat the default random forest here, showing that trying to tune on a tiny set of 180 validation rows mostly just chases random noise.The threshold choice: We set the threshold at 0.50 to get a fair balance between not giving too many false alarms and actually catching real delays.Test metrics & real talk on recall: On the test set, we got a precision of 0.7805, recall of 0.7073, and F1 of 0.7421. The test set only has 41 delayed parcels, so even if just one parcel is misclassified, it swings the recall rate by over 2.4%.

#Production Plan
(a) How we made it production-ready
Reproducibility: We used fixed random seeds (random_state=42) so it runs the exact same way every time.

Consistency: Cleaning and prepping steps are stored in shared files so training and real-world predictions use the exact same logic.

Accuracy: Checked against baseline models using proper scores (RMSE and F1).

Speed: Runs super fast in batch mode using lightweight files, taking less than 2 seconds.

Monitoring: Results and flags are cleanly saved as CSV files in the outputs/ folder.

(b) What we should keep an eye on (Metrics & Triggers)
Weekly delay share: Check the share of delayed parcels; Trigger: If it jumps more than 15% week-over-week.

Weekly flagged share: Check how many total parcels get flagged; Trigger: If more than 20% get flagged.

Realized recall: Check recall on parcels whose final status we now know; Trigger: If it drops below 65%.

(c) Retraining & Data Drift
Retraining schedule: Retrain the model every two weeks, or right after major spikes.

Handling the September Sale Window: The big sale event (sale_event_flag from Sep 15–28) has crazy high volumes, so we cannot compare it to normal weeks. We should compare it against past sales instead.

Data drift: This means things changing over time—like longer travel distances, weird rainy seasons messing up travel times, or sudden jumps in cash-on-delivery (COD) orders that make the model less accurate.

Limitations
Fake data: The project uses synthetic (simulated) data, so real-world weird edge cases might look a bit different.

Small test group: The test set only has 41 delayed parcels, meaning a single mistake can swing the metrics a lot.

Single split threshold: The decision threshold was tuned on just one validation split, which might behave slightly differently on completely unseen data.

