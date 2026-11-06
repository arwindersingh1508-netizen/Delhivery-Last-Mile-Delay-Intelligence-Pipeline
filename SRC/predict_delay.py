
import joblib
import pandas as pd

def predict_new_shipment(input_data_dict):
    model = joblib.load("outputs/best_model.pkl")
    input_df = pd.DataFrame([input_data_dict])
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    print("predict Done")
