import pandas as pd
import json
from joblib import load
from .bmi_utils import calculate_bmi

df_knn = pd.read_csv("data/user_profiles.csv")


knn_model=load("models/knn_model.joblib")
scaler=load("models/knn_scaler.joblib")

anomaly_model = load("models/anomaly_detector.joblib")
anomaly_scaler = load("models/anomaly_scaler.joblib")

with open("data/recommendations.json","r") as f:
    tips_json=json.load(f)


def recommend_json(user,used_default_glucose=False,is_anomaly=False):
    tips=[]

    if user["avg_glucose_level"] >= 180:
        tips.append("Glucose level critically HIGH. Medical attention is advised.")
    elif 125 < user["avg_glucose_level"] < 180:
        tips.append(tips_json["glucose_high"])
    elif 100 < user["avg_glucose_level"] <= 125:
        tips.append(tips_json["glucose_prediabetes"])
    elif user["avg_glucose_level"] < 70:
        tips.append("Glucose level critically LOW. Medical attention is advised.")


    
    if user['bmi']> 29:
        tips.append("BMI is critically HIGH. Medical attention is advised.")
    elif 25<=user["bmi"] <29:
        tips.append(tips_json["bmi_high"])
    elif user["bmi"] < 18.5:
        tips.append(tips_json["bmi_low"])
    elif user['bmi']<16:
        tips.append("BMI is critically LOW.Medical attention is advised.")
     
    if user["hypertension"] == 1:     
        tips.append(tips_json["has_hypertension"])
    if user["heart_disease"] == 1:
        tips.append(tips_json["has_heart_disease"])
    
    if used_default_glucose:
        tips.append("You have not entered glucose level.A neutral value was used.For accurate tips, consult doctor.")

    if is_anomaly:
        tips.append(" Your health profile is unusual. Please consult a specialist for deeper insights.")

    if not tips:
        tips.append(tips_json["default"])
    return tips


    
def recommend_health_tips(user_input):
    used_default_glucose = False

    required = ["age", "weight_kg", "height_cm", "hypertension", "heart_disease"]
    for key in required:
        if key not in user_input:
            raise ValueError(f"Missing required field: {key}")
        if "bmi" not in user_input:
            user_input["bmi"] = calculate_bmi(user_input["weight_kg"], user_input["height_cm"])

        if "avg_glucose_level" not in user_input or user_input["avg_glucose_level"] is None:
            user_input["avg_glucose_level"] = 99  # neutral value
            used_default_glucose = True
    
    knn_features = ["age", "bmi", "avg_glucose_level", "hypertension", "heart_disease"]
    user_df = pd.DataFrame([{key: user_input[key] for key in knn_features}])
    user_scaled = scaler.transform(user_df)

    distances, indices = knn_model.kneighbors(user_scaled) #find similar patients/users
    anomaly_features = ["age", "bmi", "avg_glucose_level"]
    anomaly_df = user_df[anomaly_features]
    anomaly_scaled = anomaly_scaler.transform(anomaly_df)
    is_anomaly = anomaly_model.predict(anomaly_scaled)[0] == -1

    tips = recommend_json(user_input,used_default_glucose,is_anomaly)
    return tips


if __name__ == "__main__":
    user = {
        "age": 67,
        "weight_kg": 87,
        "height_cm": 180,
        "avg_glucose_level": 220,
        "hypertension": 1,
        "heart_disease": 1
    }

    tips = recommend_health_tips(user)
    print("BMI:", calculate_bmi(user["weight_kg"], user["height_cm"]))
    print("Recommended Health Tips:")
    for tip in tips:
        print("•", tip)



