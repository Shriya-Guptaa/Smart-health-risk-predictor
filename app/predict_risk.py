import pandas as pd
from .bmi_utils import calculate_bmi
from joblib import load

model_log=load("models/stroke_classifier_logistic.joblib")
scaler=load("models/scaler.joblib")

work_type_map = {'Private': 2, 'Self-employed': 3, 'Govt_job': 0, 'children': 1, 'Never_worked': 4}
residence_map = {'Urban': 1, 'Rural': 0}
gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
married_map = {'Yes': 1, 'No': 0}
smoke_map = {'formerly smoked': 1, 'never smoked': 2, 'smokes': 3, 'Unknown': 0}

def preprocess_input(user_input):
    df = pd.DataFrame([user_input.copy()])
    if "bmi" not in df or pd.isna(df["bmi"].iloc[0]):
        df["bmi"] = calculate_bmi(df["weight_kg"].iloc[0], df["height_cm"].iloc[0])

    df['gender'] = gender_map.get(df['gender'].iloc[0], 0)
    df['ever_married'] = married_map.get(df['ever_married'].iloc[0], 0)
    df['work_type'] = work_type_map.get(df['work_type'].iloc[0], 2)
    df['Residence_type'] = residence_map.get(df['Residence_type'].iloc[0], 1)
    df['smoking_status'] = smoke_map.get(df['smoking_status'].iloc[0], 0)

    df = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]

    return df

def predict_health_risk(user_input,threshold=0.35):
    df_processed = preprocess_input(user_input)
    df_scaled = scaler.transform(df_processed)
    prob = model_log.predict_proba(df_scaled)[0][1]
  
    if prob >= 0.5:
        risk_level = "High"
    elif 0.15<=prob <0.5:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return risk_level, prob

if __name__ == "__main__":
    user = {
        "gender": "Male",
        "age": 67,
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Self-employed",
        "Residence_type": "Urban",
        "avg_glucose_level": 170,
        "smoking_status": "smokes",
        "weight_kg": 72,
        "height_cm": 172
    }

    pred, prob = predict_health_risk(user)
    print("Predicted Risk:",pred)

    print(f" Probability of Stroke: {prob:.2%}")
