import pandas as pd
from bmi_utils import calculate_bmi


work_type_map = {'Private': 2, 'Self-employed': 3, 'Govt_job': 0, 'children': 1, 'Never_worked': 4}
residence_map = {'Urban': 1, 'Rural': 0}
gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
married_map = {'Yes': 1, 'No': 0}
smoke_map = {'formerly smoked': 1, 'never smoked': 2, 'smokes': 3, 'Unknown': 0}

def preprocess_input_for_risk(user_input):
    
    df = pd.DataFrame([user_input.copy()])

    if "bmi" not in df or pd.isna(df["bmi"].iloc[0]):
        df["bmi"] = calculate_bmi(df["weight_kg"].iloc[0], df["height_cm"].iloc[0])

    df['gender'] = gender_map.get(df['gender'].iloc[0], 0)
    df['ever_married'] = married_map.get(df['ever_married'].iloc[0], 0)
    df['work_type'] = work_type_map.get(df['work_type'].iloc[0], 2)
    df['Residence_type'] = residence_map.get(df['Residence_type'].iloc[0], 1)
    df['smoking_status'] = smoke_map.get(df['smoking_status'].iloc[0], 0)

    df = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
             'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
    return df


def preprocess_input_for_tips(user_input):
   
    used_default_glucose = False
    required = ["age", "weight_kg", "height_cm", "hypertension", "heart_disease"]
    for key in required:
        if key not in user_input:
            raise ValueError(f"Missing required field: {key}")

    if "bmi" not in user_input or user_input["bmi"] is None:
        user_input["bmi"] = calculate_bmi(user_input["weight_kg"], user_input["height_cm"])

    if "avg_glucose_level" not in user_input or user_input["avg_glucose_level"] is None:
        user_input["avg_glucose_level"] = 99  # neutral value
        used_default_glucose = True

    return user_input, used_default_glucose


def preprocess_input_for_cluster(user_input):
   
    df_user = pd.DataFrame([user_input])
    df_encoded = pd.get_dummies(df_user, drop_first=True)

    ref_df = pd.read_csv("data/cluster_reference_columns.csv")
    for col in ref_df.columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[ref_df.columns]
    return df_encoded

