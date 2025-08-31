import pandas as pd
from joblib import load
import json

with open('data/cluster_names.json','r') as f:
    cluster_map=json.load(f)

kmeans_model = load("models/kmeans_clusters_k8.joblib")
scaler = load("models/cluster_scaler.joblib")


cluster_features = ['Residence_type', 'work_type', 'avg_glucose_level', 'hypertension']

def prepare_user_input(user_input):
    df_user = pd.DataFrame([user_input])

    df_encoded = pd.get_dummies(df_user, drop_first=True)

    ref_df = pd.read_csv("data/cluster_reference_columns.csv")  
    for col in ref_df.columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  

    df_encoded = df_encoded[ref_df.columns]  # keep order
    return df_encoded

def assign_cluster(user_input):

    df_encoded = prepare_user_input(user_input)
    scaled_input = scaler.transform(df_encoded)
    cluster = int(kmeans_model.predict(scaled_input)[0])
    return cluster


if __name__ == "__main__":
    user = {
        "Residence_type": "Urban",
        "work_type": "Private",
        "avg_glucose_level": 110,
        "hypertension": 1
    }

    cluster_id = assign_cluster(user)
    print(f"Assigned to Cluster: {cluster_id} -->{cluster_map[str(cluster_id)]}")
