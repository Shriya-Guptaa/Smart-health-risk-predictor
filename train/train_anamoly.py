''' Detects Anomalies --> used in healthtips_knn.py'''
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('data/healthcare-dataset-stroke-data.csv')

#features = ['age', 'bmi', 'avg_glucose_level', 'hypertension','heart_disease']
features = ['age', 'bmi', 'avg_glucose_level']

healthy_users = df[
    (df["hypertension"] == 0) &
    (df["heart_disease"] == 0) &
    (df["avg_glucose_level"] < 125) &
    (df["bmi"] >= 18.5) &
    (df["bmi"] <= 24.9)
]

df=df[features].dropna()
X = healthy_users[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
iso.fit(X_scaled)

dump(iso, "models/anomaly_detector.joblib")
dump(scaler, "models/anomaly_scaler.joblib")

print("saved")
