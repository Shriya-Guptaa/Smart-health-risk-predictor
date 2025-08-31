#recommend health tips
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from joblib import dump


df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
features = ['age', 'bmi', 'avg_glucose_level', 'hypertension', 'heart_disease']

df_knn = df[features].dropna()

df_knn.to_csv("data/user_profiles.csv", index=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_knn)

knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn_model.fit(X_scaled)

dump(knn_model, "models/knn_model.joblib")
dump(scaler, "models/knn_scaler.joblib")
print("KNN model and scaler saved.")




