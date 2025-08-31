''' Patient segmentation (create clusters)'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import dump
#from sklearn.metrics import silhouette_score

df=pd.read_csv('data/healthcare-dataset-stroke-data.csv')

features = ['Residence_type', 'work_type', 'avg_glucose_level', 'hypertension']
df_cluster=df[features].dropna()

df_encoded=pd.get_dummies(df_cluster,drop_first=True)

pd.DataFrame(columns=df_encoded.columns).to_csv("data/cluster_reference_columns.csv", index=False)

scaler=StandardScaler()
X_scaled=scaler.fit_transform(df_encoded)

kmeans=KMeans(n_clusters=8,random_state=42)
labels=kmeans.fit_predict(X_scaled)

df_cluster['Cluster']=labels #add cluster labels to dataframe

#score=silhouette_score(X_scaled,labels)
#print(f"Silhoutte Score(k-8):{score:.4f}")

dump(kmeans,"models/kmeans_clusters_k8.joblib")
dump(scaler,"models/cluster_scaler.joblib")


#print(df_cluster.groupby("Cluster")[['avg_glucose_level', 'hypertension']].mean())
