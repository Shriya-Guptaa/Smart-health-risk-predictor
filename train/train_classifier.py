''' GOAL: Predict health risk'''

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import dump

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")


df.drop_duplicates(inplace=True)
df.dropna(subset=['bmi'], inplace=True)  


le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])           # Male=1, Female=0
df['ever_married'] = le.fit_transform(df['ever_married'])
df['work_type'] = le.fit_transform(df['work_type'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'].astype(str))

X = df.drop(columns=['id', 'stroke'])  # Exclude ID and target
y = df['stroke']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


dump(scaler, "models/scaler.joblib")


log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss') #xgboost is a libraray based on the algorithm of gradient boosting but optimized

log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train_scaled, y_train)


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("ROC AUC Score:", auc)
    return auc

auc_log = evaluate(log_model, X_test_scaled, y_test, "Logistic Regression")
auc_rf = evaluate(rf_model, X_test_scaled, y_test, "Random Forest")
auc_xgb = evaluate(xgb_model, X_test_scaled, y_test, "XGBoost")

#  x[0]---> stores model object, x[1]--> auc score
best_model = max(
    [(log_model, auc_log, "logistic"), (rf_model, auc_rf, "randomforest"), (xgb_model, auc_xgb, "xgboost")],
    key=lambda x: x[1]
)

dump(best_model[0], f"models/stroke_classifier_{best_model[2]}.joblib")
print(f"\nBest model saved: {best_model[2]} with AUC {best_model[1]:.4f}")
