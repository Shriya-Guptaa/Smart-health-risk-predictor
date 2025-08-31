# Smart Health Risk Predictor

## 📌 Overview
The **Smart Health Risk Predictor** is a Streamlit-based web application that predicts the **risk of stroke** based on user health information, recommends **personalized health tips**, and segments users into **health profile clusters**.  

I developed this project as a project at my internship, which gave me hands-on experience in applying **machine learning algorithms** to real-world healthcare data and building an interactive web app for end users. It was a great learning opportunity to understand how ML models work in practice and how to integrate them into a user-friendly application.

---

## 🛠 Features
- Predict stroke risk (Low, Moderate, High) using Logistic Regression, Random Forest, and XGBoost
- Recommend personalized health tips using a KNN model
- Assign users to health profile clusters using KMeans
- Calculate BMI based on user input
- Interactive web interface via Streamlit

---
## Technologies Used
- **Machine Learning:** Logistic Regression, Random Forest, XGBoost, KNN, KMeans  
- **Frontend:** Streamlit (Python-based UI framework)  
- **Backend & Data Processing:** Python, Pandas  
- **Database:** CSV-based storage (expandable to SQL-based solutions)  
---

## Project Structure

        Smart-Health-Risk-Predictor/
        ├── main_app.py
        ├── app/ # Core app logic
        │ ├── __init__.py
        │ ├── bmi_utils.py # BMI calculation
        │ ├── cluster_users.py # KMeans clustering
        │ ├── healthtips_knn.py # KNN for health tips
        │ ├── predict_risk.py # Stroke risk prediction
        │ └── logic.py # Data preprocessing
        ├── data/ # Data folder
        │ ├── healthcare-dataset-stroke-data.csv # Kaggle Dataset
        │ ├── cluster_names.json # Clustering info
        │ └── recommendations.json # Health tips recommendations
        ├── models/ # Trained ML models
        │ ├── anomaly_detector.joblib
        │ ├── anomaly_scaler.joblib
        │ ├── cluster_scaler.joblib
        │ ├── kmeans_clusters_k8.joblib
        │ ├── knn_model.joblib
        │ ├── knn_scaler.joblib
        │ ├── scaler.joblib
        │ └── stroke_classifier_logistic.joblib
        ├── train/ # Training scripts
        │ ├── train_anomaly.py
        │ ├── train_knn.py
        │ ├── train_kmeans.py
        │ └── train_classifier.py
        |── .gitignore
        ├── README.md
        └── requirements.txt
--- 

## ⚙️ Installation
### Prerequisites
- Python 3.7 or higher  
- Pip (Python package manager)

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/Smart-Health-Risk-Predictor.git
cd Smart-Health-Risk-Predictor

```
2.**Create a Virtual Environment (optional but recommended):**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate
```
3.**Install dependencies**
```bash
pip install -r requirements.txt
```
4.**Run the Application:**
```bash
streamlit run main_app.py
```
5.**Access the Web App:**
Open your browser and visit http://localhost:8501 to start using the Smart Health Risk Predictor.

---

## Usage
- **Enter Health Data:** Fill in your health information including age, weight, height, glucose level, hypertension, heart disease, and lifestyle habits.

- **Analyze Risk:** Click Analyze My Health to view your stroke risk prediction (Low, Moderate, High).

- **Get Recommendations:** Receive personalized health tips based on your profile using KNN.

- **View Clustering:** See which health profile cluster you belong to using KMeans.

- **Monitor BMI:** Check your calculated Body Mass Index (BMI) for additional insights.

---

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License 
