import streamlit as st
import json
from app.predict_risk import predict_health_risk
from app.healthtips_knn import recommend_health_tips 
from app.cluster_users import assign_cluster
from app.bmi_utils import calculate_bmi

with open("data/cluster_names.json", "r") as f:
    cluster_map = json.load(f)


st.set_page_config(page_title="Smart Health Risk Predictor")
st.title("Smart Health Risk Predictor")
st.markdown("""
This app predicts your stroke risk, recommends health tips, and segments your health profile.
Please enter your health information below:
""")
# Input Form
with st.form("health_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 30)
        weight = st.number_input("Weight (kg)", 1.0, 200.0, 60.0)
        height = st.number_input("Height (cm)", 50.0, 250.0, 170.0)
        glucose = st.number_input("Average Glucose Level", 1.0, 500.0, 99.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    ever_married = st.selectbox("Have you ever been married?", ["Yes", "No"])
    submitted = st.form_submit_button("Analyze My Health")

if submitted:
    user = {
        "age": age,
        "gender": gender,
        "weight_kg": weight,
        "height_cm": height,
        "avg_glucose_level": glucose,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "smoking_status": smoking_status
    }

    bmi = calculate_bmi(weight, height)
    user["bmi"] = bmi

    #Stroke Risk
    
    st.markdown("## Stroke Risk Prediction")
    risk_level, probability = predict_health_risk(user)
    st.write(f"**Predicted Risk:** {risk_level}")
    st.write(f"**Probability of Stroke:** {probability:.2%}")

    if risk_level == "High":
        st.error("High risk of stroke! Please consult a doctor.")
    elif risk_level == "Moderate":
        st.warning("Moderate risk. Take preventive measures.")
    else:
        st.success("Low risk. Maintain a healthy lifestyle.")


    st.subheader("Personalized Health Tips")
    tips = recommend_health_tips(user)
    for tip in tips:
        st.success(tip)

    st.subheader(" Health Profile Cluster")
    cluster_id = assign_cluster(user)
    cluster_description = cluster_map.get(str(cluster_id), "Unknown cluster")
    st.markdown(f"You are assigned to **Cluster {cluster_id}-->{cluster_description}")

    st.info(f"Your calculated BMI is **{bmi:.2f}**")