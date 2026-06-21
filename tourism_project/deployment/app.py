import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="srbhavan/tourism-package-prediction", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package prediction App")
st.write("""
This application predicts the likelihood of a customer buying a tourism package based on the input parameters.
Please enter the details below to get a prediction.
""")

# User input
age = st.number_input("Enter customer's age", min_value=0, max_value=100)
contact_type = st.selectbox("Type Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", ["1", "2", "3"])
pitch_duration = st.number_input("Enter duratiuon of pitch", min_value=0, max_value=100)
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Number of persons visiting", min_value=0, max_value=10)
followup_count = st.number_input("Number of follow-ups", min_value=0, max_value=100)
product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxes", "Super Deluxes", "King"])
preferred_property_star = st.number_input("Preferred property star", min_value=1, max_value=5)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
trip_count = st.number_input("Number of trips", min_value=0, max_value=25)
passport = st.selectbox("Passport?", ["0", "1"])
pitch_satisfied = st.selectbox("Pitch satisfied?", ["1", "2", "3", "4", "5"])
owned_car = st.selectbox("Owns car?", ["0", "1"])
children_visiting_count = st.number_input("Number of children visiting", min_value=0, max_value=10)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': contact_type,
    'CityTier': city_tier,
    'DurationOfPitch': pitch_duration,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': age,
    'NumberOfFollowups': followup_count,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': trip_count,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfied,
    'OwnCar': owned_car,
    'NumberOfChildrenVisiting': children_visiting_count,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])


if st.button("Predict the possibility of buying the package"):
    prediction = model.predict(input_data)[0]
    result = "Customer is likely to buy package" if prediction == 1 else "Customer is not likely to buy package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
