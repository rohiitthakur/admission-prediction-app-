import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
clf = joblib.load("clf_model.pkl")  # Make sure this file is present in the same directory

# Title
st.title("Lead Admission Prediction App")
st.markdown("Enter the lead details below to predict the admission probability.")

# Input form
activity_score = st.number_input("Activity Score", min_value=0.0)
no_of_calls = st.number_input("Number of Calls", min_value=0.0)
answered_calls = st.number_input("Answered Calls", min_value=0.0)
age = st.number_input("Age", min_value=0.0)
primary_source_attempt = st.number_input("Primary Source Attempt", min_value=0)
secondary_source_attempt = st.number_input("Secondary Source Attempt", min_value=0)
tertiary_source_attempt = st.number_input("Tertiary Source Attempt", min_value=0)

lead_source = st.selectbox("Lead Source", ['organic', 'Referral', 'Paid Leads'])
source_medium = st.selectbox("Source Medium", ["admissions", "bba", "Brand-Video-Ad", "Corporate", "cpc", "direct", "firstvite", "Google_Demand_Gen", "Google_Per_Max", "Google_Search", "Herbalife", "iim", "incoming call", "inlinesolutions", "Interest", "JyotikaRajatYoutube", "kpmg", "lookalike Audience", "lookalike1", "lookalike2", "lookalike3", "lookalike3 – Copy", "lookalike4", "MBA Remarketing Ads", "mindstory", "namita_gandotra", "npf", "online degree", "paid", "Panku-Kumar-Video-Ad", "Performance_MAx", "personal refferal", "premchand", "reffral", "student Rederral", "Student Reference", "Student Referral", "student referrel", "study", "Super lookalike", "surender vats", "surender_vats", "WHATTS", "whatts app", "yes help me"])
timing_of_lead = st.selectbox("Timing of Lead", ['Early in Cycle', 'Mid in Cycle', 'Late in Cycle'])
gender = st.selectbox("Gender", ['Male', 'Female', 'Unknown'])
course_interested = st.selectbox("Course Interested", ['MBA Online', 'BBA Online', 'BCA Online', 'MCA Online'])
payment_options = st.selectbox("Payment Options", ['Pay After Placement', 'Opt-Out of Pay After Placement', 'Direct Selling', 'Unknown'])
secondary_source = st.selectbox("Secondary Source", ['organic', 'Referral', 'Paid Leads', 'Channel'])
tertiary_source = st.selectbox("Tertiary Source", ['organic', 'Referral', 'Paid Leads', 'Channel', 'Inbound Phone Call'])
present_area = st.selectbox("Present Area", ['Urban', 'Rural'])

# Make prediction
if st.button("Predict Admission Probability"):
    input_data = pd.DataFrame([{
        "activity score": activity_score,
        "no. of calls": no_of_calls,
        "answered calls": answered_calls,
        "age": age,
        "primary source attempt": primary_source_attempt,
        "secondary source attempt": secondary_source_attempt,
        "tertiary source attempt": tertiary_source_attempt,
        "lead source": lead_source,
        "source medium": source_medium,
        "timing of lead": timing_of_lead,
        "gender": gender,
        "course interested": course_interested,
        "payment options": payment_options,
        "secondary source": secondary_source,
        "tertiary source": tertiary_source,
        "present area": present_area
    }])

    st.write("Model input preview:", input_data)

    try:
        prob = clf.predict_proba(input_data)[0][1]
        st.success(f"Predicted probability of taking admission: {prob:.2%}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
