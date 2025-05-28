import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
clf = joblib.load("clf_model.pkl")  # Ensure this file is present in the same directory

# Title
st.title("Lead Admission Prediction App")
st.markdown("Enter the lead details below to predict the admission probability.")

# Input form
last_notable_activity = st.selectbox("Last Notable Activity", [
    "Call Disposition", "Dynamic Form Submission", "Email Bounced", "Email Link Clicked",
    "Email Opened", "Email Sent", "Facebook Lead Ads Submissions", "Form submitted on Portal",
    "Inbound Phone Call Activity", "Lead Capture", "Lead Enquiry", "Logged into Portal",
    "Logged out of Portal", "Mailing preference link clicked", "Modified",
    "Outbound Phone Call Activity", "Page Visited on Website", "Payment",
    "Re-registration Attempt Detected", "Resubscribed", "Smart Link Accessed",
    "Unsubscribed", "WhatsApp Message"
])

no_of_calls = st.number_input("Number of Calls", min_value=0)
answered_calls = st.number_input("Answered Calls", min_value=0)
age = st.number_input("Age", min_value=18)
primary_source_attempt = st.number_input("Primary Source Attempt", min_value=0)
secondary_source_attempt = st.number_input("Secondary Source Attempt", min_value=0)
tertiary_source_attempt = st.number_input("Tertiary Source Attempt", min_value=0)

lead_source = st.selectbox("Lead Source", ['organic', 'Referral', 'Paid Leads'])

# ✅ New Source Medium values
source_medium = st.selectbox("Source Medium", [
    "admissions", "Buddy Referral", "ChatBot/Whatsapp", "DS",
    "Google_Demand_Gen", "Google_Per_Max", "Google_Search", "Incoming Call",
    "lookalike/ MBA Remarketing Ads", "namita_gandhotra", "Personal Referrals",
    "Shoolini Offline", "Shoolini Referral"
])

timing_of_lead = st.selectbox("Timing of Lead", ['Early in Cycle', 'Mid in Cycle', 'Late in Cycle'])
gender = st.selectbox("Gender", ['Male', 'Female'])
course_interested = st.selectbox("Course Interested", [
    'MBA Online', 'BBA Online', 'BCA Online', 'MCA Online', 
    'MA ( ENGLISH LITERATURE ) Online', 'BCOM(Hons) Online'
])

secondary_source = st.selectbox("Secondary Source", [
    'organic', 'Referral', 'Paid Leads', 'Channel', 'Unknown'
])

present_area = st.selectbox("Present Area", ['Urban', 'Rural'])


# Make prediction
if st.button("Predict Admission Probability"):
    input_data = pd.DataFrame([{
        "last notable activity": last_notable_activity,
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
