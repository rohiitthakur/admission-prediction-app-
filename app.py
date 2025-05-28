import streamlit as st
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("xgb_pipeline.pkl")

st.title("Lead Admission Probability Predictor (XGBoost)")
st.markdown("Fill in the lead details below:")

# --- Inputs (Update these to match your training features) ---
no_of_calls = st.number_input("Number of Calls", min_value=0)
answered_calls = st.number_input("Answered Calls", min_value=0)
age = st.number_input("Age", min_value=18, max_value=100)
primary_source_attempt = st.number_input("Primary Source Attempt", min_value=0)
secondary_source_attempt = st.number_input("Secondary Source Attempt", min_value=0)
tertiary_source_attempt = st.number_input("Tertiary Source Attempt", min_value=0)

app_completion_pct = st.slider("Application Completion %", 0.0, 100.0, 75.0)
no_of_calls = st.number_input("Number of Calls", min_value=0, value=5)
answered_calls = st.number_input("Answered Calls", min_value=0, value=3)
lead_score = st.number_input("Lead Score", min_value=0.0, value=60.0)
engagement_score = st.number_input("Engagement Score", min_value=0.0, value=50.0)
age = st.number_input("Age", min_value=15, max_value=60, value=22)
primary_source_attempt = st.number_input("Primary Source Attempt", min_value=0, value=1)
secondary_source_attempt = st.number_input("Secondary Source Attempt", min_value=0, value=1)
tertiary_source_attempt = st.number_input("Tertiary Source Attempt", min_value=0, value=1)

# --- Categorical Inputs ---
last_notable_activity = st.selectbox("Last Notable Activity", [
    "Call Disposition", "Dynamic Form Submission", "Email Bounced", "Email Link Clicked",
    "Email Opened", "Email Sent", "Facebook Lead Ads Submissions", "Form submitted on Portal",
    "Inbound Phone Call Activity", "Lead Capture", "Lead Enquiry", "Logged into Portal",
    "Logged out of Portal", "Mailing preference link clicked", "Modified",
    "Outbound Phone Call Activity", "Page Visited on Website", "Payment",
    "Re-registration Attempt Detected", "Resubscribed", "Smart Link Accessed",
    "Unsubscribed", "WhatsApp Message"
])

lead_source = st.selectbox("Lead Source", ['organic', 'Referral', 'Paid Leads'])

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

secondary_source = st.selectbox("Secondary Source", ['organic', 'Referral', 'Paid Leads', 'Channel', 'Unknown'])
present_area = st.selectbox("Present Area", ['Urban', 'Rural'])
notes_sentiments = st.selectbox("Notes Sentiments", ['positive', 'negative', 'neutral'])


# Button
if st.button("Predict"):
    input_df = pd.DataFrame([{
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
        "secondary source": secondary_source,
        "present area": present_area,
        "last notable activity": last_notable_activity,
        "notes_sentiments": notes_sentiments
    }])

    prob = pipeline.predict_proba(input_df)[0][1]
    st.success(f"Predicted admission probability: {prob:.2%}")
