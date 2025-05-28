import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
clf = joblib.load("RandomForest_model.pkl")  # Ensure this file exists in the same folder

# Streamlit App UI
st.title("🎓 Lead Admission Prediction App")
st.markdown("Fill in the lead details below to estimate the **probability of taking admission**.")

# --- Numeric Inputs ---
app_completion_pct = st.slider("Application Completion %", 0.0, 100.0, 75.0)
no_of_calls = st.number_input("Number of Calls", min_value=0, value=1)
answered_calls = st.number_input("Answered Calls", min_value=0, value=1)
lead_score = st.number_input("Lead Score", min_value=0.0, value=65.0)
engagement_score = st.number_input("Engagement Score", min_value=0.0, value=50.0)
age = st.number_input("Age", min_value=18, max_value=60, value=22)
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

# --- Predict Button ---
if st.button("Predict Admission Probability"):
    input_data = pd.DataFrame([{
        "application completion percentage": app_completion_pct,
        "no. of calls": no_of_calls,
        "answered calls": answered_calls,
        "lead score": lead_score,
        "engagement score": engagement_score,
        "age": age,
        "primary source attempt": primary_source_attempt,
        "secondary source attempt": secondary_source_attempt,
        "tertiary source attempt": tertiary_source_attempt,
        "last notable activity": last_notable_activity,
        "lead source": lead_source,
        "source medium": source_medium,
        "timing of lead": timing_of_lead,
        "gender": gender,
        "course interested": course_interested,
        "secondary source": secondary_source,
        "present area": present_area,
        "notes_sentiments": notes_sentiments
    }])

    try:
        prob = clf.predict_proba(input_data)[0][1]
        st.success(f"📊 Predicted probability of taking admission: **{prob:.2%}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    try:
        prob = clf.predict_proba(input_data)[0][1]
        st.success(f"Predicted probability of taking admission: {prob:.2%}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
