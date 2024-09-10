import streamlit as st
import pickle
import numpy as np
import joblib

# Load your trained model
# with open("sales_prediction_model.pkl", "rb") as file:
#     model = pickle.load(file)
model = joblib.load('Adsales.pkl')

# Title and description
st.title("Ad Sales Prediction")
st.write("Predict sales based on ad spend and influencer types.")

# Input fields
tv = st.number_input("TV", min_value=0.0, value=0.0)
radio = st.number_input("Radio", min_value=0.0, value=0.0)
social_media = st.number_input("Social Media", min_value=0.0, value=0.0)
influencer_macro = st.checkbox("Influencer Macro (Followers: 100,000 to 1 million)")
influencer_mega = st.checkbox("Influencer Mega (Followers: 1 million and above)")
influencer_micro = st.checkbox("Influencer Micro (Followers: 10,000 to 100,000)")
influencer_nano = st.checkbox("Influencer Nano (Followers: 1,000 to 10,000)")

# Prepare input data
influencer_values = [
    int(influencer_macro), 
    int(influencer_mega), 
    int(influencer_micro), 
    int(influencer_nano)
]

features = np.array([tv, radio, social_media] + influencer_values).reshape(1, -1)

# Predict button
if st.button("Predict Sales"):
    prediction = model.predict(features)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

