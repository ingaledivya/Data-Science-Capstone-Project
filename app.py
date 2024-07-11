import streamlit as st
import pandas as pd
import pickle

# Load the trained model, scaler, label encoder, and expected columns

with open('used_car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 
with open('label_encoder.pkl', 'rb') as file:
    label_encoders = pickle.load(file)
with open('columns.pkl', 'rb') as file:
    expected_columns = pickle.load(file)

# Define the user input form
st.title("Car Price Prediction")

name = st.selectbox("Car Name", [ "Maruti 800","Hyundai i20", "Honda City","Maruti Wagon R LXI Minor","Hyundai Verna 1.6 SX","Datsun RediGO T Option","Honda Amaze VX i-DTEC","Hyundai i20 Magna 1.4 CRDi (Diesel)","Hyundai i20 Magna 1.4 CRDi","Maruti 800 AC BSIII","Renault KWID RXT","Hyundai Creta 1.6 CRDi SX Option"])
age = st.number_input("Age of the car", min_value=0)
Present_Price = st.number_input("Present Price of the car (in lakhs)")
Km_Driven = st.number_input("Kilometers Driven")
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of previous owners", ['First Owner', 'Second Owner', 'Fourth & Above Owner', 'Third Owner', 'Test Drive Car'])

# Create the input DataFrame
input_data = {
    'name':[name],
    'age': [age],
    'Present_Price': [Present_Price],
    'Km_Driven': [Km_Driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner]
}
input_df = pd.DataFrame(input_data)

# Encode the categorical features
for col in ['name','fuel', 'seller_type', 'transmission', 'owner']:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])


# Ensure the DataFrame columns match the expected columns
input_df = input_df[expected_columns]

# Scale the input data
scaled_input = scaler.transform(input_df)

# Predict the car price
predicted_price = model.predict(scaled_input)

st.write(f"The predicted selling price of the car is: {predicted_price[0]:.2f} lakhs")
