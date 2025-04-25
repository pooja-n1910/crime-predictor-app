import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing 
import LabelEncoder 
from sklearn.ensemble 
import RandomForestClassifier

Load data

df = pd.read_csv("chicago_crime_full.csv")

Feature engineering

df['Date'] = pd.to_datetime(df['Date']) df['hour'] = df['Date'].dt.hour df['day_of_week'] = df['Date'].dt.dayofweek  # Monday=0

Select features and target

features = ['hour', 'day_of_week', 'Domestic', 'Arrest', 'Community Area', 'Location Description'] df = df.dropna(subset=features + ['Primary Type']) X = df[features] y = df['Primary Type']

Encode categorical features

le_location = LabelEncoder() X['Location Description'] = le_location.fit_transform(X['Location Description'])

le_target = LabelEncoder() y_encoded = le_target.fit_transform(y)

Train model

model = RandomForestClassifier(n_estimators=100, random_state=42) model.fit(X, y_encoded)

Save model and encoders

joblib.dump(model, "crime_model.pkl") joblib.dump(le_location, "location_encoder.pkl") joblib.dump(le_target, "target_encoder.pkl")

Streamlit App

st.title("Chicago Crime Type Predictor")

st.subheader("Try Predicting a Crime Type") input_hour = st.slider("Select Hour of the Day", 0, 23, 12) input_day = st.selectbox("Select Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]) input_domestic = st.selectbox("Was it Domestic?", ["TRUE", "FALSE"]) input_arrest = st.selectbox("Was there an Arrest?", ["TRUE", "FALSE"]) input_community = st.number_input("Community Area (e.g. 25)", min_value=1, max_value=100, value=25) input_location = st.selectbox("Location Description", le_location.classes_)

Map day to number

day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

Prepare input

input_data = pd.DataFrame({ 'hour': [input_hour], 'day_of_week': [day_map[input_day]], 'Domestic': [input_domestic == "TRUE"], 'Arrest': [input_arrest == "TRUE"], 'Community Area': [input_community], 'Location Description': [le_location.transform([input_location])[0]] })

Load model and encoders

model = joblib.load("crime_model.pkl") le_target = joblib.load("target_encoder.pkl")

Predict

prediction_encoded = model.predict(input_data)[0] prediction_label = le_target.inverse_transform([prediction_encoded])[0]

st.success(f"Predicted Crime Type: {prediction_label}")
