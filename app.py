import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Crime Data Analysis & Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("chicago_crime_full.csv")
    df = df[['Date', 'Primary Type', 'Latitude', 'Longitude']]
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Mon, 6=Sun
    return df

# Load data
df = load_data()

# step1:crime type distribution
st.subheader("Crime Type Distribution")
crime_counts = df['Primary Type'].value_counts()
st.bar_chart(crime_counts)

# Step 2: Hourly Crime Pattern
st.subheader("Hourly Crime Pattern")
hourly_crime = df['hour'].value_counts().sort_index()
st.line_chart(hourly_crime)

# Step 3: Crimes by Day of the Week
st.subheader("Crimes by Day of the Week")
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_crime = df['day_of_week'].value_counts().sort_index()
day_crime.index = [days[i] for i in day_crime.index]
st.bar_chart(day_crime)

# Model training
st.subheader("Train Crime Prediction Model")

X = df[['hour', 'day_of_week']]
y = df['Primary Type']

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write("Model Accuracy:" round(acc,2))

# Save model
model_file = "model.pkl"
joblib.dump(model, model_file)

# Step 4: Try Predicting a Crime Type
st.subheader("Try Predicting a Crime Type")

input_hour = st.slider("Select Hour of Day", 0, 23, 12)
input_day = st.selectbox("Select Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Map day to number
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}
input_df = pd.DataFrame({
    'hour': [input_hour],
    'day_of_week': [day_map[input_day]]
})

# Predict and decode
predicted_encoded = model.predict(input_df)[0]
predicted_crime = le.inverse_transform([predicted_encoded])[0]
st.success(f"Predicted Crime Type: {predicted_crime}")
