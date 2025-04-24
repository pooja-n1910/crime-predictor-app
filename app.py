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
    df = df[['date', 'primary_type', 'latitude', 'longitude']]
    df.dropna(inplace=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Mon, 6=Sun

    return df

# Model training
st.subheader("Train Crime Prediction Model")
import joblib
import os

# Model training
st.subheader("Train Crime Prediction Model")

model_file = "model.pkl"

if os.path.exists(model_file):
    model = joblib.load(model_file)
    st.success("Model loaded from saved file.")
else:
    X = df[['hour', 'day_of_week']]
    y = df['primary_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)  # Save model
    st.success("Model trained and saved.")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.write(f"Model Accuracy: {acc:.2f}")

# Prediction
st.subheader("Predict Crime Type")
hour = st.slider("Select Hour (0–23)", 0, 23, 12)
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day = st.selectbox("Select Day of the Week", day_names)
day_index = day_names.index(day)

prediction = model.predict([[hour, day_index]])
st.write(f"**Predicted Crime:** {prediction[0]}")
# Show crime map
st.subheader("Crime Locations Map")

# Optional: limit number of points shown for performance
st.map(df[['latitude', 'longitude']].sample(500))


st.subheader("Crime Count by Hour")
hourly_crimes = df.groupby('hour')['primary_type'].count()
st.bar_chart(hourly_crimes)
