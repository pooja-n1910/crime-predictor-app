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

# Model training
st.subheader("Train Crime Prediction Model")

X = df[['hour', 'day_of_week']]
y = df['primary_type']

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {acc:.2f}")

# Save model
model_file = "model.pkl"
joblib.dump(model, model_file)
