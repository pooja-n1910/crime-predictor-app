import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Crime Data Analysis & Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("chicago_crime_sample.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    return df

df = load_data()

st.subheader("Sample Data with Features")
st.write(df[['hour', 'day_of_week', 'primary_type']].head())

# Model training
st.subheader("Train Crime Prediction Model")

X = df[['hour', 'day_of_week']]
y = df['primary_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
st.write(f"Model Accuracy: {acc:.2f}")

# Prediction
st.subheader("Predict Crime Type")
hour = st.slider("Hour (0–23)", 0, 23, 12)
day = st.slider("Day of the Week (0 = Mon, 6 = Sun)", 0, 6, 3)

prediction = model.predict([[hour, day]])
st.write(f"Predicted Crime: *{prediction[0]}*")
