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
hour = st.slider("Select Hour (0–23)", 0, 23, 12)
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day = st.selectbox("Select Day of the Week", day_names)
day_index = day_names.index(day)

prediction = model.predict([[hour, day_index]])
st.write(f"*Predicted Crime:* {prediction[0]}")

st.subheader("Crime Count by Hour")
hourly_crimes = df.groupby('hour')['primary_type'].count()
st.bar_chart(hourly_crimes)
