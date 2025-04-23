import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.title("Crime Data Analysis & Prediction App")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("chicago_crime_sample.csv")

df = load_data()

st.subheader("Raw Data Sample")
st.write(df.sample(10))

# Simple prediction mockup (based on most common crimes at specific hours)
st.subheader("Predict Crime Type")
hour = st.slider("Select Hour (0–23)", 0, 23, 12)
common_crime = df[df['hour'] == hour]['primary_type'].mode()[0]
st.write(f"Most likely crime at {hour}:00 is *{common_crime}*")
