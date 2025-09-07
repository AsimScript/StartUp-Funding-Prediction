
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Startup Funding Predictor", layout="wide")

# Load model
model = joblib.load("funding_model.pkl")

# Title and intro
st.markdown("""
# 🚀 Indian Startup Funding Predictor

This app predicts the expected **funding amount (in INR crores)** a startup may receive,
based on its domain, location, funding round, and other factors.
""")

st.markdown("---")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    vertical = st.selectbox("📌 Startup Vertical", ['E-Tech', 'E-commerce', 'FinTech', 'Fashion and Apparel', 'Hospitality'])
    subvertical = st.text_input("🔍 Subvertical (e.g., Online Food Delivery)", "Online Learning")
    city = st.selectbox("🌆 City", ['Bengaluru', 'Mumbai', 'New Delhi', 'Gurgaon', 'Hyderabad', 'Pune'])

with col2:
    round_type = st.selectbox("💸 Funding Round", ['Seed Round', 'Series A', 'Series B', 'Private Equity Round', 'Pre-series A'])
    year = st.slider("📅 Funding Year", 2015, 2021, 2020)
    month = st.slider("🗓️ Funding Month", 1, 12, 6)
    num_investors = st.slider("👥 Number of Investors", 1, 10, 2)

if st.button("🔮 Predict Funding Amount"):
    input_data = pd.DataFrame({
        'vertical': [vertical],
        'subvertical': [subvertical],
        'city': [city],
        'round': [round_type],
        'year': [year],
        'month': [month],
        'num_investors': [num_investors]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Funding Amount: ₹ {prediction:.2f} crores")

# Optional: EDA Visuals
st.markdown("---")
st.subheader("📊 Quick Insights from Dataset")

# Load data for visualization
df = pd.read_csv("startup_cleaned.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year

col3, col4 = st.columns(2)

with col3:
    top_cities = df['city'].value_counts().head(5)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_cities.values, y=top_cities.index, ax=ax1, palette='Blues_d')
    ax1.set_title("🏙️ Top 5 Funded Cities")
    st.pyplot(fig1)

with col4:
    top_rounds = df['round'].value_counts().head(5)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_rounds.values, y=top_rounds.index, ax=ax2, palette='Greens')
    ax2.set_title("💼 Popular Funding Rounds")
    st.pyplot(fig2)

# Trend chart
st.markdown("---")

st.subheader("📈 Yearly Funding Trends")
funding_by_year = df.groupby('year')['amount'].sum().reset_index()
fig3, ax3 = plt.subplots()
sns.lineplot(data=funding_by_year, x='year', y='amount', marker='o', ax=ax3)
ax3.set_title("📆 Total Funding per Year")
ax3.set_ylabel("Amount (INR Crores)")
ax3.grid(True)
st.pyplot(fig3)
