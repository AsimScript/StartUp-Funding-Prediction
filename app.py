import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout='wide', page_title='Startup Funding Analysis')

# Load data and model
df = pd.read_csv('startup_cleaned.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
model = joblib.load("funding_model.pkl")

# ---------------------- ANALYSIS FUNCTIONS ---------------------- #
def load_overall_analysis():
    st.title('📊 Overall Analysis')

    total = round(df['amount'].sum())
    max_funding = df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0]
    avg_funding = df.groupby('startup')['amount'].sum().mean()
    num_startups = df['startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total', f"{total} cr")
    col2.metric('Max', f"{max_funding} cr")
    col3.metric('Avg', f"{round(avg_funding)} cr")
    col4.metric('Funded Startups', num_startups)

    st.header('📈 Month-on-Month Investment Trend')
    selected_option = st.selectbox('Select View Type', ['Total', 'Count'])
    if selected_option == 'Total':
        temp_df = df.groupby(['year', 'month'])['amount'].sum().reset_index()
    else:
        temp_df = df.groupby(['year', 'month'])['amount'].count().reset_index()

    temp_df['axis'] = temp_df['month'].astype(str) + ' - ' + temp_df['year'].astype(str)
    fig, ax = plt.subplots()
    ax.plot(temp_df['axis'], temp_df['amount'])
    ax.set_xticklabels(temp_df['axis'], rotation=45)
    st.pyplot(fig)


def load_investor_details(investor):
    st.title(f"💼 {investor} - Investor Insights")

    last5_df = df[df['investors'].str.contains(investor)].head()[['date', 'startup', 'vertical', 'city', 'round', 'amount']]
    st.subheader('📌 Most Recent Investments')
    st.dataframe(last5_df)

    col1, col2 = st.columns(2)
    with col1:
        big_series = df[df['investors'].str.contains(investor)].groupby('startup')['amount'].sum().sort_values(ascending=False).head()
        st.subheader('🏆 Biggest Investments')
        fig, ax = plt.subplots()
        ax.bar(big_series.index, big_series.values)
        st.pyplot(fig)

    with col2:
        vertical_series = df[df['investors'].str.contains(investor)].groupby('vertical')['amount'].sum()
        st.subheader('📂 Sectors Invested In')
        fig1, ax1 = plt.subplots()
        ax1.pie(vertical_series, labels=vertical_series.index, autopct='%0.1f%%')
        st.pyplot(fig1)

    col3, col4 = st.columns(2)
    with col3:
        round_series = df[df['investors'].str.contains(investor)].groupby('round')['amount'].sum()
        st.subheader('💸 Rounds Invested In')
        fig_round, ax_round = plt.subplots(figsize=(6, 6))
        ax_round.pie(round_series, labels=round_series.index, autopct='%0.1f%%')
        st.pyplot(fig_round)

    with col4:
        city_series = df[df['investors'].str.contains(investor)].groupby('city')['amount'].sum()
        st.subheader('🌆 Cities Invested In')
        fig_city, ax_city = plt.subplots(figsize=(6, 6))
        ax_city.pie(city_series, labels=city_series.index, autopct='%0.1f%%')
        st.pyplot(fig_city)

    year_series = df[df['investors'].str.contains(investor)].groupby('year')['amount'].sum()
    st.subheader('📅 Year-on-Year Investment')
    fig2, ax2 = plt.subplots()
    ax2.plot(year_series.index, year_series.values)
    st.pyplot(fig2)


def predict_funding():
    st.title("🔮 Startup Funding Amount Predictor")

    col1, col2 = st.columns(2)
    with col1:
        vertical = st.selectbox("📌 Vertical", df['vertical'].dropna().unique())
        subvertical = st.text_input("🔍 Subvertical", "Online Learning")
        city = st.selectbox("🌆 City", df['city'].dropna().unique())

    with col2:
        round_type = st.selectbox("💸 Funding Round", df['round'].dropna().unique())
        year = st.slider("📅 Year", 2015, 2021, 2020)
        month = st.slider("🗓️ Month", 1, 12, 6)
        num_investors = st.slider("👥 Number of Investors", 1, 10, 2)

    if st.button("Predict Funding Amount"):
        input_df = pd.DataFrame({
            'vertical': [vertical],
            'subvertical': [subvertical],
            'city': [city],
            'round': [round_type],
            'year': [year],
            'month': [month],
            'num_investors': [num_investors]
        })

        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Funding Amount: ₹ {prediction:.2f} crores")

# ---------------------- MAIN APP ---------------------- #
st.sidebar.title('🧭 Navigation')
option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'Startup', 'Investor', 'Predict Funding'])

if option == 'Overall Analysis':
    load_overall_analysis()

elif option == 'Startup':
    st.sidebar.selectbox('Select Startup', sorted(df['startup'].unique().tolist()))
    st.title('Startup Analysis (Coming Soon)')

elif option == 'Investor':
    selected_investor = st.sidebar.selectbox('Select Investor', sorted(set(df['investors'].str.split(',').sum())))
    if st.sidebar.button('Find Investor Details'):
        load_investor_details(selected_investor)

elif option == 'Predict Funding':
    predict_funding()
