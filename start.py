from datetime import datetime
import pandas as pd
import numpy as np
import ast
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import google.generativeai as genai

# ---------- Streamlit Config ----------
st.set_page_config(page_title="TSLA Stock Analysis", layout="wide")
st.title("📊 TSLA Dashboard with Agentic AI Chatbot 🤖")
st.markdown("This app provides a candlestick chart for TSLA stock data and an AI chatbot to answer your questions about the data.")

# ---------- Load and Prepare Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA_data.csv", parse_dates=["timestamp"])
    df.columns = df.columns.str.lower()
    df.sort_values("timestamp", inplace=True)
    df["support"] = df["support"].apply(ast.literal_eval)
    df["resistance"] = df["resistance"].apply(ast.literal_eval)
    df['time'] = df['timestamp'].astype('int64') // 10**9
    df['sma_5'] = df['close'].rolling(window=5).mean()
    return df

df = load_data()

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📈 Chart", "🤖 Chatbot"])

# ---------- Candlestick Chart Tab ----------
with tab1:
    st.subheader("📉 TSLA Candlestick Chart")

    # --- Date range picker ---
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

    # Validate date input (make sure two dates selected)
    if len(date_range) != 2:
        st.warning("Please select a start and end date.")
    else:
        start_date, end_date = date_range

        # Filter dataframe by date range (inclusive)
        filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)].copy()

        if filtered_df.empty:
            st.warning("No data available for the selected date range.")
        else:
            df_clean = filtered_df.dropna(subset=['time', 'open', 'high', 'low', 'close']).drop_duplicates(subset=['time'])

            candlestick_data = df_clean[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')

            filtered_df['support_level'] = filtered_df['support'].apply(lambda x: x[0] if x else None)
            filtered_df['resistance_level'] = filtered_df['resistance'].apply(lambda x: x[0] if x else None)
            df_sr = filtered_df.dropna(subset=['support_level', 'resistance_level'])

            support_data = df_sr[['time', 'support_level']].rename(columns={'support_level': 'value'}).to_dict(orient='records')
            resistance_data = df_sr[['time', 'resistance_level']].rename(columns={'resistance_level': 'value'}).to_dict(orient='records')

            area_data = [{"time": point["time"], "value": point["close"]} for point in candlestick_data]

            markers = [
                {
                    "time": row['time'],
                    "position": "aboveBar" if row['direction'] == 'LONG' else "belowBar",
                    "color": "green" if row['direction'] == 'LONG' else "red",
                    "shape": "arrowUp" if row['direction'] == 'LONG' else "arrowDown"
                }
                for _, row in filtered_df.iterrows() if pd.notna(row['direction'])
            ]

            moving_avg_data = filtered_df.dropna(subset=['sma_5'])[['time', 'sma_5']].rename(columns={'sma_5': 'value'}).to_dict(orient='records')

            series = [
                {"type": "Candlestick", "data": candlestick_data,
                 "options": {"upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False,
                             "wickUpColor": '#26a69a', "wickDownColor": '#ef5350'}},
                {"type": "Line", "data": support_data, "options": {"color": "blue", "lineWidth": 1, "lineStyle": 2}},
                {"type": "Line", "data": resistance_data, "options": {"color": "red", "lineWidth": 1, "lineStyle": 2}},
                {"type": "Area", "data": area_data, "options": {"topColor": 'rgba(0,0,0,0)', "bottomColor": 'rgba(0,0,0,0)',
                                                               "lineColor": 'rgba(0,0,0,0)', "lineWidth": 0},
                 "markers": markers},
                {"type": "Line", "data": moving_avg_data, "options": {"color": "cyan", "lineWidth": 2}}
            ]

            chartOptions = {
                "height": 800,
                "layout": {"textColor": 'white', "background": {"type": 'solid', "color": 'black'}}
            }

            renderLightweightCharts([{"chart": chartOptions, "series": series}], 'candlestick')


# ---------- Chatbot UI Tab ----------
with tab2:
    st.subheader("🤖 Ask the AI About TSLA Stock")

    # Stats Summary
    bullish_days = df[df['direction'] == "LONG"].shape[0]
    bearish_days = df[df['direction'] == "SHORT"].shape[0]
    neutral_days = df[df['direction'].isna()].shape[0]
    avg_close = df['close'].mean()
    max_close = df['close'].max()
    min_close = df['close'].min()
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")

    summary = f"""
    You are a stock analysis assistant. Use TSLA stock data from {start_date} to {end_date} to answer questions.
    if u cant find the answer, say "I don't know".
    
    The data includes:
    - Bullish Days (LONG): {bullish_days}
    - Bearish Days (SHORT): {bearish_days}
    - Neutral Days: {neutral_days}
    - Avg Close Price: ${avg_close:.2f}
    - Max Close: ${max_close:.2f}
    - Min Close: ${min_close:.2f}
    """

    genai.configure(api_key="AIzaSyCu3ju_Tvcx5KbZNpmhdAJo1eqS5av73xA")  # Replace with your actual Gemini API key
    model = genai.GenerativeModel("gemini-2.0-flash")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat UI Input
    user_input = st.chat_input("Ask something about TSLA data...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("Gemini is thinking..."):
            prompt = f"{summary}\nQuestion: {user_input}"
            ai_response = model.generate_content(prompt)
            st.session_state.chat_history.append(("ai", ai_response.text))

         # ---- Suggested Questions by Category ----
    def handle_suggestion(question):
        st.session_state.chat_history.append(("user", question))
        with st.spinner("Gemini is thinking..."):
            prompt = f"{summary}\nQuestion: {question}"
            ai_response = model.generate_content(prompt)
            st.session_state.chat_history.append(("ai", ai_response.text))

    st.markdown("#### 💡 Suggested Questions")

    with st.expander("📈 Price Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What is the average closing price of TSLA?"):
                handle_suggestion("What is the average closing price of TSLA?")
            if st.button("What is the maximum and minimum closing price?"):
                handle_suggestion("What is the maximum and minimum closing price?")
        with col2:
            if st.button("Is the average price closer to the min or max?"):
                handle_suggestion("Is the average price closer to the min or max?")
            if st.button("What does the price range tell us about volatility?"):
                handle_suggestion("What does the price range tell us about volatility?")

    with st.expander("📊 Trend & Direction"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("How many bullish days are there?"):
                handle_suggestion("How many bullish days are there?")
            if st.button("How many bearish days are there?"):
                handle_suggestion("How many bearish days are there?")
        with col2:
            if st.button("How many neutral days are in the dataset?"):
                handle_suggestion("How many neutral days are in the dataset?")
            if st.button("Was TSLA more bullish or bearish overall?"):
                handle_suggestion("Was TSLA more bullish or bearish overall?")

    with st.expander("📅 Timeframe & Coverage"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What is the date range of this dataset?"):
                handle_suggestion("What is the date range of this dataset?")
            if st.button("How many trading days are included?"):
                handle_suggestion("How many trading days are included?")
        with col2:
            if st.button("What period did TSLA show strongest growth?"):
                handle_suggestion("What period did TSLA show strongest growth?")
            if st.button("What happened during the last week of the dataset?"):
                handle_suggestion("What happened during the last week of the dataset?")

    # ---- Display Chat History with Avatars and Timestamps ----
    for i, (sender, message) in enumerate(st.session_state.chat_history):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if sender == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**You**  \n{message}")
                st.caption(f"🕒 {timestamp}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"{message}")
                st.caption(f"🕒 {timestamp}")
