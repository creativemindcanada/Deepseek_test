# Live Market Data
st.subheader("Live Market Data")

def get_market_data():
    try:
        api_key = "YOUR_API_KEY"  # Replace with your actual API key
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey={api_key}"
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch market data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

market_data = get_market_data()

if market_data:
    # Extract the latest data
    time_series = market_data.get("Time Series (Daily)", {})
    latest_date = list(time_series.keys())[0]  # Get the latest date
    latest_data = time_series[latest_date]

    # Display the latest stock price
    st.write(f"**Latest Stock Price (as of {latest_date})**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Open", f"${latest_data['1. open']}")
        st.metric("High", f"${latest_data['2. high']}")
    with col2:
        st.metric("Close", f"${latest_data['4. close']}")
        st.metric("Low", f"${latest_data['3. low']}")
    with col3:
        st.metric("Volume", latest_data['5. volume'])

    # Prepare data for the trend chart
    dates = list(time_series.keys())[:30]  # Last 30 days
    closing_prices = [float(time_series[date]["4. close"]) for date in dates]
    trend_data = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Closing Price": closing_prices
    })

    # Display the trend chart
    st.write("**Stock Price Trend (Last 30 Days)**")
    fig = px.line(trend_data, x="Date", y="Closing Price", title="Closing Price Over Time")
    st.plotly_chart(fig)

else:
    st.warning("No market data available.")
