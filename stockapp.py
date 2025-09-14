import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data
def fetch_adjusted_close(tickers, start, end):
    """Download adjusted close prices for given tickers."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

def analyze_portfolio(price_df, weights=None, risk_free=0.02/252):
    """Calculate daily returns and portfolio statistics."""
    returns = price_df.pct_change().dropna()

    if weights is None:
        weights = np.ones(len(price_df.columns)) / len(price_df.columns)
    weights = np.array(weights)

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    port_return = np.dot(mean_returns, weights)
    port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe_ratio = (port_return - risk_free) / port_volatility

    return {
        "returns": returns,
        "mean_returns": mean_returns,
        "volatility": returns.std(),
        "correlation": returns.corr(),
        "portfolio_return": port_return,
        "portfolio_volatility": port_volatility,
        "sharpe_ratio": sharpe_ratio
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stock Portfolio Tracker", layout="wide")

st.title("📈 Stock Portfolio Tracker & Risk Analyzer")

# User Inputs
tickers = st.text_input("Enter tickers (comma separated):", "AAPL,MSFT,TSLA")
tickers = [t.strip().upper() for t in tickers.split(",")]

start = st.date_input("Start date", pd.to_datetime("2023-01-01"))
end = st.date_input("End date", pd.to_datetime("2024-01-01"))

# Fetch data
try:
    price_df = fetch_adjusted_close(tickers, start, end)
    if price_df.empty:
        st.error("No data fetched. Check tickers or dates.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Portfolio Weights
st.sidebar.header("Portfolio Weights")
weights = []
for ticker in tickers:
    weight = st.sidebar.slider(f"{ticker} Weight", 0.0, 1.0, 1.0/len(tickers), 0.01)
    weights.append(weight)

if sum(weights) != 1.0:
    st.sidebar.warning("⚠️ Weights should sum to 1. Normalizing automatically.")
    weights = [w/sum(weights) for w in weights]

# Run Analysis
results = analyze_portfolio(price_df, weights)

# Show Stats
st.subheader("📊 Portfolio Analysis")
col1, col2, col3 = st.columns(3)
col1.metric("Portfolio Return (daily)", f"{results['portfolio_return']:.4f}")
col2.metric("Portfolio Volatility", f"{results['portfolio_volatility']:.4f}")
col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

st.write("**Average Daily Returns:**")
st.dataframe(results["mean_returns"])

st.write("**Volatility (Std Dev):**")
st.dataframe(results["volatility"])

if len(tickers) > 1:
    st.write("**Correlation Matrix:**")
    st.dataframe(results["correlation"])

# Plots
st.subheader("📉 Stock Prices")
st.line_chart(price_df)

st.subheader("📈 Cumulative Returns")
cumulative_returns = (1 + results["returns"]).cumprod()
st.line_chart(cumulative_returns)

# Benchmark Comparison (SPY)
benchmark = fetch_adjusted_close("SPY", start, end).squeeze()
benchmark_returns = benchmark.pct_change().dropna()
benchmark_cum = (1 + benchmark_returns).cumprod()

st.subheader("📊 Portfolio vs. Benchmark (SPY)")
compare_df = pd.DataFrame({
    "Portfolio": (1 + np.dot(results["returns"], weights)).cumprod(),
    "SPY": benchmark_cum
})
st.line_chart(compare_df)
