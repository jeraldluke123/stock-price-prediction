import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(
    page_title="Next‑Day Stock Price Prediction",
    page_icon="📈",
    layout="wide",
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=True)
def load_data(ticker: str, years: int = 5) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=365 * years + 10)
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError("No data returned. Check the ticker symbol.")
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X['Return_1d'] = X['Close'].pct_change()
    X['High_Low_Spread'] = (X['High'] - X['Low']) / X['Close']
    X['Open_Close_Change'] = (X['Close'] - X['Open']) / X['Open']

    # Rolling stats
    for w in [3, 5, 10, 20]:
        X[f'SMA_{w}'] = X['Close'].rolling(window=w).mean()
        X[f'VOL_{w}'] = X['Volume'].rolling(window=w).mean()
        X[f'RET_STD_{w}'] = X['Return_1d'].rolling(window=w).std()

    # Lags
    for l in [1, 2, 3, 5, 10]:
        X[f'Close_lag{l}'] = X['Close'].shift(l)
        X[f'Return_lag{l}'] = X['Return_1d'].shift(l)

    # Target: next-day close
    X['y_next_close'] = X['Close'].shift(-1)
    X = X.dropna()
    return X


def naive_baseline_pred(last_close: float) -> float:
    # Predict next close = last close
    return float(last_close)


def sma_baseline_pred(close_series: pd.Series, window: int = 5) -> float:
    return float(close_series.tail(window).mean())


# ---------------------------
# UI
# ---------------------------
st.title("📈 Next‑Day Stock Price Prediction")
st.markdown(
    "Enter an NSE/BSE/US ticker (e.g., **RELIANCE.NS**, **TCS.NS**, **HDFCBANK.NS**, **AAPL**, **MSFT**).\\n\\n"
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.text_input("Ticker", value="RELIANCE.NS").strip()
with col2:
    years = st.slider("Years of history", 2, 12, 5)
with col3:
    test_size_days = st.slider("Test size (days)", 30, 180, 90)

run = st.button("🚀 Train & Predict", type="primary")

# ---------------------------
# Main logic
# ---------------------------
if run and ticker:
    try:
        with st.spinner("Fetching data and training model..."):
            raw = load_data(ticker, years)
            df = add_features(raw)

            # Train/test split by time (last N days as test)
            test_size_days = min(test_size_days, len(df) // 5 if len(df) >= 200 else 30)
            X_all = df.drop(columns=['y_next_close'])
            y_all = df['y_next_close']

            X_train, X_test = X_all.iloc[:-test_size_days], X_all.iloc[-test_size_days:]
            y_train, y_test = y_all.iloc[:-test_size_days], y_all.iloc[-test_size_days:]

            features = [c for c in X_all.columns if c not in raw.columns]
            X_train = X_train[features]
            X_test = X_test[features]

            # Random Forest (robust default)
            model = RandomForestRegressor(
                n_estimators=350,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Error metrics on the holdout window
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # Baselines for comparison
            last_close = float(raw['Close'].iloc[-1])
            sma5 = sma_baseline_pred(raw['Close'], 5)
            naive_next = naive_baseline_pred(last_close)

            # Next-day prediction using last available row
            last_row = df.iloc[[-1]][features]
            next_close_pred = float(model.predict(last_row)[0])

            # Simple uncertainty band from validation residuals
            resid = y_test - y_pred
            resid_std = float(resid.std())
            lower = next_close_pred - 1.96 * resid_std
            upper = next_close_pred + 1.96 * resid_std

        # ---- Display ----
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (₹)", f"{mae:,.2f}")
        m2.metric("MAPE", f"{mape*100:,.2f}%")
        m3.metric("Test days", f"{len(y_test)}")

        st.subheader("Next‑Day Close Prediction")
        st.markdown(f"**Ticker:** `{ticker}`")
        st.metric(
            "Predicted Next Close (₹/$/local)",
            f"{next_close_pred:,.2f}",
            delta=f"± {1.96*resid_std:,.2f} (95% band)",
        )
        st.caption("Uncertainty band estimated from recent validation residuals; not a statistical guarantee.")

        # Plot actual vs predicted for the test window
        plot_df = pd.DataFrame({
            'Date': y_test.index,
            'Actual': y_test.values,
            'Predicted': y_pred,
        }).set_index('Date')
        st.line_chart(plot_df)

        # Baseline comparison
        st.subheader("Baselines")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.write("**Naive (next = last close):**", f"{naive_next:,.2f}")
        with bcol2:
            st.write("**SMA‑5 (simple average of last 5 closes):**", f"{sma5:,.2f}")

        # Raw data preview
        with st.expander("Show raw price data"):
            st.dataframe(raw.tail(200))

        # Tips
        st.info(
            """
            **Tips**
            - Use NSE tickers with `.NS` suffix (e.g., `INFY.NS`, `HDFCBANK.NS`).
            - Increase *Years of history* if the chart looks too short.
            - This is a simple educational model; do not use it as financial advice.
            """
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

else:
    st.caption("Enter a ticker and click *Train & Predict* to get started.")
