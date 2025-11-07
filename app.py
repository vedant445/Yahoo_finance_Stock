import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Market Direction Prediction App", layout="centered")

st.title("💹 Stock Market Direction Prediction App")
st.write("Predict whether the stock price will go Up 📈 or Down 📉 using SVM + PCA on technical indicators.")

# --- Inputs ---
tickers_input = st.text_input("Enter Stock Ticker Symbols (comma-separated)", value="AAPL,MSFT,GOOGL").upper()
days = st.slider("Number of past days for analysis", min_value=60, max_value=730, value=365)
horizon = st.slider("Prediction horizon (days ahead)", 1, 7, 1)  # New horizon slider

tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

if st.button("🔍 Predict Direction"):
    for ticker in tickers:
        st.markdown(f"## 🏷 {ticker}")
        try:
            data = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
            if data.empty:
                st.error(f"⚠️ No data found for {ticker}.")
                continue
        except Exception as e:
            st.error(f"⚠️ Error downloading {ticker}: {e}")
            continue

        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        if len(df) < 30:
            st.warning(f"⚠️ Not enough data for {ticker} (need at least 30 rows). Skipping.")
            continue

        # --- Feature Engineering ---
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        roll_mean = df['Close'].rolling(20).mean()
        roll_std = df['Close'].rolling(20).std()
        df['BB_UPPER'] = roll_mean + 2 * roll_std
        df['BB_LOWER'] = roll_mean - 2 * roll_std
        df['RET'] = df['Close'].pct_change()
        df['MOMENTUM'] = df['Close'] - df['Close'].shift(10)

        # --- Target ---
        df['Future_Close'] = df['Close'].shift(-horizon)
        df.dropna(inplace=True)
        df['Target'] = (df['Future_Close'].values.flatten() > df['Close'].values.flatten()).astype(int)

        # --- Features ---
        features = ['Open','High','Low','Close','Volume','SMA_10','EMA_10','RSI','MACD',
                    'BB_UPPER','BB_LOWER','RET','MOMENTUM']
        X = df[features].values
        y = df['Target'].values

        st.write("Class distribution:", pd.Series(y).value_counts())

        # --- Scale & PCA ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        # --- Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, shuffle=False)
        svm = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.write(f"Historical test accuracy: {acc:.2f}")

        # --- Predict next day ---
        last_row = X_pca[-1].reshape(1, -1)
        pred = svm.predict(last_row)[0]
        pred_proba = svm.predict_proba(last_row)[0]

        st.markdown("### Prediction for next trading day:")
        if pred == 1:
            st.markdown(f"📈 **Up**")
        else:
            st.markdown(f"📉 **Down**")
        st.markdown(f"Confidence → Up: {pred_proba[1]*100:.2f}% | Down: {pred_proba[0]*100:.2f}%")

        # --- Historical Price Chart ---
        st.markdown("📆 **Historical Price Chart**")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{ticker} Historical Close Price')
        ax.legend()
        st.pyplot(fig)

        # --- PCA Feature Space (2D Projection) ---
        st.markdown("🎯 **PCA Feature Space (2D Projection)**")
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X_scaled)
        fig2, ax2 = plt.subplots(figsize=(6,5))
        ax2.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='bwr', alpha=0.6)
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')
        ax2.set_title('2D PCA Projection (Red=Down, Blue=Up)')
        st.pyplot(fig2)

        st.markdown("---")

st.markdown("0\nDeveloped with ❤️ using Streamlit + SVM + Yahoo Finance API")
st.info("💡 Tip: For feature importance, consider RandomForest or XGBoost, as SVM is a black-box model.")
