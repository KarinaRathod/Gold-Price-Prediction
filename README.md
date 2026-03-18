
# 🟡 Gold Intelligence Pro: FinTech Dashboard

An advanced financial analytics terminal built with Python and Streamlit for analyzing historical gold futures data, monitoring live spot prices, and forecasting future trends using Machine Learning.

## ✨ Features
- **Interactive Market Charts**: Pro-grade Candlestick and Area charts with Volume subplots.
- **Technical Indicators**: Real-time calculation of RSI, Moving Averages, and Bollinger Bands.
- **ML Forecasting**: Price projection using Linear Regression with confidence intervals.
- **Live Ticker**: Fetches real-time gold spot prices via API.
- **Portfolio Tracker**: Monitor ROI, Profit/Loss, and holding performance.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd gold-analytics-pro
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   Ensure the file `gold_futures_timeseries.csv` is located in the root directory.

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Visuals**: [Plotly](https://plotly.com/python/)
- **Data**: Pandas, Numpy
- **ML**: Scikit-Learn
- **API**: Metals.live API
```

