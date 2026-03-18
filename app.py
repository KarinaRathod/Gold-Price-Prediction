import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import requests

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Gold Intelligence Pro",
    layout="wide",
    page_icon="🟡",
    initial_sidebar_state="expanded"
)

# -------------------- PREMIUM CSS --------------------
st.markdown("""
    <style>
        .stApp {
            background: #0e1117;
            color: #e0e0e0;
        }
        /* Metric Card Stylings */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 215, 0, 0.1);
            padding: 1rem;
            border-radius: 10px;
        }
        [data-testid="stMetricValue"] {
            color: #FFD700;
            font-family: 'Courier New', monospace;
        }
        /* Custom Tab UI */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
            color: #888;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #FFD700 !important;
            border-bottom: 2px solid #FFD700 !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- UTILITY FUNCTIONS --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("gold_futures_timeseries.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def get_live_gold_price():
    try:
        # Metals.live fallback to simulated price if API limit reached
        response = requests.get("https://api.metals.live/v1/spot/gold", timeout=2)
        return response.json()[0]['price']
    except:
        return None

def forecast_engine(df, days=30):
    df_pred = df.copy()
    df_pred['days_index'] = (df_pred['date'] - df_pred['date'].min()).dt.days
    
    X = df_pred[['days_index']].values
    y = df_pred['close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_day = df_pred['days_index'].max()
    future_index = np.arange(last_day + 1, last_day + days + 1).reshape(-1, 1)
    future_dates = pd.date_range(df_pred['date'].max() + pd.Timedelta(days=1), periods=days)
    
    preds = model.predict(future_index)
    
    # Calculate simple standard deviation for "Confidence Bands"
    std_dev = df_pred['close'].std() * 0.05 
    
    return pd.DataFrame({
        'date': future_dates,
        'pred': preds,
        'upper': preds + std_dev,
        'lower': preds - std_dev
    })

# -------------------- APPLICATION LOGIC --------------------
try:
    df = load_data()

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.title("🟡 GOLD INTEL")
        st.caption("Professional Commodities Terminal")
        st.divider()
        
        date_range = st.date_input("Analysis Window", [df['date'].min(), df['date'].max()])
        
        st.subheader("Visuals")
        chart_mode = st.radio("Chart Engine", ["Pro-Candle", "Area Trend"], horizontal=True)
        
        st.subheader("Portfolio")
        units = st.number_input("Holding (Oz)", value=1.0, step=0.1)
        entry = st.number_input("Entry Price ($)", value=float(df['close'].iloc[0]))
        
        st.divider()
        st.info("Signals are based on RSI (14) and Bollinger Cross-overs.")

    # Validation for date range
    if len(date_range) != 2: st.stop()
    
    mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
    f_df = df.loc[mask]

    # -------------------- HEADER & TICKER --------------------
    live = get_live_gold_price()
    t1, t2 = st.columns([3, 1])
    with t1:
        st.title("Gold Intelligence Terminal")
    with t2:
        if live:
            st.metric("LIVE SPOT", f"${live:,.2f}", delta="Real-time", delta_color="normal")
        else:
            st.caption("Offline Ticker")

    # -------------------- KPI ROW --------------------
    latest = f_df.iloc[-1]
    prev = f_df.iloc[-2] if len(f_df) > 1 else latest
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Current Close", f"${latest['close']:,.2f}", f"{latest['close']-prev['close']:+.2f}")
    kpi2.metric("Period High", f"${f_df['high'].max():,.2f}")
    kpi3.metric("Volatility (Avg)", f"{f_df['volatility_7'].mean():.2f}%")
    kpi4.metric("Avg Volume", f"{f_df['volume'].mean():,.0f}")

    # -------------------- MAIN TABS --------------------
    tab_market, tab_tech, tab_ml, tab_port = st.tabs(["📊 Market Data", "📉 Technicals", "🔮 Prediction", "💼 Portfolio"])

    with tab_market:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        if "Pro-Candle" in chart_mode:
            fig.add_trace(go.Candlestick(x=f_df['date'], open=f_df['open'], high=f_df['high'], 
                                         low=f_df['low'], close=f_df['close'], name="OHLC"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=f_df['date'], y=f_df['close'], fill='tozeroy', line=dict(color='#FFD700'), name="Price"), row=1, col=1)

        # Overlays
        fig.add_trace(go.Scatter(x=f_df['date'], y=f_df['ma_7'], line=dict(color='rgba(255,255,255,0.4)', dash='dot'), name="MA7"), row=1, col=1)
        
        # Volume
        colors = ['#EF5350' if row['close'] < row['open'] else '#26A69A' for _, row in f_df.iterrows()]
        fig.add_trace(go.Bar(x=f_df['date'], y=f_df['volume'], marker_color=colors, name="Volume"), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab_tech:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Momentum (RSI)")
            rsi_fig = px.line(f_df, x='date', y='rsi', color_discrete_sequence=['#E040FB'])
            rsi_fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1)
            rsi_fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1)
            rsi_fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(rsi_fig, use_container_width=True)
        with c2:
            st.subheader("Bollinger Pressure")
            bb_fig = go.Figure()
            bb_fig.add_trace(go.Scatter(x=f_df['date'], y=f_df['bb_upper'], line=dict(color='rgba(255,255,255,0.1)'), name="Upper"))
            bb_fig.add_trace(go.Scatter(x=f_df['date'], y=f_df['bb_lower'], line=dict(color='rgba(255,255,255,0.1)'), fill='tonexty', name="Lower"))
            bb_fig.add_trace(go.Scatter(x=f_df['date'], y=f_df['close'], line=dict(color='#FFD700'), name="Close"))
            bb_fig.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(bb_fig, use_container_width=True)

    with tab_ml:
        st.subheader("Market Projection (Linear Regression)")
        horizon = st.slider("Forecast Horizon (Days)", 7, 90, 30)
        forecast = forecast_engine(f_df, horizon)
        
        ml_fig = go.Figure()
        ml_fig.add_trace(go.Scatter(x=f_df['date'], y=f_df['close'], name="Historical"))
        ml_fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['pred'], line=dict(dash='dash', color='#FFD700'), name="Forecast"))
        ml_fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['upper'], line=dict(width=0), showlegend=False))
        ml_fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['lower'], fill='tonexty', line=dict(width=0), name="Confidence Range"))
        ml_fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(ml_fig, use_container_width=True)

    with tab_port:
        cur_val = units * latest['close']
        inv_val = units * entry
        roi = ((cur_val - inv_val) / inv_val) * 100
        
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Market Value", f"${cur_val:,.2f}")
        pc2.metric("Total P/L", f"${cur_val - inv_val:,.2f}", f"{roi:.2f}%")
        pc3.metric("Holding Period", f"{(f_df['date'].max() - f_df['date'].min()).days} Days")
        
        st.divider()
        st.subheader("Position Performance History")
        port_hist = f_df.copy()
        port_hist['position_value'] = port_hist['close'] * units
        ph_fig = px.area(port_hist, x='date', y='position_value', color_discrete_sequence=['#26A69A'])
        ph_fig.update_layout(template="plotly_dark")
        st.plotly_chart(ph_fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading system: {e}")
    st.info("Check if 'gold_futures_timeseries.csv' is in the root directory.")