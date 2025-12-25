import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Amazon Sale Traffic Predictor", layout="wide", page_icon="🤖")

# Custom CSS for a cleaner "Dashboard" look
st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #262730;
    }
    .stAlert {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Based Network Traffic Autocard")
st.markdown("### Real-time Great Indian Festival Simulation")

# --- 1. DATA GENERATION FUNCTION ---
def generate_synthetic_data(n_points=1000):
    time_steps = np.arange(n_points)
    # Base traffic: Sine wave (daily cycle) + Bias
    traffic = 50 + 30 * np.sin(0.1 * time_steps) 
    # Add random noise
    noise = np.random.normal(0, 5, n_points)
    traffic = traffic + noise
    return traffic.reshape(-1, 1)

# --- 2. TRAIN GRU MODEL (On the fly) ---
@st.cache_resource
def build_and_train_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    look_back = 10
    for i in range(len(scaled_data) - look_back - 1):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(GRU(50, return_sequences=False, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Slight increase in epochs for better demo accuracy
    model.fit(X, y, epochs=8, batch_size=32, verbose=0)
    return model, scaler, look_back

# --- 3. MAIN APP LOGIC ---

# Sidebar
st.sidebar.header("🎛️ Simulation Controls")
sale_multiplier = st.sidebar.slider("Sale Intensity (Traffic Multiplier)", 1.5, 5.0, 3.0)
speed = st.sidebar.slider("Refresh Rate (Speed)", 0.01, 1.0, 0.05)
st.sidebar.markdown("---")
st.sidebar.info("Model: GRU (Gated Recurrent Unit)\n\nThreshold: 100 MBps")
start_btn = st.sidebar.button("🚀 Start Sale Simulation", type="primary")

# Initialize Data
if 'model' not in st.session_state:
    with st.spinner("🧠 Initializing AI Core and pre-training on historical data..."):
        historical_data = generate_synthetic_data(600)
        model, scaler, look_back = build_and_train_model(historical_data)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['look_back'] = look_back
        st.session_state['historical_data'] = historical_data

# Dashboard Layout
# Row 1: Key Metrics
c1, c2, c3, c4 = st.columns(4)
metric_traffic = c1.empty()
metric_pred = c2.empty()
metric_accuracy = c3.empty() # NEW: Accuracy Metric
metric_action = c4.empty()

# Row 2: Advanced Chart
chart_placeholder = st.empty()

if start_btn:
    live_traffic_history = list(st.session_state['historical_data'][-80:].flatten())
    predictions_history = list(st.session_state['historical_data'][-80:].flatten())
    error_history = [] # To store errors for MAE calculation
    
    scaler = st.session_state['scaler']
    model = st.session_state['model']
    look_back = st.session_state['look_back']

    # Simulation Loop
    for t in range(200):
        # 1. GENERATE "LIVE" DATA
        base_val = 50 + 30 * np.sin(0.1 * (600 + t)) + np.random.normal(0, 5)
        
        # SALE LOGIC
        is_sale = t > 30 # Sale starts at step 30
        if is_sale:
            # Gradual ramp up
            spike = base_val * (sale_multiplier - 1) * (1 - np.exp(-(t-30)/15)) 
            current_traffic = base_val + spike
        else:
            current_traffic = base_val

        # 2. PREDICT WITH GRU
        input_seq = np.array(live_traffic_history[-look_back:]).reshape(-1, 1)
        scaled_input = scaler.transform(input_seq)
        scaled_input = scaled_input.reshape(1, look_back, 1)
        
        predicted_scaled = model.predict(scaled_input, verbose=0)
        predicted_traffic = scaler.inverse_transform(predicted_scaled)[0][0]

        # 3. CALCULATE ACCURACY (Absolute Error)
        abs_error = abs(current_traffic - predicted_traffic)
        error_history.append(abs_error)
        mae = np.mean(error_history[-50:]) # Moving average of error

        # 4. AUTO-SCALING LOGIC
        if predicted_traffic > 100:
            metric_action.error(f"⚠️ **SCALE UP**\n\nPred: {int(predicted_traffic)} MBps")
        else:
            metric_action.success(f"✅ **STABLE**\n\nLoad Normal")

        # 5. UPDATE HISTORIES
        live_traffic_history.append(current_traffic)
        predictions_history.append(predicted_traffic)

        if len(live_traffic_history) > 80:
            live_traffic_history.pop(0)
            predictions_history.pop(0)

        # 6. UPDATE DASHBOARD
        
        # Metrics
        metric_traffic.metric("📡 Live Traffic", f"{int(current_traffic)} MBps", delta=f"{int(current_traffic - base_val)}")
        metric_pred.metric("🤖 AI Forecast", f"{int(predicted_traffic)} MBps")
        
        # Color code accuracy (Lower error is better)
        acc_color = "normal" if mae < 15 else "off"
        metric_accuracy.metric("🎯 Model Error (MAE)", f"{mae:.2f}", delta_color=acc_color, help="Mean Absolute Error: Lower is better")

        # Plotly Chart
        fig = go.Figure()
        
        # Actual Traffic Area
        fig.add_trace(go.Scatter(
            y=live_traffic_history,
            mode='lines',
            name='Actual Traffic',
            fill='tozeroy', # Fills area under line
            line=dict(color='#00CC96', width=2),
            fillcolor='rgba(0, 204, 150, 0.2)'
        ))

        # Prediction Line (Dashed)
        fig.add_trace(go.Scatter(
            y=predictions_history,
            mode='lines',
            name='AI Prediction',
            line=dict(color='#EF553B', width=2, dash='dot')
        ))

        # Sale Start Marker
        if is_sale:
            fig.add_vline(x=80 - (t - 30) if t > 30 and (80 - (t-30) > 0) else 0, 
                          line_width=1, line_dash="dash", line_color="yellow", annotation_text="Sale Start")

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            xaxis=dict(showgrid=False, title="Time Steps"),
            yaxis=dict(showgrid=True, gridcolor='#262730', title="Throughput (MBps)"),
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(speed)

    st.success("Simulation Run Complete!")