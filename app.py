import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb 
from geopy.distance import geodesic
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>

    body {
        background: linear-gradient(135deg, #e0f2ff, #d8ecff);
    }

    .main-header {
        font-size: 3rem;
        color: #134b72;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }

    .sub-header {
        font-size: 1.5rem;
        color: #2b6d8e;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff5252;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* NON-WHITE CARD BACKGROUND (blue tinted) */
    .card {
        background: rgba(30, 80, 120, 0.12);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(5px);
    }

    /* Fraud Alert */
    .fraud-alert {
        background: linear-gradient(135deg, #ff4d4d, #ff7676);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: pulse 2s infinite;
    }

    /* Legit Alert */
    .legit-alert {
        background: linear-gradient(135deg, #2ecc71, #6ee06c);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    /* NON-WHITE METRIC CARDS */
    .metric-card {
        background: rgba(30, 80, 120, 0.18);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-top: 4px solid #1f77b4;
        color: #09324d;
        backdrop-filter: blur(6px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.18);
    }

</style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_model():
    try:
        model = joblib.load("fraud_detection_model.jb")
        encoder = joblib.load("label_encoder.jb")
        return model, encoder
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'fraud_detection_model.jb' and 'label_encoder.jb' are in the correct directory.")
        return None, None

model, encoder = load_model()

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def create_location_map(lat, long, merch_lat, merch_long):
    fig = go.Figure()
    
    # Add transaction location
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[long],
        mode='markers',
        marker=dict(size=20, color='red'),
        name='Transaction Location',
        text=['Transaction Point'],
        hoverinfo='text'
    ))
    
    # Add merchant location
    fig.add_trace(go.Scattermapbox(
        lat=[merch_lat],
        lon=[merch_long],
        mode='markers',
        marker=dict(size=20, color='blue'),
        name='Merchant Location',
        text=['Merchant'],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=(lat + merch_lat)/2, lon=(long + merch_long)/2),
            zoom=10
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    
    return fig

# Header
st.markdown('<div class="main-header">🛡️ Fraud Detection System</div>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">📝 Transaction Details</div>', unsafe_allow_html=True)
    
    # Card-style containers
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        merchant = st.text_input("🏪 Merchant Name", placeholder="Enter merchant name")
        category = st.text_input("📂 Category", placeholder="Enter transaction category")
        amt = st.number_input("💰 Transaction Amount", min_value=0.0, format="%.2f", value=100.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1a, col2a = st.columns(2)
        with col1a:
            lat = st.number_input("📍 Latitude", format="%.6f", value=40.7128)
            long = st.number_input("📍 Longitude", format="%.6f", value=-74.0060)
        with col2a:
            merch_lat = st.number_input("🏪 Merchant Latitude", format="%.6f", value=40.7589)
            merch_long = st.number_input("🏪 Merchant Longitude", format="%.6f", value=-73.9851)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1b, col2b, col3b = st.columns(3)
        with col1b:
            hour = st.slider("🕒 Transaction Hour", 0, 23, 12)
        with col2b:
            day = st.slider("📅 Transaction Day", 1, 31, 15)
        with col3b:
            month = st.slider("🗓️ Transaction Month", 1, 12, 6)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        cc_num = st.text_input("💳 Credit Card Number", placeholder="Enter credit card number")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sub-header">📊 Analysis & Results</div>', unsafe_allow_html=True)
    
    # Calculate distance
    if all(v is not None and v != '' for v in [lat, long, merch_lat, merch_long]):
        distance = haversine(lat, long, merch_lat, merch_long)
        
        # Metrics
        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>📍 Distance</h3>
                <h2>{distance:.2f} km</h2>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <h3>🕒 Time</h3>
                <h2>{hour:02d}:00</h2>
            </div>
            ''', unsafe_allow_html=True)
        with col5:
            st.markdown(f'''
            <div class="metric-card">
                <h3>💰 Amount</h3>
                <h2>${amt:.2f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        # Location Map
        st.markdown("*📍 Transaction Location Map*")
        map_fig = create_location_map(lat, long, merch_lat, merch_long)
        st.plotly_chart(map_fig, use_container_width=True)
    
    # Prediction section
    st.markdown("---")
    
    if st.button("🔍 Check for Fraud", use_container_width=True):
        if merchant and category and cc_num:
            with st.spinner('Analyzing transaction...'):
                # Prepare input data
                input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                        columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])
                
                # Encode categorical variables
                categorical_col = ['merchant', 'category', 'gender']
                for col in categorical_col:
                    try:
                        input_data[col] = encoder[col].transform(input_data[col])
                    except (ValueError, KeyError):
                        input_data[col] = -1
                
                # Hash credit card number
                input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Display results
                st.markdown("---")
                if prediction == 1:
                    st.markdown(f'''
                    <div class="fraud-alert">
                        <h2>🚨 FRAUDULENT TRANSACTION DETECTED</h2>
                        <p>Confidence: {prediction_proba[1]*100:.2f}%</p>
                        <p>⚠️ This transaction has been flagged as suspicious</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="legit-alert">
                        <h2>✅ LEGITIMATE TRANSACTION</h2>
                        <p>Confidence: {prediction_proba[0]*100:.2f}%</p>
                        <p>✓ This transaction appears to be legitimate</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown("*📈 Fraud Probability Meter*")
                fraud_prob = prediction_proba[1] * 100
                st.progress(int(fraud_prob))
                st.write(f"Fraud Probability: {fraud_prob:.2f}%")
                
        else:
            st.error("❌ Please fill all required fields")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🛡️ Fraud Detection System • Secure Transaction Monitoring"
    "</div>",
    unsafe_allow_html=True
)

