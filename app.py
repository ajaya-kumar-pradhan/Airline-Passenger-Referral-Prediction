import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and preprocessors
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('airline_svc_model.joblib')
        scaler = joblib.load('scaler.joblib')
        features = joblib.load('features_list.joblib')
        le_airline = joblib.load('airline_encoder.joblib')
        return model, scaler, features, le_airline
    except:
        return None, None, None, None

def main():
    st.set_page_config(
        page_title="Airline Referral Predictor",
        page_icon="✈️",
        layout="centered"
    )

    # UI Variables (Fixed Light Theme)
    bg_main = "#f8faff"
    card_bg = "rgba(255, 255, 255, 0.9)"
    text_color = "#1e293b"
    accent = "#0284c7"
    border_color = "rgba(2, 132, 199, 0.2)"

    # Modern Flight Dashboard CSS
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {{
        background-color: {bg_main};
        font-family: 'Inter', sans-serif;
    }}

    .main-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 50px;
        gap: 20px;
    }}

    .main-header h1 {{
        background: linear-gradient(135deg, {accent} 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: clamp(1.2rem, 4vw, 2.5rem) !important;
        margin: 0 !important;
        letter-spacing: -1px;
        white-space: nowrap;
        text-align: center;
    }}

    .airplane-icon {{
        font-size: clamp(2rem, 6vw, 3.5rem);
        transform: rotate(-10deg);
        filter: drop-shadow(0 10px 15px rgba(2, 132, 199, 0.3));
    }}

    /* Responsive Adjustments */
    @media (max-width: 768px) {{
        .main-header {{
            gap: 15px;
            margin-bottom: 30px;
        }}
    }}

    .stButton>button {{
        width: 100%;
        background: {accent};
        color: white !important;
        border-radius: 12px;
        padding: 15px;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }}

    .stButton>button:hover {{
        background: #0369a1;
        box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.4);
    }}

    /* Hide default Streamlit clutter */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-header">
        <div class="airplane-icon">✈️</div>
        <h1>AIRLINE REFERRAL PREDICTION</h1>
    </div>
    """, unsafe_allow_html=True)

    model, scaler, features, le_airline = load_model_components()
    if model is None:
        st.error("💠 Engine Offline. Please initialize model.")
        return

    # Main Layout: Two columns for desktop, stacks on mobile
    main_left, main_right = st.columns([1.5, 1], gap="large")

    with main_left:
        st.subheader("📋 Flight Parameters")
        
        # Sub-grid for sliders
        grid_1, grid_2 = st.columns(2, gap="medium")
        with grid_1:
            airline_name = st.selectbox("🌐 Airline Carrier", options=sorted(le_airline.classes_), index=0)
            seat_comfort = st.select_slider("💺 Seat Comfort", options=[1,2,3,4,5], value=3)
            cabin_service = st.select_slider("🤝 Crew Service", options=[1,2,3,4,5], value=3)

        with grid_2:
            cabin_class = st.selectbox("💎 Cabin Class", ["Economy", "Business", "First", "Premium Economy"])
            food_bev = st.select_slider("🍽️ Food & Dining", options=[1,2,3,4,5], value=3)
            entertainment = st.select_slider("🎬 Entertainment", options=[1,2,3,4,5], value=3)
            ground_service = st.select_slider("🛂 Ground Service", options=[1,2,3,4,5], value=3)
            
            cabin_map = {"Economy": 0, "Business": 2, "First": 3, "Premium Economy": 1}
            cabin = cabin_map[cabin_class]
            airline_encoded = le_airline.transform([airline_name])[0]

        st.markdown("---")
        trip_type = st.radio("🎯 Travel Type", ["Solo Leisure", "Couple Leisure", "Family Leisure", "Business"], horizontal=True)
        solo_l = 1 if trip_type == "Solo Leisure" else 0
        couple_l = 1 if trip_type == "Couple Leisure" else 0
        family_l = 1 if trip_type == "Family Leisure" else 0

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🚀 PREDICT PASSENGER SENTIMENT")

    with main_right:
        if predict_clicked:
            st.subheader("🔮 Intelligence Result")
            
            input_data = pd.DataFrame([[
                airline_encoded, cabin, seat_comfort, cabin_service, food_bev,
                entertainment, ground_service, couple_l,
                family_l, solo_l
            ]], columns=features)
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)[0][1]
            
            if prediction[0] == 1:
                st.markdown(f"""
                <div style="text-align: center; padding: 25px; background: rgba(16, 185, 129, 0.05); border-radius: 20px; border: 1px solid #10b981; margin-top: 20px;">
                    <div style="font-size: 4rem; margin-bottom: 10px;">✨</div>
                    <h2 style="color: #059669; margin: 0;">PROMOTER</h2>
                    <p style="color: #64748b; font-size: 0.9rem;">High recommendation likelihood.</p>
                    <div style="margin-top: 15px;">
                        <span style="font-size: 0.8rem; color: #64748b; text-transform: uppercase;">Advocacy Score</span>
                        <h1 style="color: #059669; font-size: 3rem !important; margin: 0 !important;">{prob:.1%}</h1>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 25px; background: rgba(239, 68, 68, 0.05); border-radius: 20px; border: 1px solid #ef4444; margin-top: 20px;">
                    <div style="font-size: 4rem; margin-bottom: 10px;">⚠️</div>
                    <h2 style="color: #dc2626; margin: 0;">DETRACTOR</h2>
                    <p style="color: #64748b; font-size: 0.9rem;">Negative feedback risk detected.</p>
                    <div style="margin-top: 15px;">
                        <span style="font-size: 0.8rem; color: #64748b; text-transform: uppercase;">Churn Probability</span>
                        <h1 style="color: #dc2626; font-size: 3rem !important; margin: 0 !important;">{(1-prob):.1%}</h1>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; color: #94a3b8; padding: 80px 20px; opacity: 0.5;">
                <div style="font-size: 4rem; margin-bottom: 20px;">📡</div>
                <p>Awaiting flight parameters...</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; opacity: 0.4; font-size: 0.8rem; margin-top: 20px;">
        Neural Node: Random Forest Classifier | v4.0 Ultra Modern
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
