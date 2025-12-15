import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Calorie Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & GENERATION ---
@st.cache_data
def load_data():
    # Since we don't have the CSV file from the screenshots, 
    # we generate synthetic data matching the stats in the OCR (Page 1)
    # Range: 20 entries
    
    np.random.seed(42)
    days = np.arange(1, 21) # 1 to 20
    
    # Generate minutes (roughly increasing with noise, range 10-105)
    minutes = np.linspace(10, 105, 20) + np.random.normal(0, 5, 20)
    minutes = minutes.astype(int)
    
    # Generate calories (highly correlated with minutes, range 65-645)
    # Formula approx based on visual: Cals ~= 6 * Minutes + noise
    calories = (minutes * 6) + np.random.normal(0, 20, 20)
    calories = calories.astype(int)
    
    df = pd.DataFrame({
        'day': days,
        'minutes_exercised': minutes,
        'calories_burned': calories
    })
    return df

# --- MODEL TRAINING ---
def train_model(df):
    X = df[['day', 'minutes_exercised']]
    y = df['calories_burned']
    
    lr = LinearRegression()
    lr.fit(X, y)
    return lr

# --- MAIN APP LOGIC ---
def main():
    # Load and Train
    df = load_data()
    model = train_model(df)

    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=100)
        st.title("Input Parameters")
        st.write("Adjust the values below to predict calorie burn.")
        
        st.divider()
        
        # Inputs matching the notebook logic
        input_day = st.slider("Day of Training", min_value=1, max_value=365, value=10, help="Which day of your routine is this?")
        input_minutes = st.slider("Duration (Minutes)", min_value=5, max_value=180, value=45, help="How long did you exercise?")
        
        st.caption("Based on Linear Regression Model")

    # --- MAIN CONTENT AREA ---
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔥 Exercise Calorie Predictor")
        st.markdown("### Estimate your energy expenditure based on workout duration.")
    
    st.divider()

    # Prediction Logic
    # We use a dataframe for prediction to avoid the UserWarning seen in Page 2 of your PDF
    input_data = pd.DataFrame([[input_day, input_minutes]], columns=['day', 'minutes_exercised'])
    prediction = model.predict(input_data)[0]

    # Display Results in a nice layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.info(f"**Selected Day:** {input_day}")
    with c2:
        st.info(f"**Duration:** {input_minutes} mins")
    with c3:
        # The result styling
        st.metric(label="Predicted Calories Burned", value=f"{prediction:.2f} kcal", delta="Estimated")

    st.divider()

    # --- VISUALIZATION SECTION ---
    st.subheader("📊 Data Visualization & Model Insight")
    
    tab1, tab2 = st.tabs(["Regression Plot", "Raw Data"])
    
    with tab1:
        col_graph, col_info = st.columns([2, 1])
        
        with col_graph:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot of historical data
            sns.scatterplot(data=df, x='minutes_exercised', y='calories_burned', color='blue', s=100, label='Historical Data', ax=ax)
            
            # Regression line
            sns.regplot(data=df, x='minutes_exercised', y='calories_burned', scatter=False, color='red', ax=ax)
            
            # The User's specific prediction point
            ax.scatter(input_minutes, prediction, color='#00ff00', s=200, edgecolors='black', label='Your Prediction', zorder=5)
            
            ax.set_title('Minutes Exercised vs Calories Burned', fontsize=15)
            ax.set_xlabel('Minutes Exercised', fontsize=12)
            ax.set_ylabel('Calories Burned', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            
        with col_info:
            st.write("### How it works")
            st.write("""
            The red line represents the Linear Regression trend developed from the dataset.
            
            - **Blue Dots:** Past training sessions.
            - **Green Dot:** Your current prediction based on the sidebar inputs.
            """)
            st.success(f"Model Coefficient (Slope): {model.coef_[1]:.2f} cal/min")

    with tab2:
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
