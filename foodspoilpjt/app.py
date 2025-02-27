import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model and preprocessing tools safely
MODEL_PATH = "models/food_spoilage_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    st.error("🔴 Model files not found! Ensure they are in the correct path before running the app.")
    st.stop()

# Streamlit Title
st.title("🍏 Food Spoilage Detection App")

# Sidebar Inputs
st.sidebar.header("🛠 Enter Food Quality Parameters")
moisture = st.sidebar.slider("Moisture Level", 0.0, 100.0, 50.0)
pH = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
temperature = st.sidebar.slider("Storage Temperature (°C)", -10.0, 50.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
bacteria = st.sidebar.slider("Bacterial Growth Level", 0.0, 10.0, 1.0)

# Function to process input data
def predict_spoilage(input_data):
    """Preprocess input data, predict using trained model, and return decoded results."""
    try:
        input_scaled = scaler.transform(input_data)
        predictions = model.predict(input_scaled)
        return label_encoder.inverse_transform(predictions)
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return None

# File Upload Section
st.sidebar.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data Preview:")
    st.dataframe(data.head())

    # Ensure required columns exist
    required_cols = ['Moisture', 'pH', 'Temperature', 'Humidity', 'Bacterial_Growth']
    if all(col in data.columns for col in required_cols):
        predictions = predict_spoilage(data[required_cols])

        if predictions is not None and len(predictions) > 0:
            data["Spoilage_Status"] = predictions

            # Show results
            st.write("### ✅ Prediction Results:")
            st.dataframe(data)

            # Pie chart for spoilage distribution
            pie_chart = px.pie(
                names=data["Spoilage_Status"].value_counts().index,
                values=data["Spoilage_Status"].value_counts().values,
                title="Food Spoilage Distribution",
            )
            st.plotly_chart(pie_chart)

            # Boxplot visualization for food conditions
            st.write("### 📊 Food Quality Parameters")
            fig = px.box(data[required_cols], title="Parameter Distribution")
            st.plotly_chart(fig)
        else:
            st.error("⚠️ No predictions generated. Check the model or input data.")
    else:
        st.error(f"⚠️ CSV must contain columns: {required_cols}")

# Manual Input Prediction
if st.sidebar.button("🔍 Check Spoilage"):
    input_data = pd.DataFrame({
        'Moisture': [moisture],
        'pH': [pH],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Bacterial_Growth': [bacteria]
    })
    
    predicted_status = predict_spoilage(input_data)

    if predicted_status is not None and len(predicted_status) > 0:
        st.subheader("🍏 Prediction Result")
        result = predicted_status[0]  # Get first (and only) prediction

        # Display textual result
        if result == "Partially Spoiled":
            status_display = "⚠️ Partially Spoiled"
        elif result == "Fresh":
            status_display = "🍏 Fresh"
        else:
            status_display = "❌ Completely Spoiled"
        
        st.success(f"📝 Spoilage Status: **{status_display}**")

        # Generate a bar chart for input metrics
        fig = px.bar(
            x=["Moisture", "pH", "Temperature", "Humidity", "Bacterial_Growth"],
            y=[moisture, pH, temperature, humidity, bacteria],
            labels={"x": "Factors", "y": "Values"},
            title="🔬 Food Quality Metrics",
        )
        st.plotly_chart(fig)
    else:
        st.error("⚠️ Prediction failed. Check the input values and try again.")
