import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define file paths
data_path = "food_spoilage_data2.csv"  # Ensure the dataset is in the correct location
models_dir = "models"

# Create 'models' directory if not exists
os.makedirs(models_dir, exist_ok=True)

# Load dataset
print("📂 Loading dataset...")
data = pd.read_csv(data_path)

# Display first few rows
print("\n🧐 First 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\n📊 Checking for missing values:")
print(data.isnull().sum())

# Encode categorical target variable (e.g., Spoilage Status: Fresh or Spoiled)
print("\n🔄 Encoding categorical variables...")
label_encoder = LabelEncoder()
data['Spoilage_Status'] = label_encoder.fit_transform(data['Spoilage_Status'])

# Select features & target
X = data.drop(columns=['Spoilage_Status'])  # Features: Moisture, pH, Temp, Humidity, etc.
y = data['Spoilage_Status']  # Target: 1 (Fresh) or 0 (Spoiled)

# Normalize numerical features
print("📏 Scaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training & testing sets
print("\n✂️ Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
print("\n🤖 Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Complete! Accuracy: {accuracy:.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Save model & preprocessing tools
print("\n💾 Saving trained model and preprocessing tools...")
joblib.dump(model, os.path.join(models_dir, "food_spoilage_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))

print("\n🎉 All files saved in the 'models/' folder. Training process completed successfully!")
