import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Fertilizer Prediction.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Encode categorical columns
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Type"] = crop_encoder.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = fertilizer_encoder.fit_transform(df["Fertilizer Name"])

# Features and Target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save Model and Encoders
joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(soil_encoder, "soil_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")
joblib.dump(fertilizer_encoder, "fertilizer_encoder.pkl")

print("\n✅ Model and Encoders Saved Successfully!")