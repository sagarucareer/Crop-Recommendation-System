import joblib
import pandas as pd

# Load trained model
model = joblib.load("artifacts/crop_recommendation_model.pkl")

# Load label encoder
label_encoder = joblib.load("artifacts/label_encoder.pkl")

#Take User Input
N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH: "))
rainfall = float(input("Enter Rainfall: "))

#Convert Input into a DataFrame
input_data = pd.DataFrame(
    [[N, P, K, temperature, humidity, ph, rainfall]],
    columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
)

#Predict
prediction = model.predict(input_data)

#Decode
crop = label_encoder.inverse_transform(prediction)

print("\n==============================")
print(f"🌱 Recommended Crop: {crop[0].title()}")
print("==============================")