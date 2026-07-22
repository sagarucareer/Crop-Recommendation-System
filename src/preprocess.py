import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")

# Separate Features and Target
X = df.drop("label", axis=1)
y = df["label"]

print("Features")
print(X.head())

print("\nTarget")
print(y.head())

# Encode Target Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded Target")
print(y_encoded[:10])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape :", X_test.shape)

import joblib

joblib.dump(label_encoder, "artifacts/label_encoder.pkl")