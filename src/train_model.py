import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")

# Features & Target
X = df.drop("label", axis=1)
y = df["label"]

# Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Calculate classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate confusion_matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Save the Model
joblib.dump(model, "artifacts/crop_recommendation_model.pkl")
print("\nModel saved successfully!")