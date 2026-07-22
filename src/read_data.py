import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")

print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("Dataset Shape")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("Column Names")
print("=" * 50)
print(df.columns)

print("\n" + "=" * 50)
print("Data Types")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("Missing Values")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Statistical Summary")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("Crop Distribution")
print("=" * 50)
print(df["label"].value_counts())