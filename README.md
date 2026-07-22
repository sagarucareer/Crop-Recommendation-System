# 🌱 Crop Recommendation System

An end-to-end Machine Learning project that recommends the most suitable crop based on soil nutrients and environmental conditions using a Random Forest Classifier.

---

## Features

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Label Encoding
- Random Forest Classifier
- Model Evaluation
- Saved ML Model using Joblib
- Interactive Prediction Script

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Git
- GitHub

---

## Dataset

- 2200 samples
- 22 crop classes
- 7 input features
- No missing values

---

## Accuracy

**99.32%**

---

## How to Run

```bash
pip install -r requirements.txt

python src/train_model.py

python src/predict.py
```

---

## Sample Output

```text
Enter Nitrogen (N): 90
Enter Phosphorus (P): 42
Enter Potassium (K): 43
Enter Temperature: 20.879744
Enter Humidity: 82.002744
Enter pH: 6.502985
Enter Rainfall: 202.935536

🌱 Recommended Crop: Rice
```