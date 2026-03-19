import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "Titanic_Dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

df = pd.read_csv(DATA_PATH)
df = preprocess_data(df)

X = df.drop("Survived", axis=1)
y = df["Survived"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)

print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\n✅ Model Evaluation Completed Successfully")