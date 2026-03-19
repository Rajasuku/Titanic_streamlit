import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "Titanic_Dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
df = preprocess_data(df)

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved successfully")
