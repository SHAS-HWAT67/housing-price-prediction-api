import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import joblib

# Load dataset
df = pd.read_csv("../data/Housing.csv")

# Select features
X = df[
    [
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking"
    ]
]

y = df["price"]

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved")