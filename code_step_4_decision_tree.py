import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Feature Engineering
df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month

# Define features and target
X = df[["Hour", "Day", "Month"]]
y = df["Vehicles"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

print("Decision Tree Model Trained Successfully!")
print("Mean Absolute Error:", mae)
