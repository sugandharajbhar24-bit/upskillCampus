import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("dataset.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

print(data.columns)  # just to verify

# Convert datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Feature Engineering
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month

# Features and target
X = data[['junction', 'hour', 'day', 'month']]
y = data['vehicles']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

print("Model trained successfully!")
print("Mean Absolute Error:", mae)
