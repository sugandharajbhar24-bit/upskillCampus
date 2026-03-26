import pandas as pd

# load dataset
df = pd.read_csv("dataset.csv")

# convert DateTime column
df["DateTime"] = pd.to_datetime(df["DateTime"])

# create new features
df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month

print("Feature engineering completed!")
print(df.head())
