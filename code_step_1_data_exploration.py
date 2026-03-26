import pandas as pd

# load dataset (change filename if needed)
df = pd.read_csv("dataset.csv")

print("Data loaded successfully!")
print(df.head())   # shows first 5 rows
print("\nColumn names:")
print(df.columns)
import matplotlib.pyplot as plt

# Convert DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Select data for Junction 1
junction1 = df[df['Junction'] == 1]

# Plot traffic for Junction 1
plt.figure()
plt.plot(junction1['DateTime'], junction1['Vehicles'])
plt.xlabel("Time")
plt.ylabel("Number of Vehicles")
plt.title("Traffic Pattern at Junction 1")
plt.show()
# Convert DateTime to proper format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Filter one week data
one_week = df[(df['DateTime'] >= '2015-11-01') & (df['DateTime'] <= '2015-11-07')]

print(one_week.head())
