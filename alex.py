import numpy as np
import pandas as pd
from sklearn import linear_model

print("n data")
ndata = "n-data.csv"
n_dataset = pd.read_csv(ndata)
X = n_dataset.iloc[:, 1:].values  # Features
y_str = n_dataset.iloc[:, 0].values   # Labels (protein location) - Original string labels
print(f"Dataset shape: {n_dataset.shape}")
print(f"Features shape: {X.shape}")
print(f"Original Labels shape: {y_str.shape}")
print(f"Unique original labels: {np.unique(y_str)}")

print("\ng data")
gdata = "g_data.csv"
g_dataset = pd.read_csv(gdata)
X = g_dataset.iloc[:, 1:].values  # Features
y_str = g_dataset.iloc[:, 0].values   # Labels (protein location) - Original string labels
print(f"Dataset shape: {g_dataset.shape}")
print(f"Features shape: {X.shape}")
print(f"Original Labels shape: {y_str.shape}")
print(f"Unique original labels: {np.unique(y_str)}")

reg = linear_model.LinearRegression()
reg.fit(ndata)