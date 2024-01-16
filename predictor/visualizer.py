import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Read the dataset
df = pd.read_csv('prices.csv')

# One-hot encode 'Neighborhood'
one_hot_encoded = pd.get_dummies(df['Neighborhood']).astype(int)
df = pd.concat([df, one_hot_encoded], axis=1)
df = df.drop(columns='Neighborhood')

# Visualization 1: Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Split the data
y = df['Price']
X = df.drop(columns='Price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

# Visualization 2: Correlation Matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
