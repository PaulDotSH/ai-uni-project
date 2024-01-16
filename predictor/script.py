import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('prices.csv')

# One-hot encode 'Neighborhood'
# One-hot encoding re-encodes the dataset to replace different values with numbers, in this case we are
# adding columns representing each type of neighborhood, 0 means it's not that type and 1 that it is
one_hot_encoded = pd.get_dummies(df['Neighborhood']).astype(int)
df = pd.concat([df, one_hot_encoded], axis=1)
df = df.drop(columns='Neighborhood')

# Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=100, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Split the data
y = df['Price']
X = df.drop(columns='Price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

# Correlation Matrix between params
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".5f")
plt.title('Correlation Matrix')
plt.show()

# Using a bigger n_estimators is a bad trade-off, increasing it increases the processing needed by a lot
# but only slightly increases the performance of the Random Forest
rf_reg = RandomForestRegressor(n_estimators=125, random_state=420)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# Residual Plot for Random Forest
residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=residuals_rf, color='blue')
plt.title('Residual Plot - Random Forest')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.show()

# Feature Importance Plot for Random Forest
feature_importance = pd.Series(rf_reg.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()

rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("Random Forest RMSE:", rmse_rf)

# Actual vs Predicted Prices for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='blue')
plt.title('Actual vs Predicted Prices - Random Forest')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Linear Regression
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
y_pred_lr = lr_reg.predict(X_test)

# Residual Plot for Linear Regression
residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=residuals_lr, color='red')
plt.title('Residual Plot - Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.show()

rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
print("Linear Regression RMSE:", rmse_lr)

# Actual vs Predicted Prices for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='red')
plt.title('Actual vs Predicted Prices - Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# KNN Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
y_pred_knn = knn_reg.predict(X_test)

# Residual Plot for KNN
residuals_knn = y_test - y_pred_knn
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=residuals_knn, color='green')
plt.title('Residual Plot - KNN')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.show()

rmse_knn = mean_squared_error(y_test, y_pred_knn, squared=False)
print("KNN RMSE:", rmse_knn)

# Actual vs Predicted Prices for KNN
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, color='green')
plt.title('Actual vs Predicted Prices - KNN')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Model Comparison
rmse_values = [rmse_rf, rmse_lr, rmse_knn]
models = ['Random Forest', 'Linear Regression', 'KNN']

plt.figure(figsize=(10, 6))
sns.barplot(x=rmse_values, y=models, palette='pastel')
plt.title('Model Comparison - RMSE')
plt.xlabel('Root Mean Squared Error (RMSE) (lower is better)')
plt.ylabel('Models')
plt.show()

def calculate_accuracy(y_true, y_pred, tolerance_percentage=2):
    lower_bound = (100 - tolerance_percentage) / 100
    upper_bound = (100 + tolerance_percentage) / 100

    # Check if the predicted values are within the tolerance range of the true values
    correct_predictions = np.logical_and(y_pred >= y_true * lower_bound, y_pred <= y_true * upper_bound)

    # Calculate accuracy
    accuracy = np.sum(correct_predictions) / len(y_true)
    return accuracy

accuracy_rf = calculate_accuracy(y_test, y_pred_rf)
accuracy_lr = calculate_accuracy(y_test, y_pred_lr)
accuracy_knn = calculate_accuracy(y_test, y_pred_knn)

print("Random Forest Accuracy:", accuracy_rf)
print("Linear Regression Accuracy:", accuracy_lr)
print("KNN Accuracy:", accuracy_knn)
