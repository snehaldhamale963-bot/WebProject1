
Assignment Report: Gas Turbines and Forest Fires Prediction
Name : Snehal dhamale
cohort: 135

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load Datasets
gas_data = pd.read_csv('gas_turbines.csv')
fire_data = pd.read_csv('forestfires.csv')

# Preprocessing - Gas Turbines
X_gas = gas_data.drop('TEY', axis=1)
y_gas = gas_data['TEY']

scaler_gas = StandardScaler()
X_gas_scaled = scaler_gas.fit_transform(X_gas)

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_gas_scaled, y_gas, test_size=0.2, random_state=42
)


# Preprocessing - Forest Fires
X_fire = fire_data.drop('area', axis=1)
y_fire = fire_data['area']

# Log-transform target to reduce skewness
y_fire = np.log1p(y_fire)

scaler_fire = StandardScaler()
X_fire_scaled = scaler_fire.fit_transform(X_fire)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fire_scaled, y_fire, test_size=0.2, random_state=42
)


# Linear Regression Models

print("=== Linear Regression Results ===")

# Gas Turbines
lr_gas = LinearRegression()
lr_gas.fit(X_train_g, y_train_g)
y_pred_g = lr_gas.predict(X_test_g)
print("Gas Turbines RMSE:", mean_squared_error(y_test_g, y_pred_g, squared=False))
print("Gas Turbines R2:", r2_score(y_test_g, y_pred_g))

# Forest Fires
lr_fire = LinearRegression()
lr_fire.fit(X_train_f, y_train_f)
y_pred_f = lr_fire.predict(X_test_f)
print("Forest Fires RMSE:", mean_squared_error(y_test_f, y_pred_f, squared=False))
print("Forest Fires R2:", r2_score(y_test_f, y_pred_f))

# -----------------------------
# Neural Network Models
# -----------------------------
print("\n=== Neural Network Results ===")

# Gas Turbines NN
nn_gas = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_g.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
nn_gas.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_g = nn_gas.fit(X_train_g, y_train_g, epochs=50, batch_size=32,
                       validation_split=0.2, verbose=0)

loss_g, mae_g = nn_gas.evaluate(X_test_g, y_test_g, verbose=0)
print("Gas Turbines NN Test MAE:", mae_g)

plt.plot(history_g.history['loss'], label='train_loss')
plt.plot(history_g.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Gas Turbines NN Loss Curve")
plt.show()

# Forest Fires NN
nn_fire = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_f.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
nn_fire.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_f = nn_fire.fit(X_train_f, y_train_f, epochs=50, batch_size=32,
                        validation_split=0.2, verbose=0)

loss_f, mae_f = nn_fire.evaluate(X_test_f, y_test_f, verbose=0)
print("Forest Fires NN Test MAE:", mae_f)

plt.plot(history_f.history['loss'], label='train_loss')
plt.plot(history_f.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Forest Fires NN Loss Curve")
plt.show()

# Results Summary

print("\n=== Summary ===")
print("Gas Turbines: Linear Regression vs Neural Network")
print("Forest Fires: Linear Regression vs Neural Network")
