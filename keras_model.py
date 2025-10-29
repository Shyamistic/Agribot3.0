import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/crops_clean.csv')

# One-hot encode categorical columns
features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop', 'State']
df_dummies = pd.get_dummies(df[features])

X = df_dummies.values
y = df['Yield'].values

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_scaled, y_train, epochs=25, batch_size=32, validation_data=(X_test_scaled, y_test))

# Save model and scaler
model.save('data/yield_predictor_categorical.keras')
import joblib
joblib.dump(scaler, 'data/yield_scaler_categorical.pkl')

# Save the column order for prediction encoding
df_dummies.columns.to_series().to_csv('data/yield_features_order.csv', index=False)
