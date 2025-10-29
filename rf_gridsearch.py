import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('data/crops_clean.csv')
features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop', 'State']
df_dummies = pd.get_dummies(df[features])
X = df_dummies.values
y = df['Yield'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)
print("Test R^2:", best_rf.score(X_test_scaled, y_test))

joblib.dump(best_rf, 'data/best_rf_model.pkl')
joblib.dump(scaler, 'data/rf_scaler.pkl')
df_dummies.columns.to_series().to_csv('data/rf_features_order.csv', index=False)
