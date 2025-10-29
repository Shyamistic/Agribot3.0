import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/crops_clean.csv')

features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
X = df[features]
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
score = model.score(X_test, y_test)
print(f"Model R² score: {score:.2f}")

# Predict for a new input
sample_input = [[10, 5, 700, 150, 25]]  # Change numbers to realistic values
predicted_yield = model.predict(sample_input)
print(f"Predicted yield for test input: {predicted_yield[0]:.2f}")
df = pd.read_csv('data/crops_clean.csv')
df = pd.get_dummies(df, columns=['Crop', 'Season', 'State'])
# Continue with train/test split and model fitting
