import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# --- Load your cleaned dataset and retrain the model (you can also load a saved model.pkl with joblib.load) ---
df = pd.read_csv('data/crops_clean.csv')
features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
X = df[features]
y = df['Yield']

model = LinearRegression()
model.fit(X, y)  # Trained on entire dataset for demo; in production use train/test split and save/load model

# --- Advice function ---
def generate_advice(predicted_yield, rainfall, fertilizer):
    advice = []
    if predicted_yield < 1:
        advice.append("Low yield predicted. Consider additional fertilizers or switch crop variety.")
    if rainfall < 500:
        advice.append("Low rainfall expected. Use water-saving irrigation.")
    if fertilizer < 50:
        advice.append("Fertilizer input low. Review soil and crop needs.")
    if not advice:
        advice.append("Your inputs suggest decent yield. Continue regular monitoring and good practices.")
    return advice

# --- Main User Interface ---
def main():
    print("=== Agribot: Smart Yield & Advice ===")
    try:
        area = float(input("Enter Area (hectares): "))
        production = float(input("Enter Production (tons): "))
        rainfall = float(input("Enter Annual Rainfall (mm): "))
        fertilizer = float(input("Enter Fertilizer used (kg): "))
        pesticide = float(input("Enter Pesticide used (kg): "))
    except ValueError:
        print("Please enter valid numeric values.")
        return

    # Pack input as DataFrame to avoid sklearn warning
    sample_input = pd.DataFrame([[area, production, rainfall, fertilizer, pesticide]], columns=features)
    predicted_yield = model.predict(sample_input)[0]

    print(f"\nPredicted Yield: {predicted_yield:.2f} tons/hectare")

    advice = generate_advice(predicted_yield, rainfall, fertilizer)
    print("\nRecommended Actions:")
    for tip in advice:
        print("-", tip)

if __name__ == "__main__":
    main()
