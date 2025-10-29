import pandas as pd

df = pd.read_csv('data/crops.csv')
# Drop missing values
df_clean = df.dropna()

# Convert numeric columns
for col in ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Save for modeling use
df_clean.to_csv('data/crops_clean.csv', index=False)
print(df_clean.info())
