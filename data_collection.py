import pandas as pd

# Load crop dataset
df = pd.read_csv('data/crops.csv')
print(df.head())  # Preview the first 5 rows
print(df.columns) # Show all available columns
