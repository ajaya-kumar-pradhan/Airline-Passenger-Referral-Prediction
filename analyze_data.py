import pandas as pd
import os

data_path = r"C:\Users\ajaya\.gemini\antigravity\scratch\car-pricing-analytics\data\enterprise_car_pricing_500k.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("Dataset Information:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
else:
    print(f"File not found: {data_path}")
