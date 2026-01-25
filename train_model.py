import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to preprocess the data as found in the notebook
def train_and_save_model(data_path):
    # Load dataset
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Columns we want to keep
    keep_cols = ['airline', 'overall','traveller_type', 'cabin','seat_comfort','cabin_service', 'food_bev',
                'entertainment', 'ground_service','value_for_money', 'recommended']
    
    # Filtering available columns
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols]

    # 1. Handling Missing Values
    from sklearn.impute import SimpleImputer
    numeric_column=['overall', 'seat_comfort', 'cabin_service','food_bev', 'entertainment', 'ground_service', 'value_for_money']
    categorical_column=['airline', 'traveller_type', 'cabin']
    
    # Only process columns that exist
    num_cols_present = [c for c in numeric_column if c in df.columns]
    cat_cols_present = [c for c in categorical_column if c in df.columns]

    if num_cols_present:
        numeric_imputer = SimpleImputer(strategy='mean')
        df[num_cols_present] = numeric_imputer.fit_transform(df[num_cols_present])
    
    if cat_cols_present:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols_present] = categorical_imputer.fit_transform(df[cat_cols_present])

    if 'recommended' in df.columns:
        df.dropna(subset=['recommended'], inplace=True)
        df.drop_duplicates(inplace=True)
    
    # 2. Categorical Encoding
    if 'recommended' in df.columns:
        df['recommended'] = df['recommended'].replace({'yes': 1, 'no': 0}).astype(int)
    
    if 'cabin' in df.columns:
        df['cabin'] = df['cabin'].replace({'Economy Class':0, 'Premium Economy':1, 'Business Class' : 2, 'First Class':3}).astype(int)
    
    # Encode Airline names
    from sklearn.preprocessing import LabelEncoder
    le_airline = LabelEncoder()
    df['airline'] = le_airline.fit_transform(df['airline'].astype(str))
    
    if 'traveller_type' in df.columns:
        ohe = pd.get_dummies(df['traveller_type'], drop_first=True)
        df = pd.concat([df, ohe], axis=1)
        df.drop('traveller_type', axis=1, inplace=True)

    # Convert all columns to numeric to avoid issues
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    # 3. Feature Selection
    if 'value_for_money' in df.columns:
        df.drop('value_for_money', axis=1, inplace=True)
    if 'overall' in df.columns:
        df.drop('overall', axis=1, inplace=True)

    target = 'recommended'
    features = [c for c in df.columns if c != target]
    
    X = df[features]
    y = df[target]
    
    # 4. Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Model Training (Switching to RandomForest for speed/efficiency)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # 6. Save model components
    joblib.dump(model, 'airline_svc_model.joblib') # Keep same name to avoid breaking app.py imports initially
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(features, 'features_list.joblib')
    joblib.dump(le_airline, 'airline_encoder.joblib')
    
    print(f"Success! Model trained on {len(df)} rows.")
    print(f"Features used: {features}")
    print(f"Airlines encoded: {len(le_airline.classes_)}")

if __name__ == "__main__":
    # Look for the Excel file we just moved
    default_file = 'data_airline_reviews.xlsx'
    print(f"Attempting to train model using '{default_file}'...")
    try:
        train_and_save_model(default_file)
    except Exception as e:
        print(f"Error: {e}")
