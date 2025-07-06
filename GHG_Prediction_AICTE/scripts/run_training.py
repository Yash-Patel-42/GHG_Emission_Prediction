#!/usr/bin/env python3
"""
Model Training Script for GHG Emissions Prediction
"""

import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model():
    """Train the Random Forest model for GHG emissions prediction"""
    
    print("🚀 Starting Model Training...")
    
    # Check if dataset exists
    if not os.path.exists("data/Data_Set.xlsx"):
        print("❌ Error: Data_Set.xlsx not found in data/ directory!")
        print("Please ensure the dataset file is in the data/ folder.")
        return False
    
    try:
        # Load Excel file
        excel_path = "data/Data_Set.xlsx"
        excel_file = pd.ExcelFile(excel_path)
        
        # Select only '_Summary_Commodity' sheets
        commodity_sheets = [sheet for sheet in excel_file.sheet_names 
                          if sheet.endswith('_Summary_Commodity')]
        
        if not commodity_sheets:
            print("❌ Error: No '_Summary_Commodity' sheets found in the dataset!")
            return False
        
        print(f"📊 Found {len(commodity_sheets)} commodity sheets")
        
        # Feature columns (X) and Target column (y)
        feature_cols = [
            'Supply Chain Emission Factors without Margins',
            'Margins of Supply Chain Emission Factors',
            'DQ ReliabilityScore of Factors without Margins',
            'DQ TemporalCorrelation of Factors without Margins',
            'DQ GeographicalCorrelation of Factors without Margins',
            'DQ TechnologicalCorrelation of Factors without Margins',
            'DQ DataCollection of Factors without Margins'
        ]
        target_col = 'Supply Chain Emission Factors with Margins'
        
        # Load and combine all sheets
        dataframes = []
        for sheet in commodity_sheets:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            if all(col in df.columns for col in feature_cols + [target_col]):
                df = df[feature_cols + [target_col]].dropna()
                dataframes.append(df)
                print(f"✅ Loaded {len(df)} rows from {sheet}")
        
        if not dataframes:
            print("❌ Error: No valid data found in the sheets!")
            return False
        
        # Combine data from all sheets
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"📈 Total combined data: {len(combined_df)} rows")
        
        # Split data
        X = combined_df[feature_cols]
        y = combined_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"🎯 Training set: {len(X_train)} rows, Test set: {len(X_test)} rows")
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model Training Complete!")
        print(f"📉 Mean Squared Error: {mse:.6f}")
        print(f"📊 R² Score: {r2:.4f}")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save model and scaler
        joblib.dump(model, 'models/random_forest_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(feature_cols, 'models/feature_columns.pkl')
        
        print("💾 Models saved successfully:")
        print("   - random_forest_model.pkl")
        print("   - scaler.pkl")
        print("   - feature_columns.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        print("\n🎉 Model training completed successfully!")
        print("You can now run the Streamlit app with: python scripts/run_app.py")
    else:
        print("\n💥 Model training failed. Please check the error messages above.") 