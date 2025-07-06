# 🚀 GHG Emissions Prediction Project - Setup Guide

## 📋 Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning)
- Excel file: `Data_Set.xlsx` (should be in project root)

## 🔧 Step-by-Step Setup

### 1. **Create Virtual Environment**
```bash
# Create virtual environment
python -m venv ghg_env

# Activate virtual environment
# On Windows:
ghg_env\Scripts\activate

# On macOS/Linux:
source ghg_env/bin/activate
```

### 2. **Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. **Verify Dataset**
Ensure `Data_Set.xlsx` is in the project root directory with the following structure:
- Multiple sheets for years 2010-2016
- Each year has `_Detail_Commodity` and `_Detail_Industry` sheets
- Required columns for features and target variable

### 4. **Train the Model**
```bash
# Run the training script
python run_training.py
```

### 5. **Run the Web Application**
```bash
# Navigate to the utils directory
cd "Final Project/utils"

# Run Streamlit app
streamlit run app.py
```

## 🎯 Expected Output

### Model Training Output:
```
🚀 Starting Model Training...
📊 Found 7 commodity sheets
✅ Loaded 1576 rows from 2010_Summary_Commodity
✅ Loaded 1576 rows from 2011_Summary_Commodity
...
📈 Total combined data: 11032 rows
🎯 Training set: 8825 rows, Test set: 2207 rows
✅ Model Training Complete!
📉 Mean Squared Error: 0.000123
📊 R² Score: 0.9998
💾 Models saved successfully:
   - random_forest_model.pkl
   - scaler.pkl
   - feature_columns.pkl
```

### Streamlit App:
- Opens in your default browser (usually http://localhost:8501)
- Interactive form for input parameters
- Real-time emission factor predictions

## 🔍 Troubleshooting

### Common Issues:

1. **"Data_Set.xlsx not found"**
   - Ensure the Excel file is in the project root directory
   - Check file name spelling (case-sensitive)

2. **"No '_Summary_Commodity' sheets found"**
   - Verify Excel file contains sheets ending with `_Summary_Commodity`
   - Check sheet names in Excel

3. **Import errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Streamlit not found**
   - Install streamlit: `pip install streamlit`
   - Or reinstall all dependencies

## 📁 Project Structure After Setup
```
Green_House_Gas_Emissions_Prediction_AICTE-Internship-main/
├── ghg_env/                          # Virtual environment
├── Data_Set.xlsx                     # Dataset file
├── requirements.txt                  # Dependencies
├── run_training.py                   # Training script
├── SETUP_GUIDE.md                    # This file
├── Final Project/
│   ├── models/
│   │   ├── random_forest_model.pkl   # Trained model
│   │   ├── scaler.pkl                # Data scaler
│   │   └── feature_columns.pkl       # Feature names
│   └── utils/
│       ├── app.py                    # Streamlit app
│       └── preprocessor.py           # Data preprocessing
└── README.md
```

## 🎉 Success Indicators
- ✅ Virtual environment activated (see `(ghg_env)` in terminal)
- ✅ All dependencies installed without errors
- ✅ Model training completes with R² > 0.99
- ✅ Streamlit app opens in browser
- ✅ Can make predictions through the web interface

## 🚀 Next Steps
After successful setup:
1. Explore the Jupyter notebooks for detailed analysis
2. Modify the model parameters in `run_training.py`
3. Customize the Streamlit app interface
4. Deploy to cloud platforms if needed 