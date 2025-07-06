@echo off
echo 🚀 GHG Emissions Prediction Project - Quick Start
echo ================================================

REM Check if virtual environment exists
if not exist "ghg_env" (
    echo 📦 Creating virtual environment...
    python -m venv ghg_env
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call ghg_env\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Check if dataset exists
if not exist "data\Data_Set.xlsx" (
    echo ❌ Data_Set.xlsx not found in data directory!
    echo Please ensure the dataset file is in the data folder.
    pause
    exit /b 1
)

REM Train the model
echo 🎯 Training the model...
python run_training.py
if errorlevel 1 (
    echo ❌ Model training failed
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo 🌐 To run the web application:
echo    python scripts\run_app.py
echo.
echo 📖 For detailed instructions, see SETUP_GUIDE.md
echo.
pause 