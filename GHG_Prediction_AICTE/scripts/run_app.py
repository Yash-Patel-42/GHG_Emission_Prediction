#!/usr/bin/env python3
"""
Script to run the GHG Emissions Prediction Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def run_streamlit_app():
    """Run the Streamlit app"""
    
    # Check if models exist
    models_dir = Path("models")
    required_files = [
        "random_forest_model.pkl",
        "scaler.pkl", 
        "feature_columns.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🔧 Please run the training script first:")
        print("   python run_training.py")
        return False
    
    # Check if app.py exists
    app_path = Path("src/app.py")
    if not app_path.exists():
        print("❌ Streamlit app not found at: src/app.py")
        return False
    
    print("🚀 Starting Streamlit app...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the app")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path), "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_streamlit_app()
    if not success:
        sys.exit(1) 