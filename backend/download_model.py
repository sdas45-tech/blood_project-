"""
Model downloader script for Vercel deployment.
Downloads the trained model from cloud storage if not present locally.
"""

import os
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "vgg16_best.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# ⚠️ IMPORTANT: Replace these with your actual cloud storage URLs
# Example: Google Drive, AWS S3, GitHub Releases, etc.
MODEL_URL = "https://your-cloud-storage.com/vgg16_best.keras"
CLASS_NAMES_URL = "https://your-cloud-storage.com/class_names.json"


def download_model():
    """Download model if not present locally."""
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"✅ Model found: {MODEL_PATH}")
        return True
    
    print(f"⬇️  Downloading model from cloud storage...")
    
    try:
        # Ensure models directory exists
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download model
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ Model downloaded to {MODEL_PATH}")
        
        # Download class names if available
        if not CLASS_NAMES_PATH.exists():
            urllib.request.urlretrieve(CLASS_NAMES_URL, CLASS_NAMES_PATH)
            print(f"✅ Class names downloaded to {CLASS_NAMES_PATH}")
        
        return True
    
    except Exception as e:
        print(f"❌ ERROR downloading model: {e}")
        print(f"\n⚠️  PLEASE CONFIGURE CLOUD STORAGE:")
        print(f"1. Upload your model to a cloud service (Google Drive, AWS S3, GitHub Releases, etc.)")
        print(f"2. Get the download URL")
        print(f"3. Update MODEL_URL and CLASS_NAMES_URL in this script")
        return False


if __name__ == "__main__":
    success = download_model()
    exit(0 if success else 1)
