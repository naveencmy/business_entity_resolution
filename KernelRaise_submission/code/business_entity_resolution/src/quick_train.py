"""
Quick training validation script to test model training and threshold calibration.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from pipeline import train_model

if __name__ == "__main__":
    print("Running quick training check on 500 entities...")
    matcher = train_model(sample_entities=500)
    print("Training test completed successfully!")
