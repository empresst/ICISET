# experiments/run_delhi_climate.py
"""
Experiment script for Delhi Climate Temperature Prediction Dataset
Used in ICISET 2024 paper
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Override config for this dataset
config.HOUSEHOLD_CSV = "../data/delhi_climate.csv"   # Change filename if different
config.TARGET_COL = "meantemp"                       # Target column name

print("🚀 Running Experiment: Delhi Climate Temperature Prediction")
print("=" * 70)
print(f"Dataset Path : {config.HOUSEHOLD_CSV}")
print(f"Target Column: {config.TARGET_COL}")
print("=" * 70 + "\n")

import src.train as train_module