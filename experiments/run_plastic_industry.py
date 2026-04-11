# experiments/run_plastic_industry.py
"""
Experiment script for Plastic Industry Load Consumption Dataset
Used in ICISET 2024 paper
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Override config for this dataset
config.HOUSEHOLD_CSV = "../data/plastic_industry.csv"
config.TARGET_COL = "load"          # Change if your column name is different

print("🚀 Running Experiment: Plastic Industry Load Consumption")
print("=" * 70)
print(f"Dataset Path : {config.HOUSEHOLD_CSV}")
print(f"Target Column: {config.TARGET_COL}")
print("=" * 70 + "\n")

import src.train as train_module