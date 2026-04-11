import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
# You can override config here if needed
config.HOUSEHOLD_CSV = "../data/etaysob.csv"     # Change path if needed

print("🚀 Running Experiment: Household Power Consumption Dataset")
print("="*60)

import src.train as train_module
# Just run the main training