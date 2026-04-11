# src/config.py
import os

LOOK_BACK = 30
TRAIN_RATIO = 0.80
BATCH_SIZE = 256
EPOCHS = 20
RANDOM_SEED = 42

DATA_DIR = "../data"
HOUSEHOLD_CSV = os.path.join(DATA_DIR, "eta.csv")   # Change if you use ab.csv
TARGET_COL = "Global_active_power"