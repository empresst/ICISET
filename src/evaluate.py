# src/evaluate.py
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.metrics import calculate_error_table
from utils.visualization import plot_predictions

if __name__ == "__main__":
    print("Loading saved predictions...")

    Y_test_inv = np.load("../results/Y_test_inv.npy")
    d_test = np.load("../results/d_test.npy", allow_pickle=True)

    pred_cnn    = np.load("../results/pred_cnn.npy")
    pred_grufcn = np.load("../results/pred_grufcn.npy")
    pred_rgru   = np.load("../results/pred_rgru.npy")
    pred_hybrid = np.load("../results/pred_hybrid.npy")

    print("Calculating error percentages...")

    error_cnn    = calculate_error_table(Y_test_inv, pred_cnn, d_test)
    error_grufcn = calculate_error_table(Y_test_inv, pred_grufcn, d_test)
    error_rgru   = calculate_error_table(Y_test_inv, pred_rgru, d_test)
    error_hybrid = calculate_error_table(Y_test_inv, pred_hybrid, d_test)

    percentage_df = pd.DataFrame({
        "Date": list(error_hybrid.keys()),
        "CNN-GRU(%)": list(error_cnn.values()),
        "GRUFCN(%)": list(error_grufcn.values()),
        "R.GRU(%)": list(error_rgru.values()),
        "Hybrid(%)": list(error_hybrid.values())
    })

    print("\n" + "="*80)
    print(percentage_df.tail(40))
    print("="*80)

    avg_errors = percentage_df.mean(numeric_only=True)
    best_method = avg_errors.idxmin()
    lowest_error = avg_errors.min()

    print(f"\nOverall Best Method: {best_method} (Average Error: {lowest_error:.4f}%)")

    # Plot predictions (exactly like your notebook)
    print("\nGenerating prediction plot...")
    plot_predictions(Y_test_inv, {
        "CNN-GRU": pred_cnn,
        "GRUFCN": pred_grufcn,
        "R.GRU": pred_rgru,
        "Hybrid": pred_hybrid
    })

    print("\n✅ Evaluation completed!")