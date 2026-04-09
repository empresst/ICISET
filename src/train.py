# src/train.py
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data

# Import the 4 model functions
from models.cnn_gru import create_cnn_gru
from models.gru_fcn import create_gru_fcn
from models.l2_regularized_gru import create_l2_regularized_gru
from models.hybrid_ensemble import train_hybrid

if __name__ == "__main__":
    print("Loading data...")
    df = load_data(config.HOUSEHOLD_CSV, data_type="household")

    print("Preprocessing...")
    X_train, Y_train, X_test, Y_test, d_test, scaler, df = preprocess_data(
        df, target_col=config.TARGET_COL, look_back=config.LOOK_BACK)

    print("Training models (using your exact original code)...\n")

    # Train all 4 models using your full original code
    print("→ Training CNN-GRU...")
    pred_cnn = create_cnn_gru((config.LOOK_BACK, 1), X_train, Y_train, X_test, Y_test, scaler)

    print("\n→ Training GRU-FCN...")
    pred_grufcn = create_gru_fcn((1, config.LOOK_BACK), X_train, Y_train, X_test, Y_test, scaler)

    print("\n→ Training L2 Regularised GRU...")
    pred_rgru = create_l2_regularized_gru(X_train.shape[1:], X_train, Y_train, X_test, Y_test, scaler)

    print("\n→ Training Hybrid Ensemble (BiLSTM + BiGRU + TCN + XGBoost)...")
    pred_hybrid = train_hybrid(X_train, Y_train, X_test, Y_test, scaler)

    # Save all results for evaluate.py
    os.makedirs("../results", exist_ok=True)
    np.save("../results/pred_cnn.npy", pred_cnn)
    np.save("../results/pred_grufcn.npy", pred_grufcn)
    np.save("../results/pred_rgru.npy", pred_rgru)
    np.save("../results/pred_hybrid.npy", pred_hybrid)
    np.save("../results/Y_test_inv.npy", scaler.inverse_transform(Y_test.reshape(-1, 1)))
    np.save("../results/d_test.npy", d_test)

    print("\n✅ All models trained successfully! Results saved in ../results/")