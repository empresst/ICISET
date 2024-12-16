# Hybrid Model for Time Series Data Prediction

This project introduces a **hybrid model** that combines **BiLSTM**, **BiGRU**, **TCN**, and **XGBRegressor** for advanced time series data prediction. The model has been rigorously tested on 8–10 datasets, showing superior performance in terms of speed and accuracy compared to isolated models and other hybrid approaches like **CNN-GRU** and **GRU-FCN**.

## Features

- **Applications**: Suitable for diverse time series prediction tasks, including-
  - **Load Profile Prediction** (e.g., industrial energy demand).
  - **Stock Value Prediction**.
  - **Temperature Forecasting**.
  - **Recommendation Systems**: Adaptable for platforms like Youtube, TikTok, where sequential user behavior and temporal trends play a critical role in personalized recommendations.

- **Performance**:
  - Demonstrates higher accuracy and faster predictions.
  - Outperforms isolated models and other hybrids in diverse datasets.

- **Evaluation Metrics**:
  - **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)** used to calculate error rates.
  - Visualized performance differences using error plots across datasets.

## Workflow

1. **Data Preprocessing**:
   - Cleaned, normalized, and prepared datasets for training.
   
2. **Model Training and Testing**:
   - Trained and validated the hybrid model on multiple datasets.
   - Demonstrated robust performance in capturing both short-term and long-term dependencies.

## Why This Model?

- **Faster and More Accurate**: Outperforms isolated models and hybrids like BiLSTM, GRU-FCN, CNN-GRU.
- **Adaptability**: Ideal for time series prediction across industries and user recommendations.
- **Comprehensive Testing**: Evaluated on diverse datasets to ensure reliability and efficiency.

## How to Use

1. Prepare your time series dataset (ensure it’s preprocessed).
2. Train the hybrid model using the provided architecture.
3. Evaluate performance using MAE and RMSE, and visualize the results with error plots.

---

This hybrid model sets a benchmark for efficient and accurate time series prediction across a variety of applications.


