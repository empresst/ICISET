# src/utils/metrics.py
import numpy as np

def calculate_error_table(Y_test, test_predict, dates):
    """Exact function from your notebook"""
    error_table = {}
    date_count = {}
    for date, actual, predicted in zip(dates, Y_test, test_predict):
        absolute_error = np.abs(actual - predicted[0])
        error = (absolute_error / actual) * 100 if actual > 0 else 0
        if date not in date_count:
            date_count[date] = 0
            error_table[date] = 0
        date_count[date] += 1
        error_table[date] += error
    for date in error_table:
        error_table[date] /= date_count[date]
    return error_table