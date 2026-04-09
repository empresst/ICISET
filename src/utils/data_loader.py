# # src/data_loader.py
# import pandas as pd

# def load_data(filepath: str, data_type: str = "household"):
#     """Exactly as you did in both notebooks"""
#     if data_type == "household":
#         data = pd.read_csv(filepath)
#         data = data.iloc[:, 1:] if len(data.columns) > 0 and 'Unnamed: 0' in data.columns[0] else data
#     elif data_type == "b_txt":
#         data = pd.read_csv(filepath, delimiter=';')
#     else:
#         raise ValueError("data_type must be 'household' or 'b_txt'")
#     return data

# src/utils/data_loader.py
import pandas as pd

def load_data(filepath: str, data_type: str = "household"):
    """Exact loading code from both notebooks"""
    if data_type == "household":
        data = pd.read_csv(filepath)
        data = data.iloc[:, 1:] if len(data.columns) > 0 and data.columns[0].startswith('Unnamed') else data
        return data
    elif data_type == "b_txt":
        data = pd.read_csv(filepath, delimiter=';')
        return data
    else:
        raise ValueError("data_type must be 'household' or 'b_txt'")