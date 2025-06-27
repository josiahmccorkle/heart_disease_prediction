import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# --------------------------------
#  Load Data
# --------------------------------
output_dir = "./outputs"

def load_data():
    filePath = './data/heart.csv'
    if not os.path.exists(filePath):
        raise FileNotFoundError("Missing file: 'heart.csv' in './data/'")
    try:
        df = pd.read_csv(filePath)
        # print(df)
        # print(df.head())
        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e.args[0]} in './data/'")

