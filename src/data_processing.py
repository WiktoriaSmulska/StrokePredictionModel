import os.path
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def read_csv_file(path: str)-> pd.DataFrame:
    return pd.read_csv(path)

def encode(data_copy:pd.DataFrame)-> pd.DataFrame:
    categorical_columns = ['gender', 'ever_married', 'work_type', 'smoking_status', 'Residence_type']
    le = LabelEncoder()

    for col in categorical_columns:
        data_copy[col] = le.fit_transform(data_copy[col])

    return data_copy

def modify_data(data: pd.DataFrame)-> pd.DataFrame:
    data_copy = data.copy()
    data_copy['bmi'] = data['bmi'].fillna(data['bmi'].mean())
    data_copy2 = encode(data_copy)
    return data_copy2

def prepare_data()-> tuple:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "..", "data", "raw", "healthcare-dataset-stroke-data.csv")

    data = read_csv_file(path)
    final_data = modify_data(data)

    stroke = final_data['stroke']
    data_copy = final_data.drop(columns=['id', 'stroke'])

    zmienna_age = data_copy["age"]
    data_copy["age"] = zmienna_age / 100
    data_copy["avg_glucose_level"] = data_copy["avg_glucose_level"] / 1000
    data_copy["bmi"] = data_copy["bmi"] / 100

    return data_copy, stroke


prepare_data()