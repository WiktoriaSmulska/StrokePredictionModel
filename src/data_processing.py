from pandas.core.interchange.from_dataframe import categorical_column_to_series
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
    data = read_csv_file(r'C:\Users\wikto\Desktop\StrokePredictionModel\data\raw\healthcare-dataset-stroke-data.csv')
    final_data = modify_data(data)

    stroke = final_data['stroke']
    data_copy = final_data.drop(columns=['id', 'stroke'])

    #print(data_copy.head())
    #print(stroke)
    #print(data_copy['stroke'].value_counts())
    #print(data_copy.count(axis = 1))

    return data_copy, stroke


prepare_data()