import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer   

dataset = pd.read_csv("stroke.csv")

dataset1 = dataset[dataset.isnull().any(axis=1)]

dataset = dataset.dropna()

x = dataset.iloc[:, 1: -1]
y = dataset.iloc[:, -1]




cols = ['ever_married', 'Residence_type']
le = LabelEncoder()
x[cols] = x[cols].apply(le.fit_transform)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['gender', 'work_type', 'smoking_status'])],
                       remainder='passthrough')
x = ct.fit_transform(x)

print(x)