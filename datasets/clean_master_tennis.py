import numpy as np
import pandas as pd 
import csv 
# import matplotlib.pyplot as plt


df = pd.read_csv(r'C:/Users/Todd/Desktop/python_learn/python_data_analytics/buildweek/datasets/Master_tennis.csv')

print(df.head())

print(df.columns)

df1 = (df[['weight', 'height', 'gender','sport']].copy())
print(df1.tail())

print(df1.shape)


df1['sport'].fillna('Tennis', inplace=True)
print(df1.tail())

print(df1.columns)

df1.insert(0, 'id', range(1,1+len(df1)))
print(df1.head(66))

df1.set_index('id', inplace=True)
print(df1.head())


print(df1.isna().sum())

df1 = df1.dropna()
print(df1.shape)
print(df1.isna().sum())

df1 = df1.round(1)

print(df1.head(20))

print(df1.describe())

df1.to_csv('tennis.csv')