import pandas as pd 
import numpy as np 

df1 = pd.read_csv(r'C:/Users/Todd\Desktop/python_learn/python_data_analytics/baseball.csv')
df2 = pd.read_csv(r'C:/Users/Todd\Desktop/python_learn/python_data_analytics/basketball_f.csv')
df3 = pd.read_csv(r'C:/Users/Todd\Desktop/python_learn/python_data_analytics/basketball.csv')
df4 = pd.read_csv(r'C:/Users/Todd\Desktop/python_learn/python_data_analytics/football.csv')
df5 = pd.read_csv(r'C:/Users/Todd\Desktop/python_learn/python_data_analytics/hockey.csv')
df6 = pd.read_csv(r'C:/Users/Todd\Desktop/python_learn/python_data_analytics/tennis.csv')
df7 = pd.read_csv(r'C:/Users/Todd\Desktop/python_learn/python_data_analytics/volleyball.csv')

print(df1.head())

df = pd.concat([df1,df2,df3,df4,df5,df6,df7])
df.set_index('id', inplace=True)

print(df.head())
print(df.shape)

df.to_csv('your_sport.csv')