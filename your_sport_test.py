# import necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv(r'C:/Users/Todd/Desktop/bw_prac/datasets/your_sport.csv')
# df.set_index('id', inplace=True)

# create gender column with male=1 and female=0
gender = []
for i in df.index:
    if df["gender"][i].lower() == "male":
        gender.append(1)
    else:
        gender.append(0)
df["gender"] = gender

# sport = []
# for i in df.index:
#     if df['sport'][i].lower() == 'football':
#         sport.append(1)
#     elif df['sport'][i].lower() == 'baseball':
#         sport.append(2)
#     elif df['sport'][i].lower() == 'hockey':
#         sport.append(3)
#     elif df['sport'][i].lower() == 'basketball':
#         sport.append(4)
#     elif df['sport'][i].lower() == 'tennis':
#         sport.append(5)
#     else:
#         sport.append(6)

# df['sport'] = sport
    

# Arrange X and y vectors
X = df[['weight', 'height', 'gender']]   
y = df['sport']

# Split train into train/val sets
X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42)#, stratify=df['sport'],
print(X_train.shape, X_test.shape,y_train.shape, y_test.shape, '\n')

# baseline
print(df['sport'].value_counts(normalize=True))

# normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train, X_test)
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)


# function to calculate euclidian distance of two vectors
X_train_scaled = np.array(X_train_scaled)
y_train = np.array(y_train)


# from scratch TKNN 
# from tknn import TKNN
# model = TKNN(k=3)

# from sklearn KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# # predictions
# predictions = clf.predict(X_test_scaled)
y_pred = model.predict(X_test_scaled)

# # calculate accuracy
# acc = np.sum(predictions==y_test) / len(y_test)     # divide by number of test samples
# print('From scratch: ', acc)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(result,'\n')

result1 = classification_report(y_test, y_pred)
print("Classification Report: ")
print(result1, '\n')

result2 = accuracy_score(y_test, y_pred)
print('Accuracy: ', result2)

print(X_train_scaled.shape, y_train.shape,'\n')

# # test case
# print(model.predict([[90,54, 1]]))
# # print(model.predict([[98,100,0]]))

# print(X_train.head())