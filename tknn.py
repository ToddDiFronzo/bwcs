"""
KNN works by getting the Euclidian distance from a 'test point' to the 'k closest points' to the 'test point'.
KNN specifies how many neighbors to pick by the letter k

If you set k to 5, the model will find the five closest training points to the test set. 
The test point will be classified as whatever the majority of those 5 points are.

"""
import numpy as np 

class TKNN:
    def __init__(self, k=3):
        self.k = k 

    def fit(self, X, y):
        # store our training samples
        self.X_train = X
        self.y_train = y

    def e_distance(self, x1, x2):
        # calcluate euclidian distance
        return np.sqrt(np.sum((x1-x2)**2))

    def predict(self, X):
        # get label predictions
        label_pred = [self.distance(x) for x in X]
        return np.array(label_pred)

    def distance(self, x):
        # get distances
        # distances = [self.e_distance(x, x_train) for x_train in self.X_train]
        distances = [self.e_distance(x, x_train) for x_train in self.X_train]
        # knn samples/labels
        # print('x:', type(x))
        kn_index = np.argsort(distances)[:self.k]
        # print('kn:', type(kn_index))
        kn_labels = [self.y_train[i] for i in kn_index]

        # choose common labels via majority vote
        most_common = Counter(kn_labels).most_common(1)
        return most_common[0][0]
        

