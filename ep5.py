# pylint: disable=E1101
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

#import random

class MyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predicitions = []
        for row in X_test:
            label = self.closest(row)
            predicitions.append(label)
        return predicitions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# import a datasest
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()
my_classifier = MyKNN()

my_classifier.fit(X_train, y_train)

predicitions = my_classifier.predict(X_test)
#print(predicitions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predicitions))