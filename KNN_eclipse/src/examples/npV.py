import numpy as np
import scipy as sc
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import manifold, datasets, decomposition, discriminant_analysis
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib import offsetbox

#X_train = np.vstack((pd.read_csv("X_train.csv", header=None).values, pd.read_csv("X_test.csv", header=None).values))
X_test = pd.read_csv("X_test.csv", header=None).values
#Y_train = np.append(pd.read_csv("Y_train.csv", header=None).values.ravel(), pd.read_csv("Y_test.csv", header=None).values.ravel())
Y_test = pd.read_csv("Y_test.csv", header=None).values.ravel()
X_train = pd.read_csv("X_train.csv", header=None).values
Y_train = pd.read_csv("Y_train.csv", header=None).values.ravel()
print(X_train.shape)
X_pca = decomposition.PCA(n_components=2).fit_transform(X_train)
X_pca_t = decomposition.PCA(n_components=2).fit_transform(X_test)

#print(X_train)
#knn = KNeighborsClassifier(n_neighbors = 46)
#svm = SVC(kernel="poly", degree = 1)
#dtree = DecisionTreeClassifier()
#mlp = MLPClassifier(hidden_layer_sizes = (200,))
#nbayes = GaussianNB()

#svm.fit(X_train, Y_train)
#knn.fit(X_train, Y_train)
#dtree.fit(X_train, Y_train)
#nbayes.fit(X_train, Y_train)

#pred = knn.predict(X_test)
#pred = svm.predict(X_test)
#pred = dtree.predict(X_test)
#pred = nbayes.predict(X_test)
#print(pred)

#c = 0
#for p, t in zip(pred, Y_test):


#print(accuracy_score(Y_test, pred))

#neighbors = list(range(1,3))
#degree = list(range(1,10))
#neighbors = list(filter(lambda x : x % 5 != 0, myList))

#print(list(neighbors))
'''
cv_scores = []

for k in neighbors:
	print(k)
	svm = SVC(kernel='poly', degree = k)
	#knn = KNeighborsClassifier(n_neighbors = k)
	scores = cross_val_score(svm, X_pca, Y_train, cv = 10, scoring='accuracy')
	cv_scores.append(scores.mean())

#sc = [0.5539130434782609, 0.5062608695652174, 0.49565217391304345, 0.4882608695652174, 0.4696521739130435, 0.46339130434782605, 0.45582608695652177, 0.4509565217391304]

optimal_k = neighbors[cv_scores.index(max(cv_scores))]

print(optimal_k)

'''
#X_pca = decomposition.PCA(n_components=2).fit_transform(X)
#embedding_plot(X_pca, "PCA")
#plt.show()
'''
'''
#plt.plot(neighbors, cv_scores)
#plt.show()