import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
np.set_printoptions(threshold=np.nan)

X_train = np.genfromtxt("X_train.csv",delimiter=',')
X_test = np.genfromtxt("X_test.csv",delimiter=',')
y_train = np.genfromtxt("y_train.csv",delimiter=',')
y_test = np.genfromtxt("y_test.csv",delimiter=',')
X_train, y_train = shuffle(X_train, y_train, random_state=2)
knn = 20


for weights in ['uniform', 'distance']:
	#we create an instance of Neighbors Classifier and fit the data.
	
	clf = KNeighborsClassifier(knn, weights=weights)
	clf.fit(X_train, y_train)
	Z = clf.score(X_test,y_test)
	print(Z)
	
# bClf = BaggingClassifier(KNeighborsClassifier(knn,weights='uniform'),
# 	n_estimators=10,max_samples=0.7,max_features=0.7)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# bClf.fit(X_train,y_train)
# print(bClf.score(X_test,y_test))
