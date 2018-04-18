import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, discriminant_analysis

def embedding_plot(X, title):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
 
    plt.figure()
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c=y/10.)
 
    shown_images = np.array([[1., 1.]])
    # for i in range(X.shape[0]):
    #     if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 1e-2: continue
    #     shown_images = np.r_[shown_images, [X[i]]]
    #     ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]))
 
    plt.xticks([]), plt.yticks([])
    plt.title(title)

np.set_printoptions(threshold=np.nan)

X_train = np.genfromtxt("X_train.csv",delimiter=',')
X_test = np.genfromtxt("X_test.csv",delimiter=',')
y_train = np.genfromtxt("y_train.csv",delimiter=',')
y_test = np.genfromtxt("y_test.csv",delimiter=',')
X_train, y_train = shuffle(X_train, y_train, random_state=2)
knn = 20

'''
for weights in ['uniform', 'distance']:
	#we create an instance of Neighbors Classifier and fit the data.
	
	clf = KNeighborsClassifier(knn, weights=weights)
	clf.fit(X_train, y_train)
	Z = clf.score(X_test,y_test)
	print(Z)
'''	
# bClf = BaggingClassifier(KNeighborsClassifier(knn,weights='uniform'),
# 	n_estimators=10,max_samples=0.7,max_features=0.7)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# bClf.fit(X_train,y_train)
# print(bClf.score(X_test,y_test))

X = X_train
y = y_train
n_samples, n_features = X.shape
X_pca = decomposition.PCA(n_components=2).fit_transform(X)
embedding_plot(X_pca, "PCA")
plt.show()




