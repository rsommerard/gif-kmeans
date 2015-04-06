# coding=utf-8

import sklearn.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

X, Y = datasets.make_circles(n_samples=100, shuffle=True, noise=0.05, random_state=None, factor=0.3)
fig, axes = plt.subplots(nrows=1, ncols=2) # left, bottom, width, height (range 0 to 1)
fig.set_size_inches(15,5)
axes[0].scatter(X[:,0], X[:,1], s=100)
axes[1].scatter(X[:,0], X[:,1], c=Y, s=100)
plt.show()

kmeans = KMeans(2, random_state=8)
Y_hat = kmeans.fit(X).labels_
fig, axes = plt.subplots(nrows=1, ncols=2) # left, bottom, width, height (range 0 to 1)
fig.set_size_inches(15,5)
axes[0].scatter(X[:,0], X[:,1], c=Y, s=100)
axes[1].scatter(X[:,0], X[:,1], c=Y_hat, s=100)
plt.show()

spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors")
Y_hat = spectral.fit(X).labels_
fig, axes = plt.subplots(nrows=1, ncols=2) # left, bottom, width, height (range 0 to 1)
fig.set_size_inches(15,5)
axes[0].scatter(X[:,0], X[:,1], c=Y, s=100)
axes[1].scatter(X[:,0], X[:,1], c=Y_hat, s=100)
plt.show()



########

X, Y = datasets.make_circles(n_samples=100, shuffle=True, noise=0.05, random_state=None, factor=0.5)
Y_hat = spectral.fit(X).labels_
fig, axes = plt.subplots(nrows=1, ncols=2) # left, bottom, width, height (range 0 to 1)
fig.set_size_inches(15,5)
axes[0].scatter(X[:,0], X[:,1], c=Y, s=100)
axes[1].scatter(X[:,0], X[:,1], c=Y_hat, s=100)
plt.show()
