# coding=utf-8

import sklearn.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

print('-' * 80)
print('# Figure 1')
print('La figure 1 montre la répartition des données de base qui vont permettres la comparaison des méthodes de classification.')
print('-' * 80)

X, Y = datasets.make_circles(n_samples=100, shuffle=True, noise=0.05, random_state=None, factor=0.3)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.scatter(X[:,0], X[:,1], c=Y, s=100)
plt.show()

print('# Figure 2')
print('La figure 2 montre la classification des données par la méthode des K plus proche voisins.')
print("On remarque que cette méthode n'est pas très efficace pour cette répartition des données.")
print('-' * 80)

kmeans = KMeans(2, random_state=8)
Y_hat = kmeans.fit(X).labels_
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.scatter(X[:,0], X[:,1], c=Y_hat, s=100)
plt.show()

print('# Figure 3')
print('La figure 3 montre la classification des données par la méthode spectrale.')
print('On remarque que cette méthode est particulièrement bien adaptée pour ce type de données.')
print('-' * 80)

spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors")
Y_hat = spectral.fit(X).labels_
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.scatter(X[:,0], X[:,1], c=Y_hat, s=100)
plt.show()

print('# Figure 4')
print('La figure 4 montre la classification des données par la méthode spectrale.')
print('On a vu que la méthode spectrale est efficace pour les données précédentes. Cependant, si on modifie un peu ces données en raprochant les deux groupes, on atteint la limite de cette méthode.')
print('-' * 80)

X, Y = datasets.make_circles(n_samples=100, shuffle=True, noise=0.05, random_state=None, factor=0.5)
Y_hat = spectral.fit(X).labels_
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.scatter(X[:,0], X[:,1], c=Y_hat, s=100)
plt.show()
