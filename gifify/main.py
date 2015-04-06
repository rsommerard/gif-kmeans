# coding=utf-8

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

file = sys.argv[1]
n_colors = int(sys.argv[2])

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    img = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            img[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return img

image = cv2.imread(file)
image = np.array(image, dtype=np.float64) / 255

w, h, d = original_shape = tuple(image.shape)
image_array = np.reshape(image, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
print("done in %0.3fs." % (time() - t0))

cv2.imshow('Original image', image)
cv2.imshow('Quantized image K-Means', recreate_image(kmeans.cluster_centers_, labels, w, h))
cv2.imshow('Quantized image Random', recreate_image(codebook_random, labels_random, w, h))
cv2.waitKey(0)
