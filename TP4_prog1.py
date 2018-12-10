import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data

#file

file = open("/home/hamid/Bureau/APPRENTISSAGE-SUPERVISE/t4.8k.dat","r")
centers = []
for line in file:
   a = line.split()
   b = [float(a[0]),float(a[1])]
   centers.append(b)

X, labels_true = make_blobs(n_samples=len(centers), centers=centers, cluster_std=0.4, random_state=0)

#avec les 2 premiers dataset : eps=8.8, min_samples=17

# #############################################################################
# Compute DBSCAN
def computeDbscan(epsParam, min_samplesParam):
   db = DBSCAN(eps=epsParam, min_samples=min_samplesParam).fit(X)
   print("### computing with eps = " + str(epsParam) + " and min_samples = " + str(min_samplesParam) + "###")
   return db

db = computeDbscan(11,10).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


def printResults():
   print('Estimated number of clusters: %d' % n_clusters_)
   print('Estimated number of noise points: %d' % n_noise_)
   print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
   print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result

def printPlot(labels, X, core_samples_mask):
   # Black removed and is used for noise instead.
   unique_labels = set(labels)
   colors = [plt.cm.Spectral(each)
           for each in np.linspace(0, 1, len(unique_labels))]
   for k, col in zip(unique_labels, colors):
       if k == -1:
           # Black used for noise.
           col = [0, 0, 0, 1]

       class_member_mask = (labels == k)

       xy = X[class_member_mask & core_samples_mask]
       plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=14)

       xy = X[class_member_mask & ~core_samples_mask]
       plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=6)

   plt.title('Estimated number of clusters: %d' % n_clusters_)
   plt.show()

printResults()
printPlot(labels, X, core_samples_mask)
