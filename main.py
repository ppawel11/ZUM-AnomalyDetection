from anomaly_detector import AnomalyDetector
from grouping_algorithm import GroupingAlgorithm, KMeansGroupingAlgorithm, DBSCANGroupingAlgorithm
from anomaly_dataset import AnomalyDataset, BreastCancerDataset

# from sklearn.cluster import DBSCAN
# -------------------------------

# ad = AnomalyDetector(AnomalyDataset(), GroupingAlgorithm())
#
# print(ad.detect_anomalies(anomalies_count=2))


bcd = BreastCancerDataset('./data/breast-cancer/breast-cancer-wisconsin.data', malignant_percentage_drop=0.84)

ga = DBSCANGroupingAlgorithm(eps=5, min_samples=2)
# labels = ga.group(bcd)
# print(labels)
# print(list(labels).count(-1))
# print(set(labels))
# ga = KMeansGroupingAlgorithm(n_clusters=5, n_init="auto", random_state=0)
ad = AnomalyDetector(bcd, ga)

anomalies = ad.detect_anomalies(anomalies_percentage=0.05)

correct_count = 0
total_count = 0
for anomaly, label in zip(anomalies, bcd.get_labels()):
    if anomaly == 1:
        total_count += 1
        if label == 4:
            correct_count += 1

print("Anomalies detected correctly: ", correct_count)
print("Total anomalies detected: ", total_count)

#############################################################################3

# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
#
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(
#     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
# )
#
# X = StandardScaler().fit_transform(X)
#
# import matplotlib.pyplot as plt
#
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
#
# import numpy as np
#
# from sklearn import metrics
# from sklearn.cluster import DBSCAN
#
# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# labels = db.labels_
# print(len(labels))
# print(len(X))
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# print("Estimated number of clusters: %d" % n_clusters_)
# print("Estimated number of noise points: %d" % n_noise_)
