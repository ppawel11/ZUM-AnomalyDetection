from anomaly_detector import AnomalyDetector
from dissimilarities_calculator import NaiveDissimilarityCalculator, CBLOFDissimilarityCalculator, LDCOFDissimilarityCalculator
from grouping_algorithm import GroupingAlgorithm, KMeansGroupingAlgorithm, DBSCANGroupingAlgorithm
from anomaly_dataset import AnomalyDataset, BreastCancerDataset, WineDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
# from sklearn.cluster import DBSCAN
# -------------------------------

# ad = AnomalyDetector(AnomalyDataset(), GroupingAlgorithm())
#
# print(ad.detect_anomalies(anomalies_count=2))

import numpy as np

np.random.seed(0)

# wd = WineDataset('./data/wine/wine.data', class_to_drop=1, drop_percentage=0.84)
bcd = BreastCancerDataset('./data/breast-cancer/breast-cancer-wisconsin.data', malignant_percentage_drop=0.84)

# x_train, x_test, y_train, y_test = wd.split(outlier_class=1, training_percentage=0.7)

# print(len(x_train), len(x_test), len(y_train), len(y_test))

# clf = IsolationForest(random_state=0).fit(x_train)
# clf = OneClassSVM(gamma='auto').fit(x_train)
# prediction = clf.predict(x_test)

# print(prediction)
# print(y_test)

# correct_anomalies = 0
# all_anomalies = 0
# actual_anomalies = 0
# for pred, label in zip(prediction, y_test):
#     if label == 1:
#         actual_anomalies += 1
#     if pred == -1:
#         all_anomalies += 1
#         if label == 1:
#             correct_anomalies += 1
#
# print("Anomalies detected correctly: ", correct_anomalies)
# print("Total anomalies detected: ", all_anomalies)
# print("Total number of anomalies: ", actual_anomalies)

ga = DBSCANGroupingAlgorithm(eps=5, min_samples=2)
# ga = KMeansGroupingAlgorithm(n_clusters=5, n_init="auto", random_state=0)

dc = NaiveDissimilarityCalculator(group_center_method='average', points_distance_method='euclidian')
# dc = CBLOFDissimilarityCalculator(group_center_method='average', points_distance_method='euclidian', alpha=0.9, beta=5, u=True)
# dc = LDCOFDissimilarityCalculator(group_center_method='average', points_distance_method='euclidian', alpha=0.9, beta=5)

ad = AnomalyDetector(ga, dc)

anomalies = ad.detect_anomalies(bcd, anomalies_percentage=0.1)
# anomalies = ad.detect_anomalies(wd, anomalies_percentage=0.05)

correct_count = 0
total_count = 0
for anomaly, label in zip(anomalies, bcd.get_labels()):
# for anomaly, label in zip(anomalies, wd.get_labels()):
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

