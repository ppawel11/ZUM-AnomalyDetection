from anomaly_dataset import AnomalyDataset

from sklearn.cluster import k_means
from sklearn.cluster import DBSCAN

class GroupingAlgorithm:
    def __init__(self):
        pass

    def group(self, dataset: AnomalyDataset):
        # maybe all noise e.g. from DBSCAN should be separate groups?
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


class KMeansGroupingAlgorithm(GroupingAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def group(self, dataset: AnomalyDataset):
        centroid, label, inertia = k_means(dataset.get_list(), **self.kwargs)
        return label


class DBSCANGroupingAlgorithm(GroupingAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def group(self, dataset: AnomalyDataset):
        clustering = DBSCAN(**self.kwargs).fit(dataset.get_list())
        return list(clustering.labels_)