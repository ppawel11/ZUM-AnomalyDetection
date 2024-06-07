import math
from anomaly_dataset import AnomalyDataset
from grouping_algorithm import GroupingAlgorithm


class AnomalyDetector:
    def __init__(self, dataset: AnomalyDataset, grouping_algorithm: GroupingAlgorithm):
        self.grouping_algorithm = grouping_algorithm
        self.dataset = dataset

        self.groups = None
        self.groups_centers = None

        self.dissimilarity_method = 'simple_distance'
        self.group_center_method = 'average'
        self.distance_method = 'euclidian'

    def detect_groups(self):
        self.groups = self.grouping_algorithm.group(self.dataset)

    def calculate_groups_centers(self):
        centers = []
        for group_id in self.groups:
            if group_id == -1:
                continue
            # print("group ", group_id)
            group = self.get_group(group_id)
            centers.append(self.group_center(group))
        self.groups_centers = centers

    def detect_anomalies(self, anomalies_percentage=None, anomalies_count=None):
        self.detect_groups()

        print("Groups detected")

        self.calculate_groups_centers()

        print("Groups centers calculated")

        dissimilarities = self.calculate_dissimilarities()

        print("Dissimilarities calculated")

        if anomalies_percentage is not None:
            limit = int(len(dissimilarities) * anomalies_percentage)
            anomalies_ids = [x[0] for x in dissimilarities[:limit]]
            return [1 if example_id in anomalies_ids else 0 for example_id in range(len(self.dataset))]

        if anomalies_count is not None:
            anomalies_ids = [x[0] for x in dissimilarities[:anomalies_count]]
            return [1 if example_id in anomalies_ids else 0 for example_id in range(len(self.dataset))]

    def calculate_dissimilarities(self):
        dissimilarities = []

        for example_id, example in enumerate(self.dataset.get_list()):
            dissimilarities.append((example_id, self.dissimilarity_factor(example)))
        dissimilarities.sort(key=lambda tup: tup[1], reverse=True)

        return dissimilarities

    def group_center(self, group):
        # if self.groups_centers is not None:
        #     return self.groups_centers[group]

        center = None
        if self.group_center_method == 'average':
            center = [sum(x) for x in zip(*group)]
            center = [x/len(group) for x in center]
        return center

    def two_points_distance(self, point1, point2):
        d = -1
        if self.distance_method == 'euclidian':
            d = math.pow(sum([math.pow(x - y, 2) for x, y in zip(point1, point2)]), 0.5)
            pass
        return d

    def get_group(self, group_id):
        group = []
        for id in range(len(self.dataset)):
            if self.groups[id] == group_id:
                group.append(self.dataset[id])
        return group

    def dissimilarity_factor(self, point):
        df = None

        if self.dissimilarity_method == 'simple_distance':
            df = 10000000000
            for group_id in self.groups:
                # print("group ", group_id)
                # group = self.get_group(group_id)
                df = min(df, self.two_points_distance(point, self.groups_centers[group_id]))

        return df
