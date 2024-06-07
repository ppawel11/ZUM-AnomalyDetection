import math


class DissimilarityCalculator:
    def __init__(self, group_center_method, points_distance_method):
        self.group_center_method = group_center_method
        self.points_distance_method = points_distance_method

        self.groups_centers = None
        self.groups = None
        self.data = None

    def calculate_dissimilarities(self, dataset, groups):
        self.data = dataset
        self.groups = groups

        self.calculate_groups_centers()

        dissimilarities = []

        for example_id, example in enumerate(self.data):
            dissimilarities.append((example_id, self.dissimilarity_factor(example)))
        dissimilarities.sort(key=lambda tup: tup[1], reverse=True)

        return dissimilarities

    def calculate_groups_centers(self):
        centers = []
        for group_id in set(self.groups):
            if group_id == -1:
                # Group value -1 stands for noise
                continue
            group = self.get_group(group_id)
            centers.append(self.group_center(group))
        self.groups_centers = centers

    def group_center(self, group):
        center = None
        if self.group_center_method == 'average':
            center = [sum(x) for x in zip(*group)]
            center = [x / len(group) for x in center]
        return center

    def two_points_distance(self, point1, point2):
        d = -1
        if self.points_distance_method == 'euclidian':
            d = math.pow(sum([math.pow(x - y, 2) for x, y in zip(point1, point2)]), 0.5)
        return d

    def get_group(self, group_id):
        group = []
        for example_id, example in enumerate(self.data):
            if self.groups[example_id] == group_id:
                group.append(example)
        return group

    def dissimilarity_factor(self, point):
        pass


class NaiveDissimilarityCalculator(DissimilarityCalculator):
    def __init__(self, group_center_method, points_distance_method):
        super().__init__(group_center_method, points_distance_method)

    def dissimilarity_factor(self, point):
        df = 10000000000
        for group_id in set(self.groups):
            df = min(df, self.two_points_distance(point, self.groups_centers[group_id]))
        return df