import math
from anomaly_dataset import AnomalyDataset
from grouping_algorithm import GroupingAlgorithm
from dissimilarities_calculator import DissimilarityCalculator


class AnomalyDetector:
    def __init__(self, grouping_algorithm: GroupingAlgorithm, dissimilarity_calculator: DissimilarityCalculator):
        self.grouping_algorithm = grouping_algorithm
        self.dissimilarity_calculator = dissimilarity_calculator

    def detect_anomalies(self, dataset: AnomalyDataset, anomalies_percentage=None, anomalies_count=None):
        groups = self.grouping_algorithm.group(dataset)

        print("Groups detected")

        dissimilarities = self.dissimilarity_calculator.calculate_dissimilarities(dataset.get_list(), groups)

        print("Dissimilarities calculated")

        if anomalies_percentage is not None:
            limit = int(len(dissimilarities) * anomalies_percentage)
            anomalies_ids = [x[0] for x in dissimilarities[:limit]]
            return [1 if example_id in anomalies_ids else 0 for example_id in range(len(dataset))]

        if anomalies_count is not None:
            anomalies_ids = [x[0] for x in dissimilarities[:anomalies_count]]
            return [1 if example_id in anomalies_ids else 0 for example_id in range(len(dataset))]
