from anomaly_detector import AnomalyDetector
from dissimilarities_calculator import (
    NaiveDissimilarityCalculator,
    CBLOFDissimilarityCalculator,
    LDCOFDissimilarityCalculator,
)
from grouping_algorithm import (
    KMeansGroupingAlgorithm,
    DBSCANGroupingAlgorithm,
)
from anomaly_dataset import BreastCancerDataset, WineDataset

import sklearn.metrics as sklm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd


def eval(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    fscore = sklm.f1_score(y_true, y_pred)
    precision = sklm.precision_score(y_true, y_pred)
    accuracy = sklm.accuracy_score(y_true, y_pred)
    recall = sklm.recall_score(y_true, y_pred)

    return {
        "fscore": fscore,
        "precision": precision,
        "accuracy": accuracy,
        "recall": recall,
    }


class Evaluator:

    def __init__(
        self,
        dataset_name: str,
        anomaly_drop_rates: list[float],
        anomalies_perc_range: list[float],
        n_cluster_range: list[int],
        eps_range: list[float],
        min_samples_range: list[int],
    ):
        self.dataset_name = dataset_name
        self.anomaly_drop_rates = anomaly_drop_rates
        self.anomalies_perc_range = anomalies_perc_range
        self.n_cluster_range = n_cluster_range
        self.eps_range = eps_range
        self.min_samples_range = min_samples_range

    def eval_on_dataset(
        self,
        ad: AnomalyDetector,
    ) -> np.ndarray:
        evals = []
        for adr, ap in it.product(self.anomaly_drop_rates, self.anomalies_perc_range):
            if self.dataset_name == "breast_cancer":
                bcd = BreastCancerDataset(
                    "./data/breast-cancer/breast-cancer-wisconsin.data",
                    malignant_percentage_drop=adr,
                )
                anomaly_id = 4
            elif self.dataset_name == "wine":
                bcd = WineDataset(
                    "./data/wine/wine.data", class_to_drop=1, drop_percentage=adr
                )
                anomaly_id = 1
            else:
                raise Exception(f"'{self.dataset_name}' dataset not found")

            anomalies = np.array(ad.detect_anomalies(bcd, anomalies_percentage=ap))
            labels = np.array(bcd.get_labels())
            fpar = eval(labels == anomaly_id, anomalies == 1)
            fpar["anomalies_percentage"] = ap
            fpar["anomaly_drop_rates"] = adr
            evals.append(fpar)

        return evals

    def eval_kmeans(
        self,
        dc,
    ):
        results = []
        for n_clusters in self.n_cluster_range:
            ga = KMeansGroupingAlgorithm(n_clusters=n_clusters)
            ad = AnomalyDetector(ga, dc)
            evals = self.eval_on_dataset(ad)
            for e in evals:
                e["grouping_algorithm"] = "kmeans"
                e["n_clusters"] = n_clusters
                e["eps"] = None
                e["min_samples"] = None
            results.extend(evals)
        return results

    def eval_dbscan(
        self,
        dc,
    ) -> np.ndarray:
        results = []
        for eps in self.eps_range:
            for min_samples in self.min_samples_range:
                ga = DBSCANGroupingAlgorithm(eps=eps, min_samples=min_samples)
                ad = AnomalyDetector(ga, dc)
                evals = self.eval_on_dataset(ad)
                for e in evals:
                    e["grouping_algorithm"] = "dbscan"
                    e["n_clusters"] = None
                    e["eps"] = eps
                    e["min_samples"] = min_samples

                results.extend(evals)
        return results

    def run(self) -> pd.DataFrame:
        dc = NaiveDissimilarityCalculator(
            group_center_method="average", points_distance_method="euclidian"
        )

        def add_col(data, val):
            for x in data:
                x["dissimilarity"] = val
            return data

        results = []

        out = self.eval_kmeans(dc)
        results.extend(add_col(out, "naive"))

        out = self.eval_dbscan(dc)
        results.extend(add_col(out, "naive"))

        dc = CBLOFDissimilarityCalculator(
            group_center_method="average",
            points_distance_method="euclidian",
            alpha=0.9,
            beta=5,
            u=True,
        )

        out = self.eval_kmeans(dc)
        results.extend(add_col(out, "uCBLOF"))

        out = self.eval_dbscan(dc)
        results.extend(add_col(out, "uCBLOF"))

        dc = CBLOFDissimilarityCalculator(
            group_center_method="average",
            points_distance_method="euclidian",
            alpha=0.9,
            beta=5,
            u=False,
        )
        out = self.eval_kmeans(dc)
        results.extend(add_col(out, "CBLOF"))

        out = self.eval_dbscan(dc)
        results.extend(add_col(out, "CBLOF"))

        dc = LDCOFDissimilarityCalculator(
            group_center_method="average",
            points_distance_method="euclidian",
            alpha=0.9,
            beta=5,
        )

        out = self.eval_kmeans(dc)
        results.extend(add_col(out, "LDCOF"))

        out = self.eval_dbscan(dc)
        results.extend(add_col(out, "LDCOF"))

        df = pd.DataFrame(results)

        sns.boxplot(df, x="dissimilarity", y="fscore", hue="grouping_algorithm")
        plt.title(f"F-score on {self.dataset_name.replace('_', ' ')}")
        plt.show()

        sns.boxplot(df, x="dissimilarity", y="precision", hue="grouping_algorithm")
        plt.title(f"precision on {self.dataset_name.replace('_', ' ')}")
        plt.show()

        sns.boxplot(df, x="dissimilarity", y="accuracy", hue="grouping_algorithm")
        plt.title(f"accuracy on {self.dataset_name.replace('_', ' ')}")
        plt.show()

        sns.boxplot(df, x="dissimilarity", y="recall", hue="grouping_algorithm")
        plt.title(f"recall on {self.dataset_name.replace('_', ' ')}")
        plt.show()

        return df


def main():
    np.random.seed(0)

    dataset_name = "breast_cancer"

    evl = Evaluator(
        dataset_name,
        anomaly_drop_rates=[0.3, 0.5, 0.8],
        anomalies_perc_range=[0.05, 0.1, 0.2],
        n_cluster_range=[3, 4, 5, 6],
        eps_range=[13, 17, 23],
        min_samples_range=[2],
    )

    df = evl.run()
    df.to_csv(f"{dataset_name}_eval_data.csv")


if __name__ == "__main__":
    main()
