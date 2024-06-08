import pandas as pd
import numpy as np

np.random.seed(0)


class AnomalyDataset:
    def __init__(self):
        self.data = []
        # self.data = [
        #     [1, 2],
        #     [2, 3],
        #     [10, 10],
        #     [11, 11],
        #     [2, 1],
        #     [2, 2.5],
        #     [2, 2.2],
        #     [2.2, 2],
        #     [2, 2.1],
        #     [2.1, 2]
        # ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_list(self):
        return self.data

    def get_labels(self):
        return []


class BreastCancerDataset(AnomalyDataset):
    def __init__(self, filename, malignant_percentage_drop=0.0):
        super().__init__()

        self.data = pd.read_csv(filename, header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], names=[
            'clump_thickness',
            'uniformity_of_cell_size',
            'uniformity_of_cell_shape',
            'marginal_adhesion',
            'single_apithelial_cell_size',
            'bare_nuclei',
            'bland_chromatin',
            'normal_nucleoli',
            'mitoses',
            'class'
        ])
        self.data = self.data[(self.data != '?').all(axis=1)]

        malignant_entries = self.data[self.data['class'] == 4].index.tolist()
        drop_index = np.random.choice(malignant_entries, size=int(len(malignant_entries) * malignant_percentage_drop), replace=False)
        self.data = self.data.drop(drop_index).astype(int)

    def __getitem__(self, idx):
        return self.data.iloc[idx].tolist()[:-1]

    def __len__(self):
        return self.data.shape[0]

    def get_list(self, with_class=False):
        if not with_class:
            return self.data.drop(['class'], axis=1).values.tolist()
        else:
            return self.data.values.tolist()

    def get_labels(self):
        return self.data['class'].tolist()

    def split(self, training_percentage):
        anomalies_df = self.data[self.data["class"] == 4]
        normal_df = self.data[self.data["class"] != 4]

        anomalies_df_idx = anomalies_df.index.tolist()
        anomalies_df_train_idx = np.random.choice(anomalies_df_idx, size=int(len(anomalies_df_idx) * training_percentage), replace=False)
        anomalies_df_train = anomalies_df[anomalies_df.index.isin(anomalies_df_train_idx)]
        anomalies_df_test = anomalies_df.drop(anomalies_df_train_idx)

        normal_df_idx = normal_df.index.tolist()
        normal_df_train_idx = np.random.choice(normal_df_idx, size=int(len(normal_df_idx) * training_percentage), replace=False)
        normal_df_train = normal_df[normal_df.index.isin(normal_df_train_idx)]
        normal_df_test = normal_df.drop(normal_df_train_idx)

        x_train = pd.concat([anomalies_df_train.drop(['class'], axis=1), normal_df_train.drop(['class'], axis=1)]).sample(frac=1).reset_index(drop=True).values.tolist()
        y_train = pd.concat([anomalies_df_train['class'], normal_df_train['class']]).sample(frac=1).reset_index(drop=True).values.tolist()
        x_test = pd.concat([anomalies_df_test.drop(['class'], axis=1), normal_df_test.drop(['class'], axis=1)]).sample(frac=1).reset_index(drop=True).values.tolist()
        y_test = pd.concat([anomalies_df_test['class'], normal_df_test['class']]).sample(frac=1).reset_index(drop=True).values.tolist()

        return x_train, x_test, y_train, y_test
    
class WineDataset(AnomalyDataset):
    def __init__(self, filename, class_to_drop, drop_percentage):
        super().__init__()
        self.data = pd.read_csv(filename, header=None, names=[
            "class",
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "OD280/OD315_of_diluted_wines",
            "proline",
        ])

        anomaly_entries = self.data[self.data['class'] == class_to_drop].index.tolist()
        drop_index = np.random.choice(anomaly_entries, size=int(len(anomaly_entries) * drop_percentage), replace=False)
        self.data = self.data.drop(drop_index)

    def __getitem__(self, idx):
        return self.data.iloc[idx].tolist()[1:]

    def __len__(self):
        return self.data.shape[0]

    def get_list(self, with_class=False):
        if not with_class:
            return self.data.drop(['class'], axis=1).values.tolist()
        else:
            return self.data.values.tolist()

    def get_labels(self):
        return self.data['class'].tolist()

    def split(self, outlier_class, training_percentage):
        anomalies_df = self.data[self.data["class"] == outlier_class]
        normal_df = self.data[self.data["class"] != outlier_class]

        anomalies_df_idx = anomalies_df.index.tolist()
        anomalies_df_train_idx = np.random.choice(anomalies_df_idx, size=int(len(anomalies_df_idx) * training_percentage), replace=False)
        anomalies_df_train = anomalies_df[anomalies_df.index.isin(anomalies_df_train_idx)]
        anomalies_df_test = anomalies_df.drop(anomalies_df_train_idx)

        normal_df_idx = normal_df.index.tolist()
        normal_df_train_idx = np.random.choice(normal_df_idx, size=int(len(normal_df_idx) * training_percentage), replace=False)
        normal_df_train = normal_df[normal_df.index.isin(normal_df_train_idx)]
        normal_df_test = normal_df.drop(normal_df_train_idx)

        x_train = pd.concat([anomalies_df_train.drop(['class'], axis=1), normal_df_train.drop(['class'], axis=1)]).sample(frac=1).reset_index(drop=True).values.tolist()
        y_train = pd.concat([anomalies_df_train['class'], normal_df_train['class']]).sample(frac=1).reset_index(drop=True).values.tolist()
        x_test = pd.concat([anomalies_df_test.drop(['class'], axis=1), normal_df_test.drop(['class'], axis=1)]).sample(frac=1).reset_index(drop=True).values.tolist()
        y_test = pd.concat([anomalies_df_test['class'], normal_df_test['class']]).sample(frac=1).reset_index(drop=True).values.tolist()

        return x_train, x_test, y_train, y_test
