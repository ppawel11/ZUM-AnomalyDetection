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
