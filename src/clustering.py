import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import scipy.cluster.hierarchy as sch
from src.contact_mtx_manipulator import ContactMatrixScaling


class Hierarchical:
    def __init__(self, c_mtx_gen: ContactMatrixScaling, country_names: np.ndarray,
                 img_prefix: str,
                 dist: str = "euclidean"):
        """
        c_mtx_gen: instance of ContactMatrixScaling
        country_names: list or array of country names
        img_prefix: prefix for plots (if needed elsewhere)
        dist: 'euclidean' or 'manhattan'
        """
        self.c_mtx_gen = c_mtx_gen
        self.country_names = country_names
        self.img_prefix = img_prefix

        if dist == "euclidean":
            self.get_distance_matrix = self.get_euclidean_distance
        elif dist == "manhattan":
            self.get_distance_matrix = self.get_manhattan_distance
        else:
            raise ValueError(f"Unsupported distance: {dist}")

    def get_manhattan_distance(self):
        """Calculate Manhattan distances between countries."""
        manhattan_distance = manhattan_distances(self.c_mtx_gen.data_clustering)
        dt = pd.DataFrame(manhattan_distance,
                          index=self.country_names, columns=self.country_names)
        return dt, manhattan_distance

    def get_euclidean_distance(self):
        """Calculate Euclidean distances between countries."""
        euc_distance = euclidean_distances(self.c_mtx_gen.data_clustering)
        dt = pd.DataFrame(euc_distance,
                          index=self.country_names, columns=self.country_names)
        return dt, euc_distance

    def calculate_ordered_distance_matrix(self, threshold: float, verbose=True):
        dt, distance = self.get_distance_matrix()
        distances = distance[np.triu_indices_from(distance, k=1)]
        res = sch.linkage(distances, method="complete")

        ordered_indices = sch.leaves_list(res)
        ordered_columns = [self.country_names[i] for i in ordered_indices]
        dt = dt.reindex(index=ordered_columns, columns=ordered_columns)

        return ordered_columns, dt, res




