import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa

from document_similarity import compute_euclidean_distance, compute_cosine_similarity
from cluster_linkage import single_linkage, complete_linkage, average_linkage
from document_clustering import (
    k_means_clustering,
    one_pass_clustering,
    agglomerative_clustering
)

import unittest
import numpy as np
from sklearn.preprocessing import normalize


class TestDocumentClustering(unittest.TestCase):

    def setUp(self):
        # Простий набір документів (4 документа, 2 терми)
        self.matrix = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0]
        ])

    # === (a) K-Means ===
    def test_kmeans_euclidean(self):
        labels = k_means_clustering(self.matrix, distance='euclidean', k=2)
        self.assertEqual(len(labels), self.matrix.shape[0])

    def test_kmeans_cosine(self):
        labels = k_means_clustering(self.matrix, distance='cosine', k=2)
        self.assertEqual(len(labels), self.matrix.shape[0])

    # === (b) One-pass clustering ===
    def test_one_pass_single_linkage(self):
        dist = compute_euclidean_distance(self.matrix)
        labels = one_pass_clustering(
            self.matrix, dist, single_linkage, threshold=1.5)
        self.assertEqual(len(labels), self.matrix.shape[0])

    def test_one_pass_complete_linkage(self):
        dist = compute_cosine_similarity(self.matrix)
        labels = one_pass_clustering(
            self.matrix, dist, complete_linkage, threshold=1.0)
        self.assertEqual(len(labels), self.matrix.shape[0])

    # === (c) Agglomerative clustering ===
    def test_agglomerative_single_euclidean(self):
        labels = agglomerative_clustering(
            self.matrix, linkage_type='single', metric='euclidean', k=2)
        self.assertEqual(len(labels), self.matrix.shape[0])

    def test_agglomerative_average_cosine(self):
        labels = agglomerative_clustering(
            self.matrix, linkage_type='average', metric='cosine', k=2)
        self.assertEqual(len(labels), self.matrix.shape[0])


if __name__ == '__main__':
    unittest.main()
