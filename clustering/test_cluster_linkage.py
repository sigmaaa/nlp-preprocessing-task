import unittest
import numpy as np
from cluster_linkage import (
    single_linkage,
    complete_linkage,
    average_linkage,
    centroid_linkage
)


class TestClusterLinkage(unittest.TestCase):
    def setUp(self):
        self.dist_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.5, 2.8],
            [2.0, 2.5, 0.0, 0.5],
            [3.0, 2.8, 0.5, 0.0]
        ])
        self.feature_matrix = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

    # === SINGLE LINKAGE ===
    def test_single_link_basic(self):
        self.assertAlmostEqual(single_linkage([0], [1], self.dist_matrix), 1.0)

    def test_single_link_symmetric(self):
        d1 = single_linkage([0], [2], self.dist_matrix)
        d2 = single_linkage([2], [0], self.dist_matrix)
        self.assertAlmostEqual(d1, d2)

    def test_single_link_self(self):
        self.assertEqual(single_linkage([0], [0], self.dist_matrix), 0.0)

    def test_single_link_multi_to_single(self):
        self.assertEqual(single_linkage([0, 1], [2], self.dist_matrix), 2.0)

    def test_single_link_multi_to_multi(self):
        self.assertEqual(single_linkage([0, 1], [2, 3], self.dist_matrix), 2.0)

    # === COMPLETE LINKAGE ===
    def test_complete_link_basic(self):
        self.assertAlmostEqual(complete_linkage(
            [0], [3], self.dist_matrix), 3.0)

    def test_complete_link_self(self):
        self.assertEqual(complete_linkage([1], [1], self.dist_matrix), 0.0)

    def test_complete_link_multi(self):
        self.assertEqual(complete_linkage(
            [0, 1], [2, 3], self.dist_matrix), 3.0)

    def test_complete_link_edge(self):
        self.assertEqual(complete_linkage([2], [3], self.dist_matrix), 0.5)

    def test_complete_link_symmetry(self):
        d1 = complete_linkage([0, 2], [1, 3], self.dist_matrix)
        d2 = complete_linkage([1, 3], [0, 2], self.dist_matrix)
        self.assertAlmostEqual(d1, d2)

    # === AVERAGE LINKAGE ===
    def test_average_link_basic(self):
        self.assertAlmostEqual(average_linkage(
            [0], [3], self.dist_matrix), 3.0)

    def test_average_link_equal(self):
        self.assertAlmostEqual(average_linkage(
            [0], [1], self.dist_matrix), 1.0)

    def test_average_link_multi(self):
        expected = np.mean([2.0, 3.0, 2.5, 2.8])
        self.assertAlmostEqual(average_linkage(
            [0, 1], [2, 3], self.dist_matrix), expected)

    def test_average_link_self(self):
        self.assertEqual(average_linkage([2], [2], self.dist_matrix), 0.0)

    def test_average_link_nonoverlapping(self):
        expected = np.mean([1.0, 2.0])
        self.assertAlmostEqual(average_linkage(
            [0], [1, 2], self.dist_matrix), expected)

    # === CENTROID LINKAGE ===
    def test_centroid_link_basic(self):
        d = centroid_linkage([0], [1], self.feature_matrix)
        self.assertAlmostEqual(d, 1.0)

    def test_centroid_link_opposite(self):
        d = centroid_linkage([0], [3], self.feature_matrix)
        expected = np.linalg.norm(np.array([0, 0]) - np.array([1, 1]))
        self.assertAlmostEqual(d, expected)

    def test_centroid_link_same(self):
        self.assertEqual(centroid_linkage([2], [2], self.feature_matrix), 0.0)

    def test_centroid_link_avg_centroid(self):
        d = centroid_linkage([0, 1], [2, 3], self.feature_matrix)
        # centroid1 = [0.5, 0.0], centroid2 = [0.5, 1.0] → dist = 1.0
        self.assertAlmostEqual(d, 1.0)

    def test_centroid_link_symmetric(self):
        d1 = centroid_linkage([0, 1], [2, 3], self.feature_matrix)
        d2 = centroid_linkage([2, 3], [0, 1], self.feature_matrix)
        self.assertAlmostEqual(d1, d2)


if __name__ == '__main__':
    unittest.main()
