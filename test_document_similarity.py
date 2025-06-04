import unittest
import numpy as np
from document_similarity import (
    compute_euclidean_distance,
    compute_cosine_similarity,
    compute_kl_divergence,
    compute_jaccard_similarity
)


class TestDocumentSimilarity(unittest.TestCase):

    def test_euclidean_distance_basic(self):
        matrix = np.array([
            [1, 0],
            [0, 1]
        ])
        dist = compute_euclidean_distance(matrix)
        expected = np.array([
            [0.0, np.sqrt(2)],
            [np.sqrt(2), 0.0]
        ])
        np.testing.assert_almost_equal(dist, expected)

    def test_euclidean_distance_identical_rows(self):
        matrix = np.array([
            [3, 4],
            [3, 4]
        ])
        dist = compute_euclidean_distance(matrix)
        expected = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        np.testing.assert_array_equal(dist, expected)

    def test_euclidean_distance_multiple_documents(self):
        matrix = np.array([
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        dist = compute_euclidean_distance(matrix)
        expected = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, np.sqrt(2)],
            [1.0, np.sqrt(2), 0.0]
        ])
        np.testing.assert_almost_equal(dist, expected)

    def test_cosine_similarity_identical(self):
        matrix = np.array([
            [1, 2],
            [1, 2]
        ])
        sim = compute_cosine_similarity(matrix)
        expected = np.zeros((2, 2))
        np.testing.assert_allclose(sim, expected, rtol=1e-5, atol=1e-8)

    def test_cosine_similarity_orthogonal(self):
        matrix = np.array([
            [1, 0],
            [0, 1]
        ])
        sim = compute_cosine_similarity(matrix)
        expected = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        np.testing.assert_allclose(sim, expected, rtol=1e-5, atol=1e-8)

    def test_cosine_similarity_zero_vector(self):
        matrix = np.array([
            [0, 0],
            [1, 1]
        ])
        sim = compute_cosine_similarity(matrix)
        expected = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        np.testing.assert_allclose(sim, expected, rtol=1e-5, atol=1e-8)

    def test_cosine_similarity_different_magnitude(self):
        matrix = np.array([
            [1, 0],
            [10, 0]
        ])
        sim = compute_cosine_similarity(matrix)
        expected = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])  # Angle is zero => cos = 1 => 1 - cos = 0
        np.testing.assert_allclose(sim, expected, rtol=1e-5, atol=1e-8)

    def test_cosine_similarity_three_docs(self):
        matrix = np.array([
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        sim = compute_cosine_similarity(matrix)
        # Expected is manually calculated: 1 - cos(theta)
        dot_01 = 1
        norm_0 = 1
        norm_1 = np.sqrt(2)
        cos_01 = dot_01 / (norm_0 * norm_1)

        expected = np.array([
            [0.0, 1 - cos_01, 1.0],
            [1 - cos_01, 0.0, 1 - cos_01],
            [1.0, 1 - cos_01, 0.0]
        ])
        np.testing.assert_allclose(sim, expected, rtol=1e-5, atol=1e-8)

    def test_kl_divergence_identical(self):
        matrix = np.array([
            [0.4, 0.6],
            [0.4, 0.6]
        ])
        kl = compute_kl_divergence(matrix)
        expected = np.zeros((2, 2))
        np.testing.assert_allclose(kl, expected, rtol=1e-5, atol=1e-8)

    def test_kl_divergence_asymmetric_distributions(self):
        matrix = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        kl = compute_kl_divergence(matrix)

        self.assertGreater(kl[0, 1], 0)
        self.assertAlmostEqual(kl[0, 1], kl[1, 0], places=6)

    def test_jaccard_similarity_identical(self):
        matrix = np.array([
            [1, 1, 0],
            [1, 1, 0]
        ])
        jac = compute_jaccard_similarity(matrix)
        expected = np.zeros((2, 2))  # 1 - 1 = 0 distance
        np.testing.assert_allclose(jac, expected, rtol=1e-5, atol=1e-8)

    def test_jaccard_similarity_disjoint(self):
        matrix = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        jac = compute_jaccard_similarity(matrix)
        expected = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        np.testing.assert_allclose(jac, expected, rtol=1e-5, atol=1e-8)

    def test_jaccard_similarity_partial_overlap(self):
        matrix = np.array([
            [1, 1, 0],
            [1, 0, 1]
        ])
        jac = compute_jaccard_similarity(matrix)
        # |A∩B| = 1, |A∪B| = 3 => similarity = 1 - 1/3 = 0.666...
        expected_val = 1 - 1 / 3
        self.assertAlmostEqual(jac[0, 1], expected_val, places=6)
        self.assertAlmostEqual(jac[1, 0], expected_val, places=6)

    def test_kl_divergence_three_documents(self):
        matrix = np.array([
            [0.5, 0.5],
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        kl = compute_kl_divergence(matrix)
        self.assertEqual(kl.shape, (3, 3))
        np.testing.assert_allclose(np.diag(kl), np.zeros(3), atol=1e-8)
        self.assertAlmostEqual(kl[1, 2], kl[2, 1], places=6)
        self.assertGreater(kl[0, 1], 0)

    def test_kl_divergence_zeros_in_vector(self):
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        kl = compute_kl_divergence(matrix)
        # Should not be infinite due to epsilon
        self.assertTrue(np.all(np.isfinite(kl)))
        self.assertAlmostEqual(kl[0, 1], kl[1, 0], places=6)
        self.assertGreater(kl[0, 1], 0)

    def test_kl_divergence_row_sum_not_1(self):
        matrix = np.array([
            [2, 3],
            [4, 1]
        ])
        # Should normalize rows internally
        kl = compute_kl_divergence(matrix)
        self.assertAlmostEqual(kl[0, 0], 0.0)
        self.assertAlmostEqual(kl[1, 1], 0.0)
        self.assertTrue(np.all(kl >= 0))

    def test_jaccard_similarity_more_docs(self):
        matrix = np.array([
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])
        jac = compute_jaccard_similarity(matrix)
        self.assertEqual(jac.shape, (3, 3))
        np.testing.assert_allclose(np.diag(jac), np.zeros(3), atol=1e-8)
        self.assertAlmostEqual(jac[0, 1], 1 - 2/3, places=6)
        self.assertAlmostEqual(jac[1, 2], 1 - 2/3, places=6)

    def test_jaccard_similarity_zero_vector(self):
        matrix = np.array([
            [0, 0, 0],
            [1, 0, 1]
        ])
        jac = compute_jaccard_similarity(matrix)
        self.assertEqual(jac[0, 1], 1.0)
        self.assertEqual(jac[1, 0], 1.0)

    def test_jaccard_similarity_all_disjoint(self):
        matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        jac = compute_jaccard_similarity(matrix)
        self.assertTrue(np.allclose(jac, np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ]), atol=1e-8))

    def test_jaccard_similarity_full_overlap(self):
        matrix = np.array([
            [1, 1, 1],
            [1, 1, 1]
        ])
        jac = compute_jaccard_similarity(matrix)
        expected = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        np.testing.assert_allclose(jac, expected, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
