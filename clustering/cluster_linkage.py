import numpy as np


def single_linkage(cluster1, cluster2, distance_matrix):

    return np.min([distance_matrix[i, j] for i in cluster1 for j in cluster2])


def complete_linkage(cluster1, cluster2, distance_matrix):

    return np.max([distance_matrix[i, j] for i in cluster1 for j in cluster2])


def average_linkage(cluster1, cluster2, distance_matrix):

    distances = [distance_matrix[i, j] for i in cluster1 for j in cluster2]
    return np.mean(distances)


def centroid_linkage(cluster1, cluster2, feature_matrix):
    centroid1 = np.mean(feature_matrix[cluster1], axis=0)
    centroid2 = np.mean(feature_matrix[cluster2], axis=0)
    return np.linalg.norm(centroid1 - centroid2)
