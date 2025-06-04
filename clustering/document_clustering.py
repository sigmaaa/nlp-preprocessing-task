import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def k_means_clustering(doc_word_matrix, k=3):
    model = KMeans(n_clusters=k, init=doc_word_matrix[:k], n_init=1)
    return model.fit_predict(doc_word_matrix)


def one_pass_clustering(distance_matrix, linkage_func, threshold):
    clusters = [[0]]

    for i in range(1, distance_matrix.shape[0]):
        assigned = False
        for cluster in clusters:
            dist = linkage_func([i], cluster, distance_matrix)
            if dist < threshold:
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    labels = np.zeros(distance_matrix.shape[0], dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for doc_idx in cluster:
            labels[doc_idx] = cluster_idx
    return labels


def agglomerative_clustering(distance_matrix=None, linkage_type='single', k=3):
    if linkage_type in ['single', 'complete', 'average']:
        model = AgglomerativeClustering(
            n_clusters=k, metric='precomputed', linkage=linkage_type)
        return model.fit_predict(distance_matrix)
    else:
        raise ValueError(
            "Unsupported configuration. Use 'single', 'complete' or 'average' with euclidean/cosine.")
