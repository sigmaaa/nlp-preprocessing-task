import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances


def k_means_clustering(doc_word_matrix, distance='euclidean', k=3):
    if distance == 'euclidean':
        model = KMeans(n_clusters=k, init=doc_word_matrix[:k], n_init=1)
        return model.fit_predict(doc_word_matrix)
    elif distance == 'cosine':
        from sklearn.preprocessing import normalize
        norm_matrix = normalize(doc_word_matrix)
        model = KMeans(n_clusters=k, init=norm_matrix[:k], n_init=1)
        return model.fit_predict(norm_matrix)
    else:
        raise ValueError("Unsupported distance metric")


def one_pass_clustering(doc_word_matrix, distance_matrix, linkage_func, threshold):
    clusters = [[0]]

    for i in range(1, doc_word_matrix.shape[0]):
        assigned = False
        for cluster in clusters:
            dist = linkage_func([i], cluster, distance_matrix)
            if dist < threshold:
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    labels = np.zeros(doc_word_matrix.shape[0], dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for doc_idx in cluster:
            labels[doc_idx] = cluster_idx
    return labels


def agglomerative_clustering(doc_word_matrix, distance_matrix=None, linkage_type='single', metric='euclidean', k=3):
    if linkage_type in ['single', 'complete', 'average'] and metric in ['euclidean', 'cosine']:
        if metric == 'cosine':
            distance_matrix = pairwise_distances(
                doc_word_matrix, metric='cosine')
            model = AgglomerativeClustering(
                n_clusters=k, metric='precomputed', linkage=linkage_type)
            return model.fit_predict(distance_matrix)
        else:
            model = AgglomerativeClustering(
                n_clusters=k, metric='euclidean', linkage=linkage_type)
            return model.fit_predict(doc_word_matrix)
    else:
        raise ValueError(
            "Unsupported configuration. Use 'single', 'complete' or 'average' with euclidean/cosine.")
