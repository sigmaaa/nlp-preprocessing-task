import numpy as np
from numpy.linalg import norm


def compute_euclidean_distance(doc_word_matrix):
    num_docs = doc_word_matrix.shape[0]
    euclidean_dist = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(i, num_docs):
            distance = norm(doc_word_matrix[i] - doc_word_matrix[j])
            euclidean_dist[i, j] = distance
            euclidean_dist[j, i] = distance

    return euclidean_dist


def compute_cosine_similarity(doc_word_matrix):
    num_docs = doc_word_matrix.shape[0]
    cosine_sim = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(i, num_docs):
            dot_product = np.dot(doc_word_matrix[i], doc_word_matrix[j])
            norm_i = norm(doc_word_matrix[i])
            norm_j = norm(doc_word_matrix[j])
            cosine_sim[i, j] = 1 - dot_product / \
                (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0
            cosine_sim[j, i] = cosine_sim[i, j]

    return cosine_sim


def compute_kl_divergence(doc_word_matrix):

    epsilon = 1e-10  # to avoid log(0)
    num_docs = doc_word_matrix.shape[0]
    kl_div = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        p = doc_word_matrix[i] + epsilon
        p = p / p.sum()
        for j in range(i, num_docs):
            q = doc_word_matrix[j] + epsilon
            q = q / q.sum()
            kl_pq = np.sum(p * np.log(p / q))
            kl_qp = np.sum(q * np.log(q / p))
            avg_kl = 0.5 * (kl_pq + kl_qp)
            kl_div[i, j] = avg_kl
            kl_div[j, i] = avg_kl

    return kl_div


def compute_jaccard_similarity(doc_word_matrix):
    num_docs = doc_word_matrix.shape[0]
    jaccard_sim = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        set_i = doc_word_matrix[i] > 0
        for j in range(i, num_docs):
            set_j = doc_word_matrix[j] > 0
            intersection = np.logical_and(set_i, set_j).sum()
            union = np.logical_or(set_i, set_j).sum()
            similarity = 1 - intersection / union if union > 0 else 0
            jaccard_sim[i, j] = similarity
            jaccard_sim[j, i] = similarity

    return jaccard_sim
