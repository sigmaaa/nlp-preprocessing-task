import pandas as pd
import numpy as np


def get_term_term_freq_matrix(doc_word_matrix):
    return doc_word_matrix.T @ doc_word_matrix


def get_doc_word_matrix(docs, vocab):
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    doc_word_matrix = np.zeros((len(docs), len(vocab)), dtype=int)

    for doc_idx, doc in enumerate(docs):
        for word in doc:
            if word in vocab_index:
                word_idx = vocab_index[word]
                doc_word_matrix[doc_idx, word_idx] += 1

    return doc_word_matrix


def compute_tf(doc_word_matrix, docs):
    tf_matrix = np.zeros_like(doc_word_matrix, dtype=float)
    for i in range(len(doc_word_matrix)):
        for j in range(len(doc_word_matrix[i])):
            tf_matrix[i][j] = doc_word_matrix[i][j] / \
                len(docs[i]) if len(docs[i]) > 0 else 0
    return tf_matrix


def compute_idf(doc_word_matrix):
    N = doc_word_matrix.shape[0]
    df = np.count_nonzero(doc_word_matrix > 0, axis=0)
    return np.log(N / (df + 1))


def compute_tfidf(doc_word_matrix, docs):
    tf = compute_tf(doc_word_matrix, docs)
    idf = compute_idf(doc_word_matrix)
    return tf * idf


def compute_tf_slf(tf_matrix, slf_series):
    tf_slf_matrix = tf_matrix * slf_series.values
    return pd.DataFrame(tf_slf_matrix, index=tf_matrix.index, columns=tf_matrix.columns)


def get_term_category_matrix(doc_word_df, doc_category_df):
    term_category_matrix = doc_word_df.T @ doc_category_df
    return term_category_matrix


def compute_slf(term_category_matrix, doc_category_df):
    # Перетворення one-hot в Series з категоріями
    doc_categories = doc_category_df.idxmax(axis=1)
    categories = term_category_matrix.columns
    Nc = len(categories)

    # Кількість документів у кожній категорії
    docs_per_category = doc_categories.value_counts().reindex(categories).fillna(0)

    # NDF_tc: нормалізована частота терма в категорії
    NDF_tc = term_category_matrix.divide(docs_per_category, axis=1).fillna(0)

    # Rt: сума NDF_tc по всіх категоріях
    R_t = NDF_tc.sum(axis=1)
    # SLF_t: логарифм (Nc / Rt)
    with np.errstate(divide='ignore'):
        SLF_t = np.log(Nc / R_t.replace(0, np.nan))

    return SLF_t.fillna(0)
