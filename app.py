import os
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


def term_extraction(text_array):
    clean_array = cleanup_text(text_array)
    return stemming(clean_array)


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


def cleanup_text(text_array):
    clean_arr = np.array([clean_string(word) for word in text_array])
    stop_words = set(stopwords.words('english'))
    clean_arr = np.array(
        [word for word in clean_arr if word not in stop_words])
    return clean_arr


def clean_string(s):
    s = s.lower()
    if isinstance(s, str):
        return re.sub(r'[^a-zA-Z]+', '', s)
    return s


def stemming(text_array):
    ps = PorterStemmer()
    stemmed_array = np.array([ps.stem(word)
                             for word in text_array])

    return stemmed_array


def compute_tf(doc_word_matrix, docs):
    tf_matrix = np.zeros_like(doc_word_matrix, dtype=float)
    for i in range(len(doc_word_matrix)):
        for j in range(len(doc_word_matrix[i])):
            tf_matrix[i][j] = doc_word_matrix[i][j]/len(docs[i])
    return tf_matrix


def compute_tfidf(tf, doc_word_matrix):
    N = doc_word_matrix.shape[0]  # кількість документів

    # DF — кількість документів, в яких зустрічається кожен термін
    df = np.count_nonzero(doc_word_matrix > 0, axis=0)

    # IDF
    idf = np.log(N / (df + 1))  # додаємо 1, щоб уникнути ділення на 0

    # TF-IDF
    tfidf = tf * idf

    return tfidf


base_directory = './texts/'
all_terms = []

for root, dirs, files in os.walk(base_directory):
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(root, filename)

            file_data = np.genfromtxt(file_path, delimiter=' ', dtype=str)

            extracted_terms = term_extraction(file_data)
            all_terms.append(extracted_terms)
flat_terms = [term for sublist in all_terms for term in sublist]
unique_terms = np.array(list(set(flat_terms)))
unique_terms = np.sort(unique_terms)
doc_word_matrix = get_doc_word_matrix(all_terms, unique_terms)

print(pd.DataFrame(get_term_term_freq_matrix(doc_word_matrix),
      index=unique_terms, columns=unique_terms))
tf = compute_tf(doc_word_matrix, all_terms)
print(pd.DataFrame(compute_tfidf(tf, doc_word_matrix)))
