import re
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
from numpy.linalg import norm
from sklearn.decomposition import PCA
import plotly.express as px
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

# ======= NLP FUNCTIONS =========


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
        [word for word in clean_arr if word and word not in stop_words])
    return clean_arr


def clean_string(s):
    s = s.lower()
    return re.sub(r'[^a-zA-Z]+', '', s) if isinstance(s, str) else s


def stemming(text_array):
    ps = PorterStemmer()
    return np.array([ps.stem(word) for word in text_array])


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

def compute_euclidean_distance(doc_word_matrix):
    num_docs = doc_word_matrix.shape[0]
    euclidean_dist = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(i, num_docs):
            distance = norm(doc_word_matrix[i] - doc_word_matrix[j])
            euclidean_dist[i, j] = distance
            euclidean_dist[j, i] = distance  # Symmetric matrix

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
            cosine_sim[j, i] = cosine_sim[i, j]  # Symmetric matrix

    return cosine_sim


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


# ====== LOAD FILES FROM FOLDER =======


def process_texts_from_folder(directory='./texts/'):
    all_terms = []
    filenames = []
    categories = []

    for root, dirs, files in os.walk(directory):
        for filename in sorted(files):
            if filename.endswith('.txt'):
                category = os.path.basename(root)
                file_path = os.path.join(root, filename)
                try:
                    file_data = np.genfromtxt(
                        file_path, delimiter=' ', dtype=str)
                    extracted_terms = term_extraction(file_data)
                    all_terms.append(extracted_terms)
                    filenames.append(filename)
                    categories.append(category)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    if not all_terms:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    flat_terms = [term for sublist in all_terms for term in sublist]
    unique_terms = np.array(sorted(set(flat_terms)))
    doc_word_matrix = get_doc_word_matrix(all_terms, unique_terms)

    doc_word_df = pd.DataFrame(
        doc_word_matrix, index=filenames, columns=unique_terms)
    term_term_df = pd.DataFrame(get_term_term_freq_matrix(doc_word_matrix),
                                index=unique_terms, columns=unique_terms)
    tf_df = pd.DataFrame(compute_tf(doc_word_matrix, all_terms), index=filenames, columns=unique_terms)
    tfidf_df = pd.DataFrame(compute_tfidf(doc_word_matrix, all_terms),
                            index=filenames, columns=unique_terms)
    doc_category_df = pd.DataFrame(
        0, index=filenames, columns=sorted(set(categories)))
    for fname, cat in zip(filenames, categories):
        doc_category_df.loc[fname, cat] = 1

    term_cat_matrix = get_term_category_matrix(doc_word_df, doc_category_df)
    slf_series = compute_slf(term_cat_matrix, doc_category_df)

    # TF-SLF
    tf_slf_df = compute_tf_slf(tf_df, slf_series)

    # ---- EUCLIDEAN DISTANCE MATRIX ----
    euclidean_dist_df = pd.DataFrame(
        compute_euclidean_distance(doc_word_matrix), index=filenames, columns=filenames)

    # ---- COSINE SIMILARITY MATRIX ----
    cosine_sim_df = pd.DataFrame(
        compute_cosine_similarity(doc_word_matrix), index=filenames, columns=filenames)

    return doc_word_df, term_term_df, tf_df,  tfidf_df, doc_category_df, tf_slf_df, euclidean_dist_df, cosine_sim_df, filenames


def apply_pca(matrix):
    pca = PCA(n_components=2)
    return pca.fit_transform(matrix)

# ====== DASH APP =======


app = Dash(__name__)
app.title = "TF-IDF & Matrix Viewer"

doc_word_df, term_term_df, tf_df, tfidf_df, doc_category_df, tf_slf_df, euclidean_dist_df, cosine_sim_df, filenames = process_texts_from_folder()
euclidean_pca = apply_pca(euclidean_dist_df.values)
cosine_pca = apply_pca(cosine_sim_df.values)

euclidean_pca_df = pd.DataFrame(euclidean_pca, columns=["PCA1", "PCA2"])
euclidean_pca_df["Document"] = filenames
euclidean_pca_df["Category"] = [doc_category_df.loc[f, :].idxmax()
                                for f in filenames]

cosine_pca_df = pd.DataFrame(cosine_pca, columns=["PCA1", "PCA2"])
cosine_pca_df["Document"] = filenames
cosine_pca_df["Category"] = [doc_category_df.loc[f, :].idxmax()
                             for f in filenames]
print(cosine_pca_df)

doc_word_display = doc_word_df.copy()
doc_word_display.insert(0, "Document", doc_word_display.index)

term_term_display = term_term_df.copy()
term_term_display.insert(0, "Document", term_term_display.index)

tf_display = tf_df.round(3).copy()
tf_display.insert(0, "Document", tf_display.index)

tfidf_display = tfidf_df.round(3).copy()
tfidf_display.insert(0, "Document", tfidf_display.index)

doc_category_display = doc_category_df.copy()
doc_category_display.insert(0, "Document", doc_category_display.index)

tf_slf_display = tf_slf_df.round(3).copy()
tf_slf_display.insert(0, "Document", tf_slf_display.index)

euclidean_display = euclidean_dist_df.round(3).copy()
euclidean_display.insert(0, "Document", euclidean_display.index)

cosine_display = cosine_sim_df.round(3).copy()
cosine_display.insert(0, "Document", cosine_display.index)

app.layout = html.Div([
    html.H1("Text Matrix Visualizer"),

    html.H3("Document-Word Matrix"),
    dash_table.DataTable(
        data=doc_word_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in doc_word_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("Term-Term Frequency Matrix"),
    dash_table.DataTable(
        data=term_term_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in term_term_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("TF Matrix"),
    dash_table.DataTable(
        data=tf_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in tf_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("TF-IDF Matrix"),
    dash_table.DataTable(
        data=tfidf_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in tfidf_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("Document-Category Matrix"),
    dash_table.DataTable(
        data=doc_category_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in doc_category_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("TF-SLF Matrix"),
    dash_table.DataTable(
        data=tf_slf_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in tf_slf_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("Euclidean Distance Matrix"),
    dash_table.DataTable(
        data=euclidean_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in euclidean_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("Cosine Similarity Matrix"),
    dash_table.DataTable(
        data=cosine_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in cosine_display.columns],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),

    html.H3("PCA Projection of Euclidean Distances"),
    dcc.Graph(
        figure=px.scatter(
            euclidean_pca_df,
            x="PCA1",
            y="PCA2",
            color="Category",
            hover_data=["Document"],
            title="Euclidean Similarity PCA"
        )
    ),

    html.H3("PCA Projection of Cosine Similarities"),
    dcc.Graph(
        figure=px.scatter(
            cosine_pca_df,
            x="PCA1",
            y="PCA2",
            color="Category",
            hover_data=["Document"],
            title="Cosine Similarity PCA"
        )
    ),
])
if __name__ == '__main__':
    app.run(debug=True)
