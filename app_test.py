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

nltk.download('punkt_tab')
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


def predict_category_naive_bayes(new_text, doc_word_df, doc_category_df):
    """
    Класичний наївний баєсівський класифікатор на основі частот термів у категоріях.
    """
    # 1. Отримання словника
    vocab = list(doc_word_df.columns)

    # 2. Обробка нового документа
    tokens = nltk.word_tokenize(new_text)
    clean = cleanup_text(tokens)
    stemmed = stemming(clean)

    # 3. Побудова частот вхідного документа
    doc_vec = np.zeros(len(vocab), dtype=int)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    for word in stemmed:
        if word in vocab_index:
            doc_vec[vocab_index[word]] += 1

    # 4. Побудова терм-категорія матриці
    term_category_matrix = get_term_category_matrix(
        doc_word_df, doc_category_df)
    term_category_matrix += 1  # псевдорахунок для уникнення нулів
    prob_w_given_c = term_category_matrix.divide(
        term_category_matrix.sum(axis=0), axis=1)

    # 5. Логарифм апріорних ймовірностей P(c)
    D = len(doc_category_df)
    Dc = doc_category_df.sum(axis=0)
    log_priors = np.log(Dc / D)

    # 6. Обчислення log P(c) + ∑ n(w) * log P(w|c)
    scores = {}
    for category in doc_category_df.columns:
        log_likelihoods = np.log(prob_w_given_c[category].values)
        score = log_priors[category] + np.dot(doc_vec, log_likelihoods)
        scores[category] = score

    predicted_category = max(scores, key=scores.get)
    return predicted_category, scores


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
    tf_df = pd.DataFrame(compute_tf(doc_word_matrix, all_terms),
                         index=filenames, columns=unique_terms)
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

app.layout = html.Div([
    html.H1("Text Matrix Visualizer"),

    html.Label("Select Dataset Folder:"),
    dcc.Dropdown(
        id="folder-dropdown",
        options=[
            {"label": "Dataset A", "value": "texts"},
            {"label": "Dataset B", "value": "texts_2"},
        ],
        value="texts"
    ),

    html.Div(id="output-container")
])


@app.callback(
    Output("output-container", "children"),
    Input("folder-dropdown", "value")
)
def update_output(selected_folder):
    # Your function should take the parameter:
    doc_word_df, term_term_df, tf_df, tfidf_df, doc_category_df, tf_slf_df, \
        euclidean_dist_df, cosine_sim_df, filenames = process_texts_from_folder(
            selected_folder)

    test_texts = [
        "doctors prescribed medicines in hospitals and nurses provided therapies",
        "doctors diagnosed conditions in hospitals and provided medicine to patients",
        "The government has announced a new policy aimed at reducing taxes for small businesses",
    ]
    cat_scores_dfs = []
    predicted_categories = []
    for text in test_texts:
        pred_cat, cat_scores = predict_category_naive_bayes(
            text, doc_word_df, doc_category_df)
        cat_scores_dfs.append(pd.DataFrame(
            list(cat_scores.items()), columns=['Category', 'Score']))
        predicted_categories.append(pred_cat)

    print("Predicted category:", pred_cat)
    print("Scores:", cat_scores)

    euclidean_pca = apply_pca(euclidean_dist_df.values)
    cosine_pca = apply_pca(cosine_sim_df.values)

    euclidean_pca_df = pd.DataFrame(euclidean_pca, columns=["PCA1", "PCA2"])
    euclidean_pca_df["Document"] = filenames
    euclidean_pca_df["Category"] = [doc_category_df.loc[f].idxmax()
                                    for f in filenames]

    cosine_pca_df = pd.DataFrame(cosine_pca, columns=["PCA1", "PCA2"])
    cosine_pca_df["Document"] = filenames
    cosine_pca_df["Category"] = [doc_category_df.loc[f].idxmax()
                                 for f in filenames]

    def make_table(title, df, index_name):
        df_display = df.copy()
        df_display.insert(0, index_name, df_display.index)
        return html.Div([
            html.H3(title),
            dash_table.DataTable(
                data=df_display.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_display.columns],
                style_table={"overflowX": "auto"},
                page_size=10
            )
        ])

    def showBayesPredictionResult(test_text, predicted_category, cat_scores_df):
        return html.Div([
            html.Label(f"Tested text: {test_text}"),
            html.Hr(),
            html.Label(f"Predicted category: {predicted_category}"),
            dash_table.DataTable(
                data=cat_scores_df.to_dict('records')
            )
        ])

    # Generate results for Bayes predictions
    bayes_results = [
        showBayesPredictionResult(
            test_texts[i], predicted_categories[i], cat_scores_dfs[i])
        for i in range(len(test_texts))
    ]

    return [
        *bayes_results,
        make_table("Document-Word Matrix", doc_word_df, "Document"),
        make_table("Term-Term Matrix", term_term_df, "Term"),
        make_table("TF Matrix", tf_df.round(3), "Document"),
        make_table("TF-IDF Matrix", tfidf_df.round(3), "Document"),
        make_table("Document-Category Matrix", doc_category_df, "Document"),
        make_table("TF-SLF Matrix", tf_slf_df.round(3), "Document"),
        make_table("Euclidean Distance Matrix",
                   euclidean_dist_df.round(3), "Document"),
        make_table("Cosine Similarity Matrix",
                   cosine_sim_df.round(3), "Document"),

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
        )
    ]


if __name__ == '__main__':
    app.run(debug=True)
