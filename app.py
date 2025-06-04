import os
import numpy as np
import nltk
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
from sklearn.decomposition import PCA
import plotly.express as px
import ssl
from text_preprocessing import term_extraction
from text_representation import (
    get_doc_word_matrix, get_term_term_freq_matrix,
    compute_tf, compute_tfidf,
    compute_tf_slf, compute_slf,
    get_term_category_matrix
)
from document_similarity import (
    compute_cosine_similarity, compute_euclidean_distance,
    compute_kl_divergence, compute_jaccard_similarity
)
from clustering.document_clustering import (
    k_means_clustering,
    one_pass_clustering,
    agglomerative_clustering
)
from clustering.cluster_linkage import (
    single_linkage,
    complete_linkage,
    average_linkage
)

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

app = Dash(__name__)
app.title = "TF-IDF & Matrix Viewer"

available_folders = [d for d in os.listdir(
    "./") if os.path.isdir(d) and d.startswith("texts")]

app.layout = html.Div([
    html.H1("Text Matrix Visualizer with Clustering"),
    html.Label("Select folder:"),
    dcc.Dropdown(
        id="folder-dropdown",
        options=[{"label": name, "value": name} for name in available_folders],
        value=available_folders[0] if available_folders else None
    ),
    html.Div(id="output-container")
])


def process_texts_from_folder(directory):
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
        return [pd.DataFrame()] * 12

    flat_terms = [term for sublist in all_terms for term in sublist]
    unique_terms = np.array(sorted(set(flat_terms)))
    doc_word_matrix = get_doc_word_matrix(all_terms, unique_terms)

    doc_word_df = pd.DataFrame(
        doc_word_matrix, index=filenames, columns=unique_terms)
    term_term_df = pd.DataFrame(get_term_term_freq_matrix(
        doc_word_matrix), index=unique_terms, columns=unique_terms)
    tf_df = pd.DataFrame(compute_tf(doc_word_matrix, all_terms),
                         index=filenames, columns=unique_terms)
    tfidf_df = pd.DataFrame(compute_tfidf(
        doc_word_matrix, all_terms), index=filenames, columns=unique_terms)

    doc_category_df = pd.DataFrame(
        0, index=filenames, columns=sorted(set(categories)))
    for fname, cat in zip(filenames, categories):
        doc_category_df.loc[fname, cat] = 1

    term_cat_matrix = get_term_category_matrix(doc_word_df, doc_category_df)
    slf_series = compute_slf(term_cat_matrix, doc_category_df)
    tf_slf_df = compute_tf_slf(tf_df, slf_series)

    euclidean_dist_df = pd.DataFrame(compute_euclidean_distance(
        doc_word_matrix), index=filenames, columns=filenames)
    cosine_sim_df = pd.DataFrame(compute_cosine_similarity(
        doc_word_matrix), index=filenames, columns=filenames)
    kl_df = pd.DataFrame(compute_kl_divergence(
        doc_word_matrix), index=filenames, columns=filenames)
    jaccard_df = pd.DataFrame(compute_jaccard_similarity(
        doc_word_matrix), index=filenames, columns=filenames)

    return doc_word_df, term_term_df, tf_df, tfidf_df, doc_category_df, tf_slf_df, \
        euclidean_dist_df, cosine_sim_df, kl_df, jaccard_df, filenames, doc_word_matrix


def apply_pca(matrix):
    pca = PCA(n_components=2)
    return pca.fit_transform(matrix)


@app.callback(
    Output("output-container", "children"),
    Input("folder-dropdown", "value")
)
def update_output(selected_folder):
    doc_word_df, term_term_df, tf_df, tfidf_df, doc_category_df, tf_slf_df, \
        euclidean_dist_df, cosine_sim_df, kl_df, jaccard_df, filenames, doc_word_matrix = process_texts_from_folder(
            selected_folder)

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

    def project_and_plot(df_or_matrix, labels, title):
        projected = apply_pca(df_or_matrix)
        df = pd.DataFrame(projected, columns=["PCA1", "PCA2"])
        df["Document"] = filenames
        df["Cluster"] = [str(label) for label in labels]
        return dcc.Graph(figure=px.scatter(df, x="PCA1", y="PCA2", color="Cluster", hover_data=["Document"], title=title))

    # PCA by category
    euclidean_pca = apply_pca(euclidean_dist_df.values)
    euclidean_pca_df = pd.DataFrame(euclidean_pca, columns=["PCA1", "PCA2"])
    euclidean_pca_df["Document"] = filenames
    euclidean_pca_df["Category"] = [doc_category_df.loc[f, :].idxmax()
                                    for f in filenames]

    cosine_pca = apply_pca(cosine_sim_df.values)
    cosine_pca_df = pd.DataFrame(cosine_pca, columns=["PCA1", "PCA2"])
    cosine_pca_df["Document"] = filenames
    cosine_pca_df["Category"] = [doc_category_df.loc[f, :].idxmax()
                                 for f in filenames]

    # PCA before clustering
    reduced_doc_word_matrix = apply_pca(doc_word_matrix)
    reduced_doc_word_matrix_kmeans_eu_labels = k_means_clustering(
        reduced_doc_word_matrix, distance="euclidean", k=3)

    kmeans_eu_labels = k_means_clustering(
        doc_word_matrix, distance="euclidean", k=3)
    kmeans_cos_labels = k_means_clustering(
        doc_word_matrix, distance="cosine", k=3)

    onepass_eu_labels = one_pass_clustering(
        doc_word_matrix, euclidean_dist_df.values, single_linkage, threshold=3)
    onepass_cos_labels = one_pass_clustering(
        doc_word_matrix, cosine_sim_df.values, complete_linkage, threshold=0.9)

    agglom_eu_labels = agglomerative_clustering(
        doc_word_matrix, metric='euclidean', linkage_type='single', k=3)
    agglom_cos_labels = agglomerative_clustering(
        doc_word_matrix, metric='cosine', linkage_type='average', k=3)

    return [
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
        make_table("KL Divergence Matrix", kl_df.round(3), "Document"),
        make_table("Jaccard Similarity Matrix",
                   jaccard_df.round(3), "Document"),

        html.H3("PCA Projection of Euclidean Distances (by Category)"),
        dcc.Graph(figure=px.scatter(euclidean_pca_df, x="PCA1", y="PCA2",
                  color="Category", hover_data=["Document"], title="Euclidean Distance PCA")),

        html.H3("PCA Projection of Cosine Similarities (by Category)"),
        dcc.Graph(figure=px.scatter(cosine_pca_df, x="PCA1", y="PCA2",
                  color="Category", hover_data=["Document"], title="Cosine Similarity PCA")),

        html.H3("K-Means Clustering (Euclidean)"),
        project_and_plot(doc_word_matrix, kmeans_eu_labels,
                         "K-Means Clustering (Euclidean)"),

        html.H3("K-Means Clustering (Cosine)"),
        project_and_plot(doc_word_matrix, kmeans_cos_labels,
                         "K-Means Clustering (Cosine)"),

        html.H3("One-Pass Clustering (Euclidean + Single Linkage)"),
        project_and_plot(doc_word_matrix, onepass_eu_labels,
                         "One-Pass Clustering (Euclidean + Single Linkage)"),

        html.H3("One-Pass Clustering (Cosine + Complete Linkage)"),
        project_and_plot(doc_word_matrix, onepass_cos_labels,
                         "One-Pass Clustering (Cosine + Complete Linkage)"),

        html.H3("Agglomerative Clustering (Euclidean + Single Linkage)"),
        project_and_plot(doc_word_matrix, agglom_eu_labels,
                         "Agglomerative Clustering (Euclidean + Single Linkage)"),

        html.H3("Agglomerative Clustering (Cosine + Average Linkage)"),
        project_and_plot(doc_word_matrix, agglom_cos_labels,
                         "Agglomerative Clustering (Cosine + Average Linkage)"),

        html.H3("PCA Before Clustering (Raw Doc Matrix)"),
        project_and_plot(reduced_doc_word_matrix, reduced_doc_word_matrix_kmeans_eu_labels,
                         "K-Means Clustering on PCA (Euclidean)"),
        project_and_plot(reduced_doc_word_matrix, kmeans_cos_labels,
                         "K-Means Clustering on PCA (Cosine)")
    ]


if __name__ == '__main__':
    app.run(debug=True)
