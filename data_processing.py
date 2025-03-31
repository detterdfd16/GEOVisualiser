import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score


def construct_df_idf(datasets: list[dict[str, str]], num_clusters: int = 2):
    # Convert the dataset into a pandas DataFrame
    df = pd.DataFrame(datasets)

    # Step 1: Combine fields using join
    # Convert all fields to strings and handle missing values
    df["combined_text"] = (df[["title", "summary", "organism", "experiment_type"]]
                           .fillna(" ")
                           .agg(" ".join, axis=1)
                           )

    # Step 2: TF-IDF Vectorization (compute over the whole dataset)
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\-?\w+\b')
    # Compute TF-IDF matrix
    X = vectorizer.fit_transform(df["combined_text"])

    # Step 3: Calculate Term Frequency (TF) and Inverse Document Frequency (IDF)
    tf_matrix = X.toarray()  # Term Frequency matrix
    idf_values = vectorizer.idf_  # Inverse Document Frequency values for each term

    # Step 4: Calculate total TF across the entire dataset (for each term)
    term_frequencies = np.sum(tf_matrix, axis=0)  # Sum over all documents for each term

    # Step 5: Calculate true TF-IDF for each term by multiplying TF and IDF
    tfidf_values = term_frequencies * idf_values

    # Step 6: Create a DataFrame with the term, TF, IDF, and TF-IDF values
    terms = vectorizer.get_feature_names_out()
    df_terms = pd.DataFrame({
        'Term': terms,
        'TF': term_frequencies,
        'IDF': idf_values,
        'TF-IDF': tfidf_values
    })

    # Sort by TF-IDF values (optional)
    df_terms = df_terms.sort_values(by='TF-IDF', ascending=False).reset_index(drop=True)

    print(df_terms)

    num_clusters = silhouette_method(X)
    # Step 3: Cluster the data using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Step 4: Reduce dimensions for visualization (using t-SNE)
    tsne = TSNE(n_components=2, random_state=42, perplexity=1)
    tsne_results = tsne.fit_transform(X.toarray())

    # Step 5: Create a new DataFrame to store the results (preserve original df)
    results_df = df.copy()  # Copy the original df to keep it intact
    results_df["cluster"] = kmeans.labels_  # Add cluster information
    results_df["tsne_x"] = tsne_results[:, 0]  # Add t-SNE X values
    results_df["tsne_y"] = tsne_results[:, 1]  # Add t-SNE Y values

    return results_df, kmeans.labels_, tsne_results

def silhouette_method(tfidf_matrix):
    """
    Use the Silhouette Score to determine the optimal number of clusters.
    tfidf_matrix: The matrix of TF-IDF vectors.
    """
    best_score = -1
    best_num_clusters = 2

    for n_clusters in range(2, 20):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_num_clusters = n_clusters

    return best_num_clusters

def visualize(df):
    # Create the scatter plot using Plotly
    fig = px.scatter(df,
                     x="tsne_x",
                     y="tsne_y",
                     color="cluster",
                     title="Dataset Clusters Based on TF-IDF Vectors",
                     labels={"tsne_x": "TSNE X", "tsne_y": "TSNE Y"},
                     hover_data={"pmid": True, "title": True},
                     )


    # Improved line color and hover consistency
    for pmid, group in df.groupby("pmid"):
        line_color = "black"
        if len(group) > 1:  # Only draw lines if multiple datasets share a PMID
            fig.add_trace(go.Scatter(
                x=group["tsne_x"],
                y=group["tsne_y"],
                mode="lines",
                line=dict(color=line_color, width=1, dash="dot"),  # Semi-transparent gray
                hoverinfo="none",  # Ensure hover info is visible
                name=f"PMID {pmid}",
                showlegend=False  # Hide from legend to reduce clutter
            ))

    fig.update_layout(width=800,
                      height=600,
                      plot_bgcolor='rgba(240, 240, 240, 1)',
                      paper_bgcolor='rgba(243, 244, 246, 1)',
                      title={
                          'x': 0.5,
                          'xanchor': 'center',
                      },
                      )

    # Return the figure's HTML representation
    return fig.to_html(full_html=False)


if __name__ == "__main__":
    datasets = [
        {
            "pmid": "1231232",
            "title": "In vivo molecular signatures of severe dengue infection revealed by viscRNA-Seq",
            "summary": "Dengue virus infection can result in severe symptoms including shock and hemorrhage...",
            "organism": "Homo sapiens",
            "experiment_type": "Expression profiling by high throughput sequencing",
            "overall_design": "Blood cells from dengue virus infected human patients were subjected to virus-inclusive single cell RNA-Seq."
        },
        {
            "pmid": "3345346645",
            "title": "Human LSD2/KDM1b/AOF1 regulates gene transcription by modulating intragenic H3K4me2 methylation",
            "summary": "Human LSD2/KDM1b/AOF1 regulates gene transcription by modulating intragenic H3K4me2 methylation...",
            "organism": "Homo sapiens",
            "experiment_type": "ChIP-chip",
            "overall_design": "ChIP-chip analysis of human cells treated with LSD2/KDM1b inhibitors."
        },
        # Add more datasets here...
    ]
    df, cluster_labels, tsne_results = construct_df_idf(datasets)
    print(df)
    print(visualize(df))
