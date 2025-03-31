from flask import Flask, render_template, request
from transform import *
from data_processing import *

from dotenv import load_dotenv

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    try:
        with open("PMIDs_list.txt") as f:
            pmid_list = list(map(int, f.read().splitlines()))
    except FileNotFoundError or FileExistsError:
        pmid_list = []

    if request.method == 'POST':
        pmids = request.form["pmids"]
        pmid_list += [int(pmid.strip()) for pmid in pmids.split(",")]

    load_dotenv()

    num_clusters = len(pmid_list)
    print("Getting GEO numeric IDs")
    geo_ids = get_geo_datasets(pmid_list)
    flattened = [(pmid, gse_id) for pmid, gse_list in geo_ids for gse_id in gse_list]
    text_fields = fetch_geo_essumary(flattened)
    print(text_fields)
    print("Clustering based on TF-IDF")
    df, cluster_labels, tsne_results = construct_df_idf(text_fields, num_clusters=num_clusters)
    px_graph = visualize(df)

    gse_data = []
    for pmid in pmid_list:

        # Find the dataset corresponding to the PMID
        datasets = [d for d in text_fields if d["pmid"] == pmid]
        if not datasets:
            print(f"No dataset found for {pmid}!")
            continue

        for dataset in datasets:
            dataset_title = dataset["title"]
            gse_data.append({"pmid": pmid, "dataset_title": dataset_title})

    return render_template('index.html',
                           gse_data=gse_data,
                           geo_ids=geo_ids,
                           text_fields=text_fields,
                           px_graph=px_graph,
                           )


if __name__ == '__main__':
    app.run(debug=True)
