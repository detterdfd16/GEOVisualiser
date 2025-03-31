# GEO Visualizer Flask App

A Flask-based web application for visualizing GEO datasets based on PMIDs. 

The app fetches metadata using EUtilities(ELink -> ESummary), clusters datasets using K-means on TF-IDF, and provides a t-SNE visual representation of high-dimensional data via Plotly.

### Example graph for provided PMIDs

![example_graph](static/newplot%20(1).png)

Clusters are marked via same marker color, dotted lines between them represent datasets linked to same PMIDs

## Prerequisites

- Python 3 or higher
- Internet access to download dependencies and access NCBI API

## Setup Instructions

1. **Clone the repository**

   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/geo-visualizer.git
   cd geo-visualizer
2. **Install requirements**
   Install requirements from `requirements.txt`
   ```bash
   pip install -r requirements.txt
3. **Setup the NCBI API key**

   For enhanced performance get your API key from [NCBI](https://www.ncbi.nlm.nih.gov/account/)
   ```bash
   NCBI_API_KEY=YOUR_API_KEY_HERE
4. **Run Flask**
   
   Start flask aplication via `flask run` in cloned directory.
5. **Open the application**
   
   Open the ip provided by Flask(default is http://127.0.0.1:5000/) in a web browser(Chrome/Firefox etc)
   
   On the website you can fill "Enter PMID..." field to specify PMIDs for visualisation

   By default, PMIDs from `PMIDs_list.txt` are inserted in visualisation. You can clear PMIDs if needed