import os
import xml.etree.ElementTree as ET
import requests


def get_geo_datasets(pmids: list[int]) -> list[(int, list[int])]:
    """
    Obtains all GEO Dataset IDs from a given PMIDs
    """
    api_key = os.getenv("NCBI_API_KEY")

    geo_ids = []
    for pmid in pmids:
        url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?'
               f'dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={pmid}&retmode=xml')
        if api_key:
            url += f'&api_key={api_key}'

        response = requests.get(url)

        if response.status_code != 200:
            print("error during getting GEO numeric IDs")

        root = ET.fromstring(response.text)

        geo_ids.append((pmid, [int(link.find("Id").text) for link in root.findall(".//Link")]))

    return geo_ids


def get_accession_ids(geo_idss: list[(int, list[int])]) -> list[(int, list[str])]:
    """
        Converts GEO numeric IDs to GEO Accession IDs
    """
    api_key = os.getenv("NCBI_API_KEY")
    result_ids = []
    for (pmid, geo_ids) in geo_idss:

        url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?'
               f'db=gds&id={','.join([str(geo_id) for geo_id in geo_ids])}&retmode=xml')
        if api_key:
            url += f'&api_key={api_key}'

        response = requests.get(url)
        if response.status_code != 200:
            print("error during conversion of GEO numeric IDs")

        root = ET.fromstring(response.text)
        accession_ids = []
        for item in root.findall(".//DocSum"):
            accession = item.find(".//Item[@Name='Accession']")
            if accession is not None:
                if not accession.text.startswith(
                        "GSE"):  # Getting only GSE IDs since only they have all required fields
                    continue
                accession_ids.append(accession.text)
        result_ids.append((pmid, accession_ids))
    return result_ids


def fetch_text_fields(accession_id: int) -> dict[str, str]:
    """
    Fetch text fields(Title, Experiment type, Summary, Organism, Overall design) for a provided accession ID
    """
    # Construct the URL for the GEO query service (this uses SOFT format)
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession_id}&targ=self&form=text&view=full"

    # Send the request to GEO
    response = requests.get(url)
    response.encoding = "utf-8"  # setting encoding utf-8 explicitly since some fields may contain non ASCII characters

    if response.status_code != 200:
        print(f"Error fetching data for {accession_id}: HTTP {response.status_code}")
        return {}

    # Initialize metadata fields
    metadata = {
        "title": "N/A",
        "summary": "N/A",
        "overall_design": "N/A",
        "experiment_type": "N/A",
        "organism": "N/A"
    }

    # Parse the SOFT format to extract key metadata
    lines = response.text.splitlines()
    for line in lines:
        if line.startswith("!Series_title"):
            metadata["title"] = line.split("= ", 1)[1]  # Extract value after "= "
        elif line.startswith("!Series_summary"):
            metadata["summary"] = line.split("= ", 1)[1]
        elif line.startswith("!Series_overall_design"):
            metadata["overall_design"] = line.split("= ", 1)[1]
        elif line.startswith("!Series_type"):
            metadata["experiment_type"] = line.split("= ", 1)[1]
        elif line.startswith("!Series_platform_organism") or line.startswith("!Series_sample_organism"):
            metadata["organism"] = line.split("= ", 1)[1]

    return metadata


def fetch_geo_essumary(geo_ids: list[(int, int)]) -> list[dict[str, str]]:
    api_key = os.getenv("NCBI_API_KEY")

    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?'
           f'db=gds&id={','.join([str(geo_id[1]) for geo_id in geo_ids])}&retmode=xml')
    if api_key:
        url += f'&api_key={api_key}'

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching summaries for {geo_ids}: HTTP {response.status_code}")

    root = ET.fromstring(response.text)

    datasets = []
    for index, docsum in enumerate(root.findall(".//DocSum")):
        print(docsum.find(".//Item[@Name='title']").text, docsum.find(".//Item[@Name='title']"))
        dataset = {
            "pmid": geo_ids[index][0],
            "title": docsum.find(".//Item[@Name='title']").text
            if docsum.find(".//Item[@Name='title']") is not None else "N/A",
            "summary": docsum.find(".//Item[@Name='summary']").text
            if docsum.find(".//Item[@Name='summary']") is not None else "N/A",
            "organism": docsum.find(".//Item[@Name='taxon']").text
            if docsum.find(".//Item[@Name='taxon']") is not None else "N/A",
            "experiment_type": docsum.find(".//Item[@Name='gdsType']").text
            if docsum.find(".//Item[@Name='gdsType']") is not None else "N/A",
        }
        datasets.append(dataset)
    return datasets


def fetch_all_text_fields(accession_ids: list[(int, list[str])]) -> list[dict[str, str]]:
    text_fields = []
    for (pmid, ids) in accession_ids:
        for id in ids:
            fields = fetch_text_fields(id)
            fields["pmid"] = pmid
            text_fields.append(fields)

    return text_fields
