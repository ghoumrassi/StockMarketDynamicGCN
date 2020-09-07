from pandas import json_normalize
import requests
import time
from tqdm import tqdm
from researchproject.src import WD_QUERIES, WD_OUTPUT
from ftfy import fix_text


def get_query_data(query):
    wd = "https://query.wikidata.org/sparql"
    r = requests.get(wd, params={'format': 'json', 'query': query})
    results = r.json(strict=False)
    df = json_normalize(results['results']['bindings'])

    # Remove junk columns
    columns = [col for col in df.columns if col.split('.')[-1] == 'value']
    df = df[columns]
    df.columns = [col.split('.')[0] for col in df.columns]

    time.sleep(5)
    return df


def run_wd_queries(test=False):
    pbar = tqdm(WD_QUERIES.iterdir())
    for i, query_file in enumerate(pbar):
        if i == 3:

            pbar.set_description(query_file.stem)
            with open(query_file, 'r') as f:
                query = f.read()
            if test:
                query += "\nLIMIT 50"
            output = get_query_data(query)
            output.to_csv((WD_OUTPUT / (query_file.stem + ".csv")), index=False)


if __name__ == "__main__":
    run_wd_queries(test=False)
