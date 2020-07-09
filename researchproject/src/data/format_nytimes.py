import os

from researchproject.src import NY_FOLDER
import json
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

"""Gets raw NYTimes JSON data and exports a weighted adj matrix of company relations."""

def get_entity_list():
    """Returns a list of all organisations mentioned in the store of NYTimes articles."""
    orgs = []
    for fn in NY_FOLDER.iterdir():
        data = json.load(open(fn, 'r'))
        data = data['response']['docs']
        for item in data:
            keywords = item['keywords']
            for kw in keywords:
                if kw['name'] == 'organizations':
                    orgs.append(kw['value'])

    c = Counter(orgs)
    df = pd.DataFrame(c.items())
    df.columns = ["name", "count"]
    df.to_csv("entity_list.csv")

def format_ny(file):
    data = json.load(file)
    data = data['response']['docs']
    print("# of records", len(data))

    relevant_data = []
    for item in data:
        keywords = item['keywords']
        for kw in keywords:
            if kw['name'] == 'organizations':
                relevant_data.append(item)
                break
    print("# of relevant records", len(relevant_data))
    the_list = []
    for item in relevant_data:
        keywords = item['keywords']
        for kw in keywords:
            if kw['name'] == 'organizations':
                the_list.append(kw['value'])
    c = Counter(the_list)
    print(len(list(c.keys())))
    print(c.keys())


if __name__ == "__main__":
    get_entity_list()