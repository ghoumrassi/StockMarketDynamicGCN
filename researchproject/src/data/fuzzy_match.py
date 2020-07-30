"""
Takes in a list of organisations and maps them to a "true" organisation name as stated in
Wikidata, or returns (not listed) if not sufficiently close enough.
"""
import pickle

from researchproject.src import FUZZY_STOP_FILE, FM_OUTPUT
from ftfy import fix_text
import pandas as pd
import re

import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def clean(x: str):
    """Cleans string of unnecessary formatting."""
    x = fix_text(x)
    x = x.encode('ascii', errors='ignore')
    x = x.decode()
    x = x.lower()
    x = x.translate(str.maketrans('', '', string.punctuation))
    return x


def hex_iterator(start):
    while True:
        char = chr(start)
        yield char
        start += 1


def pad_clean(name_list, min_length, repl_dict=None):
    clean_list = []
    for name in name_list:
        new_name = clean(name)
        if repl_dict:
            for key in repl_dict:
                new_name = re.sub(r"\b" + key + r"\b", repl_dict[key], new_name)
        if not new_name.replace(" ", ""):
            new_name = "????????"
        name_len = len(new_name)
        if name_len < min_length:
            padding = " " * (min_length - name_len)
            new_name += padding
        clean_list.append(new_name)
    return clean_list


class FuzzyMatcher:
    """
    Used to create a mapping between two lists of entities that may not match exactly.

    fuzzy_names:  Fuzzy names of potential entities to be matched to actual entities.
    true_names: Actual names of entities to match with.
    """

    def __init__(self, fuzzy_names, true_names):
        # Creates vector repr based on mono-, bi- and tri-grams
        self.ngram_lb, self.ngram_ub = (2, 4)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.ngram_lb, self.ngram_ub))
        self.fuzzy_names = fuzzy_names
        self.true_names = true_names
        self.nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)  # Only matches nearest entry

        # Make term replacement dictionary
        rare_char_gen = hex_iterator(2048)
        with open(FUZZY_STOP_FILE, 'r') as fn:
            stop_terms = fn.read().split("\n")
        self.stop_term_dict = {stop_term: next(rare_char_gen) for stop_term in stop_terms}

    def match(self):
        """Returns a DataFrame of the predicted link between the two entries and a confidence score."""
        tfidf = self.vectorizer.fit_transform(
            pad_clean(self.true_names, self.ngram_lb, repl_dict=self.stop_term_dict)
        )  # Fit vectorizer and transform true names to vector form
        self.nn.fit(tfidf)  # Fit KNN model

        # Return the distance from, and index of, the closest true name to each fuzzy name
        fuzzy_names_vectorized = self.vectorizer.transform(
            pad_clean(self.fuzzy_names, self.ngram_lb, repl_dict=self.stop_term_dict)
        )
        fn_active_units = fuzzy_names_vectorized.getnnz(axis=1)
        dist, idx = self.nn.kneighbors(fuzzy_names_vectorized)

        # creates output in form [Distance from true name, Predicted true name, Fuzzy name]
        output_mapping = []
        for i, j in enumerate(idx):
            output_mapping.append([dist[i][0], self.true_names[j[0]], self.fuzzy_names[i], fn_active_units[i]])
        output_df = pd.DataFrame(output_mapping, columns=['distance', 'predicted', 'original', 'components'])
        return output_df


def make_fm_mapping(fuzzy_names, true_names, file_name, threshold=0.6):
    matcher = FuzzyMatcher(fuzzy_names, true_names)
    fm_results = matcher.match()
    # fm_results.to_csv((FM_OUTPUT / f"{file_name}_results.csv"))
    mapper = fm_results[fm_results['distance'] < threshold].set_index('original')['predicted'].to_dict()

    # Make sure that for all values there is a key that is exactly the same as it's value
    for v in list(mapper.values()):
        if v not in mapper:
            mapper[v] = v

    with open((FM_OUTPUT / f"{file_name}_mapper.p"), 'wb') as fn:
        pickle.dump(mapper, fn)


if __name__ == "__main__":
    # Load NY Times unique entity names from pre-loaded list
    pass
