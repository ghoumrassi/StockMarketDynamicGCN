from ftfy import fix_text
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from researchproject.src import COMPANY_DATA

""" Takes in a list of organisations from NYTImes and maps them to a true NYSE organisation name, as stated in 
Tiingo, or returns (not listed) if not sufficiently close enough."""


def ngrams(x: str, n=3):
    """Returns n-gram representation of string."""
    x = clean(x)
    ngrams = zip(*[x[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def clean(x: str):
    """Cleans string of unnecessary formatting."""
    x = fix_text(x)
    x = x.encode('ascii', errors='ignore').decode()
    x = x.lower()
    return x


class FuzzyMatcher:
    """
    Used to create a mapping between two lists of entities that may not match exactly.

    entity_list: Actual names of entities to match with.
    match_list:  Fuzzy names of potential entities to be matched to actual entities.
    """
    def __init__(self, match_list, entity_list):
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
        self.match_list = match_list
        self.entity_list = entity_list
        self.nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)

    def match(self):
        """Returns a dataframe of the predicted link between the two entries and a confidence score."""
        tfidf = self.vectorizer.fit_transform(self.entity_list)
        self.nn.fit(tfidf)
        dist, idx = self._get_nearest(self.match_list)
        output_mapping = []
        for i, j in enumerate(idx):
            output_mapping.append([dist[i][0], self.entity_list[j[0]], self.match_list[i]])
        output_df = pd.DataFrame(output_mapping, columns=['distance', 'pred_entity', 'orig_entity'])
        output_df['len_diff'] = (
                output_df['pred_entity'].str.len() - output_df['orig_entity'].apply(clean).str.len()
        ).abs()
        output_df['dist_len'] = output_df['distance'] / output_df['pred_entity'].str.len()

        return output_df

    def _get_nearest(self, query):
        return self.nn.kneighbors(self.vectorizer.transform(query))


if __name__ == "__main__":
    nyt_entities = pd.read_csv('out.csv')
    entity_list = list(nyt_entities['name'].dropna().unique())

    company_data = pd.read_csv(COMPANY_DATA)
    company_list = list(company_data['Name'].values)

    matcher = FuzzyMatcher(entity_list, company_list)
    output = matcher.match()
    print(output)
    output_joined = pd.merge(
        nyt_entities, output[output['distance'] < 0.8], left_on='name', right_on='pred_entity', how='inner'
    )
    output_joined.to_csv('temporary_output.csv')
