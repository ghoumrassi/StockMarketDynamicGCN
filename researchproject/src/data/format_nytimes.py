import os

from researchproject.src import NY_FOLDER, COMPANY_NAME_MAPPING, NYT_OUTPUT

from collections import Counter
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import json
import numpy as np
import pickle
import datetime as dt


def get_entity_list(start_date=None):
    """Creates a dict of all organisations mentioned in the store of NYTimes articles and the # of times mentioned."""
    orgs = []
    if start_date:
        start_date = dt.datetime.strptime(start_date, '%d/%m/%Y')
    pbar = tqdm(NY_FOLDER.iterdir())
    for fn in pbar:
        pbar.set_description(f"Current file: {fn.stem}")
        if start_date:
            file_date = dt.datetime.strptime(fn.stem.replace("nytimes", ""), '%Y%d')
            if file_date < start_date:
                continue
        data = json.load(open(fn, 'r'))
        data = data['response']['docs']
        for item in data:
            keywords = item['keywords']
            for kw in keywords:
                if kw['name'] == 'organizations':
                    orgs.append(kw['value'])
    c = Counter(orgs)
    with open((NYT_OUTPUT / 'entities.p'), 'wb') as fn:
        pickle.dump(c, fn)


# def format_ny(file):
#     data = json.load(file)
#     data = data['response']['docs']
#     print("# of records", len(data))
#
#     relevant_data = []
#     for item in data:
#         keywords = item['keywords']
#         for kw in keywords:
#             if kw['name'] == 'organizations':
#                 relevant_data.append(item)
#                 break
#     print("# of relevant records", len(relevant_data))
#     the_list = []
#     for item in relevant_data:
#         keywords = item['keywords']
#         for kw in keywords:
#             if kw['name'] == 'organizations':
#                 the_list.append(kw['value'])
#     c = Counter(the_list)
#     print(len(list(c.keys())))
#     print(c.keys())

def get_official_names(mapper, name_list):
    names = []
    for name in name_list:
        if name in mapper:
            company_name = mapper[name]
            names.append(company_name)
    if len(names) > 0:
        return names
    else:
        return None


# class NYTFormatter:
#     def __init__(self):
#         self.data_folder = NY_FOLDER
#         self.num_files = len([f for f in self.data_folder.iterdir()])
#         self.article_summaries = None
#         self.frame = None
#         self.adj_matrix = None
#         self.org_list = None
#         self.org_idx_map = None
#         self.idx_org_map = None
#
#     def make_summaries(self, start_date=None):
#         if start_date:
#             start_date = dt.datetime.strptime(start_date, '%d/%m/%Y')
#         self.article_summaries = []
#         with tqdm(total=self.num_files) as pbar:
#             for i, fn in enumerate(self.data_folder.iterdir()):
#                 if start_date:
#                     file_date = dt.datetime.strptime(fn.stem.replace("nytimes", ""), '%Y%d')
#                     if file_date < start_date:
#                         continue
#                 data = json.load(open(fn, 'r'))
#                 data = data['response']['docs']
#                 for item in data:
#                     orgs = []
#                     section = item['section_name'] if "section_name" in item else None
#                     news_desk = item['news_desk'] if "news_desk" in item else None
#                     date = item['pub_date'].split("T")[0]
#                     keywords = item['keywords']
#                     for kw in keywords:
#                         if kw['name'] == 'organizations':
#                             orgs.append(kw['value'])
#                     self.article_summaries.append([date, section, news_desk, orgs])
#                 pbar.update()
#         df = pd.DataFrame(self.article_summaries, columns=['Date', 'Section', 'News Desk', 'Organisations'])
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")
#         df['Organisations'] = df['Organisations'].apply(get_official_names)
#         df = df[df['Organisations'].notna()]
#         df = df[df['Organisations'].apply(len) >= 2]
#         df['Pairs'] = df['Organisations'].apply(lambda x: list(combinations(x, 2)))
#         df = df.drop('Organisations', 1).explode('Pairs')
#
#     def make_frame(self):
#         try:
#             assert self.article_summaries
#         except AssertionError:
#             raise AssertionError("No article summaries were found. Run 'make_summaries' first'.")
#
#         df = pd.DataFrame(self.article_summaries, columns=['Date', 'Section', 'News Desk', 'Organisations'])
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")
#         df = df[df['Date'].dt.year >= 2000]
#         df['Organisations'] = df['Organisations'].apply(get_official_names)
#         df = df[df['Organisations'].notna()]
#         df = df[df['Organisations'].apply(len) >= 2]
#         df['Pairs'] = df['Organisations'].apply(lambda x: list(combinations(x, 2)))
#         df = df.drop('Organisations', 1).explode('Pairs')
#         self.frame = df
#         self.frame.to_csv(NYT_SUMMARIES, index=False)
#
#     def make_mappings(self):
#         unique_pairs = self.frame['Pairs'].unique()
#         self.org_list = list(set(org for orgs in unique_pairs for org in orgs))
#         self.org_idx_map = {org: i for i, org in enumerate(self.org_list)}
#         self.idx_org_map = {i: org for org, i in self.org_idx_map.items()}
#
#     def make_adj_matrix(self, persistence=30):
#         min_date = self.frame['Date'].min()
#         max_date = self.frame['Date'].max()
#         time_period = (max_date - min_date).days + 1
#         num_orgs = len(self.org_list)
#         adj_matrix = np.zeros((time_period, num_orgs, num_orgs))
#         for i, row in self.frame.iterrows():
#             date_index = (row['Date'] - min_date).days
#             date_indices = np.array(range(date_index, date_index + persistence))
#             date_indices = date_indices[date_indices < time_period]
#             pair = row['Pairs']
#             item_1_idx = self.org_idx_map[pair[0]]
#             item_2_idx = self.org_idx_map[pair[1]]
#             adj_matrix[date_indices, item_1_idx, item_2_idx] += 1
#             adj_matrix[date_indices, item_2_idx, item_1_idx] += 1
#         self.adj_matrix = adj_matrix


def make_summaries(mapper, output_file, start_date=None):
    if start_date:
        start_date = dt.datetime.strptime(start_date, '%d/%m/%Y')
    num_files = len([f for f in NY_FOLDER.iterdir()])
    article_summaries = []
    with tqdm(total=num_files) as pbar:
        for i, fn in enumerate(NY_FOLDER.iterdir()):
            if start_date:
                file_date = dt.datetime.strptime(fn.stem.replace("nytimes", ""), '%Y%d')
                if file_date < start_date:
                    continue
            data = json.load(open(fn, 'r'))
            data = data['response']['docs']
            for item in data:
                orgs = []
                section = item['section_name'] if "section_name" in item else None
                news_desk = item['news_desk'] if "news_desk" in item else None
                date = item['pub_date'].split("T")[0]
                keywords = item['keywords']
                for kw in keywords:
                    if kw['name'] == 'organizations':
                        orgs.append(kw['value'])
                article_summaries.append([date, section, news_desk, orgs])
            pbar.update()
    df = pd.DataFrame(article_summaries, columns=['Date', 'Section', 'News Desk', 'Organisations'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")
    df['Organisations'] = df['Organisations'].apply(lambda x: get_official_names(mapper, x))
    df = df[df['Organisations'].notna()]
    df = df[df['Organisations'].apply(len) >= 2]
    df['Pairs'] = df['Organisations'].apply(lambda x: list(combinations(x, 2)))
    df = df.drop('Organisations', 1).explode('Pairs')
    df['Company A'] = df['Pairs'].apply(lambda x: x[0])
    df['Company B'] = df['Pairs'].apply(lambda x: x[1])
    df.drop('Pairs', 1, inplace=True)

    df.to_csv((NYT_OUTPUT / output_file), index=False)


if __name__ == "__main__":
    # formatter = NYTFormatter(NY_FOLDER)
    # formatter.make_summaries()
    # formatter.make_frame()
    # formatter.make_mappings()
    # formatter.make_adj_matrix()
    # formatter.adj_matrix.save(NYT_ADJ_MATRIX)
    # adj = np.sum(formatter.adj_matrix, axis=1)
    pass
