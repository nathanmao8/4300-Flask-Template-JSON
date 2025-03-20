from collections import Counter
import json
import math
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from tokenize import tokenize

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])

app = Flask(__name__)
CORS(app)

def build_inverted_index(msgs) -> dict: 
    """Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    """
    inverted_index = dict()
    for i in range(len(msgs)): #i is the doc id
      msg = msgs[i]
      tokens = msg["toks"]
      token_counts = Counter(tokens) #get the count of each token in this doc
      for token, count in token_counts.items(): #the count of the token goes into the tuple
        if token in inverted_index: #found in some other doc
          inverted_index[token].append((i, count))
        else: 
          inverted_index[token] = [(i, count)]
    return inverted_index

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    idf = dict()
    for term in inv_idx:
      #get the idf and map it to the term
      df = len(inv_idx[term])
      if df >= min_df and df/n_docs <= max_df_ratio: #successfully filters
        score = math.log2(n_docs/(1+df)) #df is the number of docs with this term
        idf[term] = score
    return idf

def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.

    Arguments
    =========

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.

    Returns
    =======

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    norms = np.zeros(n_docs)
    #first sum up the term frequency for words in docs * idf of those words
    for term in index:
      inv_df = idf.get(term, 0) #default to 0 if it doesn't exist in idf
      docs = index[term]
      for doc, tf in docs:
        norms[doc] += (tf*inv_df)**2 #sum the products squared
    #then sqrt norms
    return np.sqrt(norms)


def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.


    Returns
    =======
    
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    doc_scores = dict()
    for term, query_tf in query_word_counts.items():
      if term in index: #idf is a subset of index so don't need to check
        qi = query_tf*idf[term]
        docs = index[term] #associated docs with query term
        for doc, count in docs:
          dij = count * idf[term] #relies on idf as well
          if doc in doc_scores:
            doc_scores[doc] += qi * dij #the summation
          else:
            doc_scores[doc] = qi * dij
    return doc_scores

#return the cosine similarity score between a query and a document with some weighting
def cossim(query, doc, index, query_word_counts, idf, weight=None):
    #compute the norms
    q_norm = compute_doc_norms(query)
    doc_norm = compute_doc_norms(doc)
    if q_norm == 0 or doc_norm == 0: return 0 #0 score if either is 0 long to remove errors here
    dot_product = accumulate_dot_scores(query_word_counts, index, idf)
    return dot_product/(q_norm*doc_norm)

#find the top k documents corresponding to a query
def top_k_docs(query, docs, k):
    #run cosine similarity 
    top_k = []
    query_tokens = tokenize.tokenize(query.lower())
    query_word_counts = Counter(query_tokens)
    index = build_inverted_index(docs)
    for doc in docs:
        score = cossim(query, doc, index, query_word_counts, compute_idf(index, len(docs)))
        top_k.append((doc, score))
    return top_k.sorted(key=lambda x:x[1])[-k:] #sort by score and get the top k
'''TODO - There is definitely stuff wrong with this considering I pulled it straight from my A4, 
    but this is a starting point! update the types and such based on our data and make it functional using backend calls
'''

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)