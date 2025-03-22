from collections import Counter
import json
from sklearn.feature_extraction.text import TfidfVectorizer
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

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    """Returns a TfidfVectorizer object with the above preprocessing properties.
    
    Note: This function may log a deprecation warning. This is normal, and you
    can simply ignore it.
    
    Parameters
    ----------
    max_features : int
        Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer 
        constructer.
    stop_words : str
        Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer. 
    max_df : float
        Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer. 
    min_df : float
        Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer. 
    norm : str
        Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer. 

    Returns
    -------
    TfidfVectorizer
        A TfidfVectorizer object with the given parameters as its preprocessing properties.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words, max_df=max_df,min_df=min_df,norm=norm)
    return vectorizer

#return the cosine similarity score between a query and a document
def cossim(query, doc):
    #compute the norms and dot product, divide dot by norms multiplied
    vectorizer = build_vectorizer(5000, "english")
    doc_vector = vectorizer.fit_transform([d['description'] for d in data]).toarray()
    tokens = tokenize(query)
    query_vector = vectorizer.fit_transform(tokens)
    numerator = np.dot(query_vector, doc_vector)
    denominator = np.linalg.norm(query) * np.linalg.norm(doc)
    return numerator/denominator

#find the top k documents corresponding to a query - pass in a set of documents to check through
def top_k_docs(query, docs, k):
    #run cosine similarity 
    top_k = []
    for doc in docs:
        score = cossim(query, doc)
        top_k.append((doc, score))
    return top_k.sorted(key=lambda x:x[1])[-k:] #sort by score and get the top k

#get the exercise percentage split for a plan given a sport - should add to num_exercises
def get_split(sport, num_exercises):
    given_sports = ["Basketball", "Baseball", "Lacrosse", "Tennis", "Volleyball", "Tennis", "Weightlifting", "Other"]
    exercise_data = {
        "Basketball": {"Glutes": 15, "Hamstrings": 15, "Quads": 20, "Biceps": 10, "Triceps": 10, "Shoulders": 10, "Back": 5, "Chest": 5, "Cardio": 30, "Abs/Core": 15},
        "Baseball": {"Glutes": 10, "Hamstrings": 10, "Quads": 10, "Biceps": 15, "Triceps": 15, "Shoulders": 15, "Back": 10, "Chest": 5, "Cardio": 15, "Abs/Core": 15},
        "Lacrosse": {"Glutes": 15, "Hamstrings": 15, "Quads": 20, "Biceps": 10, "Triceps": 10, "Shoulders": 10, "Back": 10, "Chest": 5, "Cardio": 35, "Abs/Core": 15},
        "Tennis": {"Glutes": 10, "Hamstrings": 10, "Quads": 15, "Biceps": 10, "Triceps": 10, "Shoulders": 15, "Back": 10, "Chest": 5, "Cardio": 35, "Abs/Core": 15},
        "Volleyball": {"Glutes": 15, "Hamstrings": 10, "Quads": 15, "Biceps": 10, "Triceps": 10, "Shoulders": 15, "Back": 10, "Chest": 5, "Cardio": 25, "Abs/Core": 15},
        "Weightlifting": {"Glutes": 20, "Hamstrings": 15, "Quads": 20, "Biceps": 10, "Triceps": 10, "Shoulders": 10, "Back": 15, "Chest": 10, "Cardio": 5, "Abs/Core": 10},
        "Other": {"Glutes": 15, "Hamstrings": 10, "Quads": 15, "Biceps": 10, "Triceps": 10, "Shoulders": 10, "Back": 10, "Chest": 10, "Cardio": 20, "Abs/Core": 10},
    }
    if sport not in given_sports:
        sport = "Other" #default for now - eventually we will do edit distance
    percentages = exercise_data[sport]
    sport_dict = {muscle: round((percent / 100) * num_exercises) for muscle, percent in percentages.items()}
    total_assigned_exercises = sum(sport_dict.values())
    # make sure the total exercises adds to num_exercises
    while total_assigned_exercises > num_exercises:
        max_muscle = max(sport_dict, key=sport_dict.get)
        if sport_dict[max_muscle] > 0:
            sport_dict[max_muscle] -= 1
            total_assigned_exercises -= 1
    while total_assigned_exercises < num_exercises:
        min_muscle = min((m for m in sport_dict if sport_dict[m] > 0), key=sport_dict.get, default=None)
        if min_muscle:
            sport_dict[min_muscle] += 1
            total_assigned_exercises += 1
    return sport_dict

#should filter a json file to only contain items from a specific group
def sort_json_by_group(group, filename):
    json_file = filename #idk - fix this later
    docs = dict()
    for item in json_file:
        if item["BodyPart"] == group:
            docs[item]= item 
    return docs

#takes in a sport and a query, and generates a split according to both
def sport_search(sport, query, num_exercises=30):
    split = get_split(sport, num_exercises)
    exercises = top_k
    for group, num in split.values():
        #sort the json file by this group
        grouped_docs = sort_json_by_group(group)
        top_k = top_k_docs(query, grouped_docs, num)
    return exercises

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
