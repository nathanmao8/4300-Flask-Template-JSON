from collections import Counter
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from nltk import TreebankWordTokenizer

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

#function to convert 'megaGymDataset.csv' to 'exerciseDataset.json' 
csv_filename = '../data/megaGymDataset.csv'
#json_filename = 'backend/exerciseDataset.json' #can change back to relative path for local development
json_filename = 'exerciseDataset.json'
'''
# Read CSV and convert to JSON
data = []
with open(csv_filename, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Create new combined 'description' field
        title = row.get('Title', '').strip()
        desc = row.get('Desc', '').strip()
        row['Description'] = f"{title} {desc}".strip()

        # Convert numerical fields to proper types
        row['Rating'] = float(row['Rating']) if row['Rating'] else None

        # Remove original Title and Desc fields
        row.pop('Title', None)
        row.pop('Desc', None)

        data.append(row)

# Write to JSON file
with open(json_filename, 'w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, indent=4, ensure_ascii=False)
'''
with open(json_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)



def build_vectorizer(max_features, stop_words, max_df=90, min_df=1, norm='l2'):
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

def cossim(query, doc):
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    
    # Tokenize the doc and the query (no need to tokenize manually)
    corpus = [doc["Description"], query]  # Combine doc and query into a single list
    
    # Fit the vectorizer on the combined corpus (doc and query)
    vectorizer.fit(corpus)
    
    # Transform both the doc and the query using the fitted vectorizer
    doc_vector = vectorizer.transform([doc["Description"]]).toarray()  # Transform doc as a single string
    query_vector = vectorizer.transform([query]).toarray()  # Transform query as a single string
    
    # Now both doc_vector and query_vector are 2D arrays with shape (1, n_features)
    # Compute the cosine similarity
    numerator = np.dot(query_vector, doc_vector.T)  # Transpose doc_vector for correct shape
    denominator = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
    
    # Return the cosine similarity (should be a scalar)
    return numerator[0][0] / denominator

#find the top k documents corresponding to a query - pass in a set of documents to check through
def top_k_docs(query, docs, k):
    #run cosine similarity 
    top_k = []
    for doc in docs:
        score = cossim(query, doc)
        top_k.append((doc, score))
    top_k = sorted(top_k, key=lambda x:x[1])[-k:] #sort by score and get the top k
    return [exercise for exercise, score in top_k]

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
def sort_json_by_group(group):
    body_parts_to_categories = {
    'Glutes': ['Glutes', 'Abductors'],
    'Hamstrings': ['Hamstrings'],
    'Quads': ['Quadriceps', 'Adductors'],
    'Biceps': ['Biceps'],
    'Triceps': ['Triceps'],
    'Shoulders': ['Shoulders'],
    'Back': ['Lats', 'Middle Back', 'Traps', 'Lower Back', 'Forearms'],
    'Chest': ['Chest'],
    'Abs/Core': ['Abdominals'],
    'Cardio': ['Neck', 'Calves']
    }
    acceptable_parts = body_parts_to_categories[group]
    docs = []
    for item in data: #iterate through the json file and filter by this group
        if item["BodyPart"] in acceptable_parts:
            docs.append(item)
    return docs

#takes in a sport and a query, and generates a split according to both
def sport_search(sport, query, num_exercises=30):
    split = get_split(sport, num_exercises)
    exercises = []
    for group in split:
        num = split[group]
        #sort the json file by this group
        grouped_docs = sort_json_by_group(group)
        top_k = top_k_docs(query, grouped_docs, num)
        exercises.append(top_k)
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

@app.route("/exercises")
def exercises_search():
    sport = request.args.get("sport")
    query = request.args.get("level") + " " + request.args.get("goals")
    exercises = sport_search(sport, query, 30)
    return exercises

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
