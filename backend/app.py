import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import math

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
#json_filename = 'exerciseDataset.json' #can change back to relative path for local development
json_filename = 'exerciseDatasetWithComments.json' #with the social data now
#json_filename = 'exerciseDataset.json'
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

def cossim(level, query, doc, svd=True): #TODO - add a toggle on the frontend for SVD
    comments = " ".join(doc["reddit_comments"]) #should be a list joined into one long comment (or none)
    equipment = doc["Equipment"]
    corpus = [doc["Description"] + " " + comments + " " + equipment, query]  #combine doc and query into a single list, add equipment for better fit
    score = max(doc["Rating"], 1.6) if doc["Rating"] else 1.3 
    score = math.log(score) 
    #apply this as a logarithmic multiple, minimum of .5x ish for docs that have ratings to prioritize them, max of 2.3x
    exercise_level = doc["Level"]
    level_mult = 1 if exercise_level == level else .8 #decrease scoring for non equal level
    #utilize svd on the corpus or just normal vectorizer
    if svd:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vectorizer.fit_transform(corpus) #fit and change dimensionality
        svd_model = TruncatedSVD(n_iter=10)
        X_svd = svd_model.fit_transform(X) #use svd on the data
    else:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vectorizer.fit_transform(corpus)
        X_svd = X.toarray()
        #fit on the query and documents, no svd - this is just x_svd to make it easier
    doc_vector = X_svd[0]  #transform doc as a single string
    query_vector = X_svd[1]  #transform query as a single string
    numerator = np.dot(query_vector, doc_vector.T)  # transpose doc_vector for correct shape
    denominator = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
    cossim = numerator/denominator
    return min(1, score*level_mult*cossim) #return the cosine similarity (capped at one due to the rating weighting)


#find the top k documents corresponding to a query - pass in a set of documents to check through
def top_k_docs(level, query, docs, k, svd):
    #run cosine similarity 
    top_k = []
    for doc in docs:
        score = cossim(level, query, doc, svd)
        rating = doc["Rating"] if doc["Rating"] else None
        comment = doc["reddit_comments"][0] if doc["reddit_comments"] else None #just gets the first associated reddit comment
        if score > 0.0001 or len(top_k) < 30: #for efficiency, cut poor exercises early
            top_k.append((doc, score, rating, comment))
    top_k = sorted(top_k, key=lambda x:x[1])[-k:][::-1] #sort by score and get the top k in reverse
    return [(exercise, score, rating, comment) for exercise, score, rating, comment in top_k] #return the exercise and the similarity score

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
        sport = "Other" #default for now - eventually we will do edit distance or something
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
    'Cardio': ['Cardio']
     #cardio filters by type instead of body part!
    }
    acceptable_parts = body_parts_to_categories[group]
    docs = []
    if group != "Cardio":
        for item in data: #iterate through the json file and filter by this group
            if item["BodyPart"] in acceptable_parts:
                docs.append(item)
    else:
        for item in data:
            if item["Type"] == "Cardio":
                docs.append(item)
    return docs

#takes in a sport, a level, and a query, and generates a split accordingly
def sport_search(sport, level, query, num_exercises=30, svd=True):
    split = get_split(sport, num_exercises)
    exercises = []
    for group in split:
        num = split[group]
        #sort the json file by this group
        grouped_docs = sort_json_by_group(group)
        top_k = top_k_docs(level, query, grouped_docs, num, svd)
        exercises.append(top_k)
    return exercises #also returns similarity scores and data ratings

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
    level = request.args.get("level")
    svd = request.args.get("svd")
    svd_bool = svd=="true"
    #print(svd == "true")
    if level.lower() not in ["beginner", "intermediate", "expert"]:
        level = "Intermediate"
    query = request.args.get("level") + " " + request.args.get("goals")
    exercises = sport_search(sport, level, query, 30, svd_bool)
    return exercises

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
