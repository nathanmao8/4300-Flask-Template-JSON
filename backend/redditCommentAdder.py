import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def assign_reddit_comments_to_exercises_json(exercise_data, reddit_comments, filename='exerciseDatasetWithComments.json', n_components=100):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_directory, filename)

    exercise_texts = [exercise['Description'] for exercise in exercise_data]
    reddit_texts = [comment['text'] for comment in reddit_comments]

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(exercise_texts + reddit_texts)

    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)

    exercise_vectors = X_reduced[:len(exercise_texts)]
    reddit_vectors = X_reduced[len(exercise_texts):]

    similarity_matrix = cosine_similarity(reddit_vectors, exercise_vectors)

    assigned_comments = {desc: [] for desc in exercise_texts}

    for idx, similarities in enumerate(similarity_matrix):
        best_match_idx = similarities.argmax()
        best_match_desc = exercise_texts[best_match_idx]
        assigned_comments[best_match_desc].append(reddit_comments[idx]['text'])

    for exercise in exercise_data:
        exercise['reddit_comments'] = assigned_comments.get(exercise['Description'], [])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(exercise_data, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Updated dataset written to: {output_path}")


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_directory, 'exerciseDataset.json'), 'r') as f:
        exercise_data = json.load(f)

    with open(os.path.join(current_directory, 'utterancesFiltered.json'), 'r') as f:
        reddit_comments = json.load(f)

    assign_reddit_comments_to_exercises_json(exercise_data, reddit_comments)
