from flask import Flask, request, jsonify
import sqlite3
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def get_colleges():
    with open("colleges.json", "r", encoding="utf-8") as file:
        colleges = json.load(file)
    return colleges

def get_user_history(user_id):
    conn = sqlite3.connect("colleges.db")
    cursor = conn.cursor()
    cursor.execute("SELECT search_query FROM search_history WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0].lower() for row in rows] if rows else []

def recommend_colleges(user_id, top_n=10):
    user_history = get_user_history(user_id)
    all_colleges = get_colleges()

    if not user_history or not all_colleges:
        return []

    # Combine name, location, and courses for each college
    college_texts = []
    for college in all_colleges:
        combined_text = f"{college['name'].lower()} {college['location'].lower()} {' '.join(college['courses']).lower()}"
        college_texts.append(combined_text)

    user_search_text = " ".join(user_history)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_search_text] + college_texts)

    user_vector = tfidf_matrix[0]
    college_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vector, college_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommended_colleges = [all_colleges[i] for i in top_indices]

    return recommended_colleges

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "User ID is required"}), 400
    
    recommendations = recommend_colleges(user_id)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
