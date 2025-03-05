from flask import Flask, request, jsonify
import sqlite3
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def get_colleges():
    with open("colleges.json", "r", encoding="utf-8") as file:
        return json.load(file)

def get_user_history(user_id):
    db_path = os.path.abspath("../backend/colleges.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT search_query FROM search_history WHERE user_id = ? ORDER BY id DESC LIMIT 50", (user_id,))
        rows = cursor.fetchall()
        return [row[0].lower() for row in rows] if rows else []
    finally:
        conn.close()

def recommend_colleges(user_id, top_n=30):
    user_history = get_user_history(user_id)
    all_colleges = get_colleges()

    if not user_history:
        print(f"No search history found for user {user_id}")
        return []
    
    if not all_colleges:
        print("No colleges data found!")
        return []

    college_texts = [f"{c['name'].lower()} {c['location'].lower()} {' '.join(c['courses']).lower()}" for c in all_colleges]

    user_search_text = " ".join(set(user_history))  # Remove duplicates for better TF-IDF scoring
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_search_text] + college_texts)

    user_vector = tfidf_matrix[0]
    college_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vector, college_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get top 10 most similar colleges

    return [all_colleges[i] for i in top_indices]

@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = request.args.get("user_id", type=int)
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        recommendations = recommend_colleges(user_id)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
