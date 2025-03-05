from flask import Flask, request, jsonify
import sqlite3
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

def get_colleges():
    with open("colleges.json", "r", encoding="utf-8") as file:
        colleges = json.load(file)
    return colleges

def get_user_history(user_id):
    db_path = os.path.abspath("colleges.db")
    # print(f"Using database at: {db_path}")  # Debugging

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA table_info(search_history);")  # Print table structure
        columns = cursor.fetchall()
        # print(f"Columns in search_history: {columns}")

        cursor.execute("SELECT search_query FROM search_history WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        return [row[0].lower() for row in rows] if rows else []
    except Exception as e:
        print(f"Database Query Error: {e}")  # Debugging
    finally:
        conn.close()

    return []

def recommend_colleges(user_id, top_n=10):
    user_history = get_user_history(user_id)
    all_colleges = get_colleges()

    if not user_history:
        print(f"No search history found for user {user_id}")
        return []

    if not all_colleges:
        print("No colleges data found!")
        return []

    college_texts = []
    for college in all_colleges:
        try:
            combined_text = f"{college['name'].lower()} {college['location'].lower()} {' '.join(college['courses']).lower()}"
            college_texts.append(combined_text)
        except KeyError as e:
            print(f"Missing key in college data: {e}")
            continue

    user_search_text = " ".join(user_history)
    
    if not college_texts:
        print("No valid college data to process.")
        return []

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
    try:
        user_id = request.args.get("user_id", type=int)
        if user_id is None:
            return jsonify({"error": "User ID is required"}), 400

        # print(f"User ID received: {user_id}")  # Debugging

        recommendations = recommend_colleges(user_id)
        # print(f"Generated Recommendations: {recommendations}")  # Debugging

        return jsonify(recommendations)

    except Exception as e:
        print("Error in /recommend:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
