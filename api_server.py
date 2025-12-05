"""
Flask API Server for Movie Recommendation System

Provides REST API endpoints for recommendation generation.
"""

from flask import Flask, jsonify, request
import json
from src.data.dataset import generate_sample_dataset
from src.recommender_system import MovieRecommender

app = Flask(__name__)

# Load dataset globally
DATASET = generate_sample_dataset()
RECOMMENDERS = {}


def get_recommender(user_id: int) -> MovieRecommender:
    """Get or create recommender for a user."""
    if user_id not in RECOMMENDERS:
        RECOMMENDERS[user_id] = MovieRecommender(DATASET, user_id)
    return RECOMMENDERS[user_id]


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Movie Recommendation System',
        'version': '1.0'
    }), 200


@app.route('/users/<int:user_id>/recommend', methods=['GET'])
def get_recommendations(user_id: int):
    """Get recommendations for a user."""
    try:
        n_recs = request.args.get('n', 5, type=int)
        
        if user_id not in DATASET.users:
            return jsonify({'error': 'User not found'}), 404
        
        recommender = get_recommender(user_id)
        recommendations = recommender.recommend(n_recommendations=n_recs)
        
        result = {
            'user_id': user_id,
            'recommendations': [
                {
                    'movie_id': rec.movie_id,
                    'title': rec.movie_title,
                    'confidence': rec.confidence,
                    'explanation': rec.explanation,
                    'reasoning': rec.reasoning
                }
                for rec in recommendations
            ]
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/users/<int:user_id>/feedback', methods=['POST'])
def add_feedback(user_id: int):
    """Add user feedback (rating) and update models."""
    try:
        data = request.json
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        
        if not movie_id or not rating:
            return jsonify({'error': 'movie_id and rating required'}), 400
        
        if user_id not in DATASET.users:
            return jsonify({'error': 'User not found'}), 404
        
        recommender = get_recommender(user_id)
        recommender.add_feedback(movie_id, rating)
        
        return jsonify({
            'status': 'feedback added',
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/users/<int:user_id>/insights', methods=['GET'])
def get_insights(user_id: int):
    """Get system insights for a user."""
    try:
        if user_id not in DATASET.users:
            return jsonify({'error': 'User not found'}), 404
        
        recommender = get_recommender(user_id)
        insights = recommender.get_system_insights()
        
        return jsonify({
            'user_id': user_id,
            'insights': insights
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/users', methods=['GET'])
def list_users():
    """List all available users."""
    users = [
        {
            'user_id': uid,
            'name': user.name,
            'rated_movies': len(user.rated_movies)
        }
        for uid, user in DATASET.users.items()
    ]
    
    return jsonify({'users': users}), 200


@app.route('/movies', methods=['GET'])
def list_movies():
    """List all available movies."""
    movies = [
        {
            'movie_id': mid,
            'title': movie.title,
            'genres': movie.genres,
            'rating': movie.rating,
            'year': movie.year
        }
        for mid, movie in DATASET.movies.items()
    ]
    
    return jsonify({'movies': movies}), 200


@app.route('/api/version', methods=['GET'])
def api_version():
    """Get API version."""
    return jsonify({
        'api_version': '1.0',
        'modules': {
            'concept_learning': 'Module 1',
            'rule_learning': 'Module 2',
            'clustering_ensemble': 'Module 3',
            'probabilistic_models': 'Module 5'
        }
    }), 200


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   Movie Recommendation System - Flask API Server       ║
    ╚════════════════════════════════════════════════════════╝
    
    Available Endpoints:
    
    GET  /health
         Check if API is running
    
    GET  /users
         List all available users
    
    GET  /movies
         List all available movies
    
    GET  /users/<user_id>/recommend?n=5
         Get recommendations for user (n=number of recommendations)
    
    POST /users/<user_id>/feedback
         Add rating feedback: {"movie_id": 10, "rating": 8.5}
    
    GET  /users/<user_id>/insights
         Get preference insights for user
    
    GET  /api/version
         Get API version and available modules
    
    Starting server on http://localhost:5000
    Press CTRL+C to stop
    """)
    
    app.run(debug=True, port=5000)
