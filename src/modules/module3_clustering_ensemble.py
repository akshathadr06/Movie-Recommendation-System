"""
Module 3: Clustering, Similarity & Ensemble Prediction

This module implements:
- K-Means clustering for grouping similar movies and users
- Self-Organizing Maps (SOM) for topology-preserving visualization
- Ensemble methods (Bagging, Random Forests, AdaBoost) for robust predictions
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


class MovieClusterer:
    """
    Clusters movies using K-Means to group similar movies together.
    Enables "people who liked X also liked Y" type recommendations.
    """
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.movie_features = []
        self.movie_ids = []
        self.clusters = None
    
    def _extract_numeric_features(self, movies: List[Dict]) -> np.ndarray:
        """Convert movie features to numeric vectors."""
        features = []
        
        for movie in movies:
            feature_vector = [
                movie.get('rating', 5.0),
                len(movie.get('genres', [])),
                len(movie.get('cast', [])),
                len(movie.get('keywords', [])),
                movie.get('year', 2000),
                movie.get('runtime', 120),
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, movies: List[Dict]) -> Dict:
        """
        Fit K-Means clustering on movies.
        
        Returns:
            Dictionary mapping cluster_id to list of movie_ids
        """
        if len(movies) < self.n_clusters:
            self.n_clusters = max(1, len(movies) - 1)
        
        self.movie_ids = [m.get('movie_id', i) for i, m in enumerate(movies)]
        features = self._extract_numeric_features(movies)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(features_scaled)
        
        # Create cluster mapping
        self.clusters = {}
        for movie_id, label in zip(self.movie_ids, labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(movie_id)
        
        return self.clusters
    
    def get_similar_movies(self, movie_id: int, n_similar: int = 5) -> List[int]:
        """Get similar movies from the same cluster."""
        if not self.clusters:
            return []
        
        # Find which cluster this movie belongs to
        for cluster_id, movie_ids in self.clusters.items():
            if movie_id in movie_ids:
                # Return other movies from same cluster
                similar = [m for m in movie_ids if m != movie_id]
                return similar[:n_similar]
        
        return []


class SelfOrganizingMap:
    """
    Self-Organizing Map for topology-preserving visualization of movie space.
    Groups movies while preserving their similarity relationships.
    """
    
    def __init__(self, grid_size: int = 3, learning_rate: float = 0.5):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.weights = None
        self.scaler = StandardScaler()
    
    def _extract_numeric_features(self, movies: List[Dict]) -> np.ndarray:
        """Convert movie features to numeric vectors."""
        features = []
        for movie in movies:
            feature_vector = [
                movie.get('rating', 5.0),
                len(movie.get('genres', [])),
                len(movie.get('cast', [])),
                movie.get('year', 2000),
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def fit(self, movies: List[Dict], epochs: int = 100) -> np.ndarray:
        """Train SOM on movies."""
        features = self._extract_numeric_features(movies)
        features = self.scaler.fit_transform(features)
        
        # Initialize weights
        n_features = features.shape[1]
        self.weights = np.random.randn(self.grid_size, self.grid_size, n_features)
        
        for epoch in range(epochs):
            for feature_vector in features:
                # Find Best Matching Unit (BMU)
                distances = np.sum((self.weights - feature_vector) ** 2, axis=2)
                bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
                
                # Update weights
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        distance = np.sqrt((i - bmu_idx[0]) ** 2 + (j - bmu_idx[1]) ** 2)
                        influence = np.exp(-distance ** 2 / (2 * (self.grid_size / 2) ** 2))
                        self.weights[i, j] += self.learning_rate * influence * (feature_vector - self.weights[i, j])
            
            # Decay learning rate
            self.learning_rate *= 0.99
        
        return self.weights


class EnsemblePredictor:
    """
    Ensemble learning for robust movie preference prediction.
    Combines Random Forests, AdaBoost, and Bagging for improved accuracy.
    """
    
    def __init__(self):
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        self.adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
        self.is_trained = False
    
    def _prepare_training_data(self, user_ratings: Dict[int, float], 
                               movies: Dict[int, 'Movie'], threshold: float = 7.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert user ratings and movies to features and labels for training.
        
        Args:
            user_ratings: Dictionary of movie_id -> rating
            movies: Dictionary of movie_id -> movie object
            threshold: Rating threshold for positive class
        """
        X = []
        y = []
        
        for movie_id, rating in user_ratings.items():
            if movie_id in movies:
                movie = movies[movie_id]
                # Handle both dict and object types
                if hasattr(movie, 'rating'):
                    # It's a Movie object
                    features = [
                        movie.rating,
                        len(movie.genres),
                        len(movie.cast),
                        len(movie.keywords),
                        movie.year,
                    ]
                else:
                    # It's a dict
                    features = [
                        movie.get('rating', 5.0),
                        len(movie.get('genres', [])),
                        len(movie.get('cast', [])),
                        len(movie.get('keywords', [])),
                        movie.get('year', 2000),
                    ]
                X.append(features)
                y.append(1 if rating >= threshold else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, user_ratings: Dict[int, float], movies: Dict[int, Dict]):
        """Train ensemble on user's rated movies."""
        X, y = self._prepare_training_data(user_ratings, movies)
        
        if len(np.unique(y)) < 2 or len(X) < 2:
            # Need both positive and negative examples
            self.is_trained = False
            return
        
        self.random_forest.fit(X, y)
        self.adaboost.fit(X, y)
        self.is_trained = True
    
    def predict_preference(self, movie) -> float:
        """
        Predict probability user will like a movie using ensemble voting.
        
        Returns:
            Confidence score between 0 and 1
        """
        if not self.is_trained:
            return 0.5
        
        # Handle both Movie objects and dicts
        if hasattr(movie, 'rating'):
            features = np.array([[
                movie.rating,
                len(movie.genres),
                len(movie.cast),
                len(movie.keywords),
                movie.year,
            ]])
        else:
            features = np.array([[
                movie.get('rating', 5.0),
                len(movie.get('genres', [])),
                len(movie.get('cast', [])),
                len(movie.get('keywords', [])),
                movie.get('year', 2000),
            ]])
        
        # Ensemble prediction (average)
        rf_pred = self.random_forest.predict_proba(features)[0][1]
        ab_pred = self.adaboost.predict_proba(features)[0][1]
        
        return (rf_pred + ab_pred) / 2
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest."""
        if not self.is_trained:
            return {}
        
        feature_names = ['rating', 'num_genres', 'num_cast', 'num_keywords', 'year']
        importances = self.random_forest.feature_importances_
        
        return {name: imp for name, imp in zip(feature_names, importances)}


class CommitteeBasedEnsemble:
    """
    Decision by Committee: Multiple base learners vote on whether to recommend.
    """
    
    def __init__(self, n_learners: int = 5):
        self.learners = [RandomForestClassifier(n_estimators=10, random_state=i) for i in range(n_learners)]
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train all learners."""
        for learner in self.learners:
            learner.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction from committee.
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            return np.array([]), np.array([])
        
        votes = np.array([learner.predict_proba(X)[:, 1] for learner in self.learners])
        predictions = np.mean(votes, axis=0)  # Average vote
        confidence = np.std(votes, axis=0)     # Consistency
        
        return predictions, 1 - confidence  # Higher confidence if all agree
