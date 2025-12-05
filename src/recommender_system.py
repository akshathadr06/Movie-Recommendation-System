"""
Unified Movie Recommendation System

Integrates all modules:
- Module 1: Concept Learning for preference identification
- Module 2: Rule Learning for explainable recommendations
- Module 3: Clustering & Ensemble for similarity and prediction
- Module 5: Probabilistic models for uncertainty handling

This system provides multi-layered recommendations with explanations.
"""

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

from src.modules.module1_concept_learning import ConceptLearner
from src.modules.module2_rule_learning import RuleLearner
from src.modules.module3_clustering_ensemble import MovieClusterer, EnsemblePredictor
from src.modules.module5_probabilistic_models import BayesianPreferenceNetwork, HiddenMarkovModel, KalmanFilter


@dataclass
class Recommendation:
    """A single recommendation with confidence and explanation."""
    movie_id: int
    movie_title: str
    confidence: float
    explanation: str
    reasoning: Dict  # Detailed breakdown of why recommended


class MovieRecommender:
    """
    Unified recommendation system combining multiple learning approaches.
    """
    
    def __init__(self, dataset, user_id: int):
        """
        Initialize recommender for a specific user.
        
        Args:
            dataset: MovieDataset object
            user_id: ID of user to recommend for
        """
        self.dataset = dataset
        self.user_id = user_id
        self.user = dataset.users.get(user_id)
        
        # Initialize all modules
        self.concept_learner = ConceptLearner()
        self.rule_learner = RuleLearner()
        self.movie_clusterer = MovieClusterer(n_clusters=3)
        self.ensemble_predictor = EnsemblePredictor()
        self.bayesian_network = BayesianPreferenceNetwork()
        self.hmm = HiddenMarkovModel()
        self.kalman_filter = KalmanFilter()
        
        # Train all modules
        self._train_all_modules()
    
    def _train_all_modules(self):
        """Train all learning modules on user's history."""
        
        if not self.user:
            return
        
        # Prepare data
        user_movies = self.user.rated_movies
        rated_movies = [self.dataset.movies[mid] for mid in user_movies.keys() 
                       if mid in self.dataset.movies]
        
        # Convert to feature dictionaries for learning
        positive_examples = []
        negative_examples = []
        
        for movie in rated_movies:
            rating = user_movies.get(movie.movie_id, 5)
            features = self._movie_to_features(movie)
            
            if rating >= 7.0:
                positive_examples.append(features)
            else:
                negative_examples.append(features)
        
        # Module 1: Learn preference concept
        if positive_examples and negative_examples:
            self.learned_concept = self.concept_learner.learn_preferences(
                positive_examples, negative_examples
            )
        else:
            self.learned_concept = {}
        
        # Module 2: Learn rules
        if positive_examples and negative_examples:
            self.rules = self.rule_learner.sequential_covering(
                positive_examples, negative_examples
            )
        else:
            self.rules = []
        
        # Module 3: Cluster movies and train ensemble
        self.movie_clusterer.fit(rated_movies)
        self.ensemble_predictor.train(user_movies, self.dataset.movies)
        
        # Module 5: Learn probabilistic models
        self.bayesian_network.learn_from_ratings(user_movies, self.dataset.movies)
        
        # Initialize Kalman filter with current belief
        avg_rating = np.mean(list(user_movies.values()))
        self.kalman_filter.state = np.array([avg_rating / 10.0])
        self.kalman_filter.update(avg_rating / 10.0)
        
        # Update HMM with recent behavior
        ratings_list = list(user_movies.values())[-10:]
        if ratings_list:
            self.hmm.update_preference_state(ratings_list)
    
    def _movie_to_features(self, movie) -> Dict:
        """Convert movie object to feature dictionary."""
        return {
            'movie_id': movie.movie_id,
            'title': movie.title,
            'genres': movie.genres,
            'rating': movie.rating,
            'cast': movie.cast,
            'keywords': movie.keywords,
            'year': movie.year,
            'runtime': movie.runtime,
        }
    
    def recommend(self, n_recommendations: int = 5, 
                 exclude_already_rated: bool = True) -> List[Recommendation]:
        """
        Generate recommendations using all modules.
        
        Args:
            n_recommendations: Number of recommendations to return
            exclude_already_rated: If True, don't recommend already-rated movies
        
        Returns:
            List of Recommendation objects sorted by confidence
        """
        
        if not self.user:
            return []
        
        recommendations = []
        
        # Filter movies to consider
        candidate_movies = []
        for movie_id, movie in self.dataset.movies.items():
            if exclude_already_rated and movie_id in self.user.rated_movies:
                continue
            candidate_movies.append(movie)
        
        if not candidate_movies:
            return recommendations
        
        # Score each candidate using multiple approaches
        for movie in candidate_movies:
            features = self._movie_to_features(movie)
            
            # Module 1: Concept-based score
            concept_score = self.concept_learner.predict_preference(
                features, self.learned_concept
            ) if self.learned_concept else 0.5
            
            # Module 2: Rule-based score
            rule_score = self._score_by_rules(movie)
            
            # Module 3: Ensemble score
            ensemble_score = self.ensemble_predictor.predict_preference(movie)
            
            # Module 5: Probabilistic score
            bayes_score = self.bayesian_network.infer_recommendation_probability(movie)
            
            # Similarity-based score (from clustering)
            similarity_score = self._similarity_score(movie)
            
            # Combine scores with weighted ensemble
            final_score = (
                0.2 * concept_score +      # Module 1
                0.15 * rule_score +        # Module 2
                0.25 * ensemble_score +    # Module 3
                0.25 * bayes_score +       # Module 5
                0.15 * similarity_score    # Similarity
            )
            
            # Generate explanation
            explanation = self._generate_explanation(movie, final_score)
            
            # Create recommendation
            rec = Recommendation(
                movie_id=movie.movie_id,
                movie_title=movie.title,
                confidence=final_score,
                explanation=explanation,
                reasoning={
                    'concept_score': concept_score,
                    'rule_score': rule_score,
                    'ensemble_score': ensemble_score,
                    'bayes_score': bayes_score,
                    'similarity_score': similarity_score,
                }
            )
            recommendations.append(rec)
        
        # Sort by confidence and return top N
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:n_recommendations]
    
    def _score_by_rules(self, movie) -> float:
        """Score a movie based on learned rules."""
        if not self.rules:
            return 0.5
        
        features = self._movie_to_features(movie)
        matching_rules = []
        
        for rule in self.rules:
            if self.rule_learner._rule_matches(rule, features):
                matching_rules.append(rule)
        
        if not matching_rules:
            return 0.5
        
        # Average confidence of matching rules
        avg_confidence = np.mean([rule.confidence for rule in matching_rules])
        return avg_confidence
    
    def _similarity_score(self, movie) -> float:
        """Score based on similarity to liked movies."""
        if not self.movie_clusterer.clusters:
            return 0.5
        
        similar_count = 0
        for cluster_movies in self.movie_clusterer.clusters.values():
            if movie.movie_id in cluster_movies:
                # Count how many liked movies are in this cluster
                liked = sum(1 for m_id in cluster_movies 
                           if m_id in self.user.rated_movies and 
                           self.user.rated_movies[m_id] >= 7)
                if liked > 0:
                    similar_count += 1
        
        return min(1.0, similar_count * 0.3)
    
    def _generate_explanation(self, movie, score: float) -> str:
        """Generate human-readable explanation for recommendation."""
        
        features = self._movie_to_features(movie)
        reasons = []
        
        # Check if matches learned concept
        if self.learned_concept:
            genres = self.learned_concept.get('genres', [])
            if any(g in movie.genres for g in genres):
                reasons.append(f"matches your preferred genres: {', '.join(movie.genres)}")
        
        # Check rating threshold
        if self.learned_concept.get('min_rating'):
            if movie.rating >= self.learned_concept['min_rating']:
                reasons.append(f"highly-rated ({movie.rating}/10)")
        
        # Check cast preferences
        cast_prefs = self.bayesian_network.actor_preference
        matching_actors = [a for a in movie.cast if cast_prefs.get(a, 0) > 0.5]
        if matching_actors:
            reasons.append(f"features actors you like: {', '.join(matching_actors[:2])}")
        
        if not reasons:
            reasons = [f"recommended based on your preference patterns"]
        
        return f"{movie.title} is recommended because it {', and '.join(reasons)}."
    
    def add_feedback(self, movie_id: int, rating: float):
        """
        Add new user feedback and update all models.
        
        Args:
            movie_id: ID of movie being rated
            rating: User's rating (1-10)
        """
        if movie_id not in self.dataset.movies:
            return
        
        # Update user's rating
        if self.user:
            self.user.rated_movies[movie_id] = rating
        
        # Update Kalman filter
        normalized_rating = rating / 10.0
        self.kalman_filter.predict()
        self.kalman_filter.update(normalized_rating)
        
        # Update HMM
        if len(self.user.rated_movies) % 3 == 0:  # Periodic update
            ratings = list(self.user.rated_movies.values())[-5:]
            self.hmm.update_preference_state(ratings)
        
        # Retrain ensemble on new data
        self.ensemble_predictor.train(self.user.rated_movies, self.dataset.movies)
        
        # Retrain Bayesian network
        self.bayesian_network.learn_from_ratings(
            self.user.rated_movies, self.dataset.movies
        )
    
    def get_system_insights(self) -> Dict:
        """Get insights about how different modules contribute to recommendations."""
        
        insights = {
            'current_preference_state': self.hmm.states[np.argmax(self.hmm.current_state_dist)],
            'preference_certainty': float(1.0 - self.kalman_filter.get_belief()[1]),
            'learned_genres': self.learned_concept.get('genres', []),
            'learned_cast': self.learned_concept.get('cast', []),
            'number_of_rules': len(self.rules),
            'bayesian_genre_prefs': self.bayesian_network.genre_preference,
        }
        
        return insights
