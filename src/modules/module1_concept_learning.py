"""
Module 1: Concept Learning & Hypothesis Search

This module implements concept learning algorithms to identify user preference patterns.
The system frames user taste as a target concept that can be learned from examples.
"""

import numpy as np
from typing import Set, Dict, List, Tuple
from itertools import combinations

class ConceptLearner:
    """
    Learns user preference concepts using Version Spaces and Candidate-Elimination.
    
    Instance Space: Movies with features [genre, rating, cast, keywords]
    Hypothesis Space: All possible user preference patterns
    """
    
    def __init__(self):
        self.hypotheses = set()
        self.specific = None  # Most specific hypothesis
        self.general = None   # Most general hypothesis
    
    def _feature_matches(self, movie_features: Dict, hypothesis: Dict) -> bool:
        """Check if a movie matches a hypothesis."""
        for key, value in hypothesis.items():
            if key == 'min_rating':
                if movie_features.get('rating', 0) < value:
                    return False
            elif key == 'genres':
                if not any(g in movie_features.get('genres', []) for g in value):
                    return False
            elif key == 'cast':
                if value and not any(c in movie_features.get('cast', []) for c in value):
                    return False
        return True
    
    def learn_preferences(self, positive_examples: List[Dict], 
                         negative_examples: List[Dict]) -> Dict:
        """
        Candidate-Elimination algorithm: learns a concept from positive and negative examples.
        
        Args:
            positive_examples: List of liked movies (features dict)
            negative_examples: List of disliked movies (features dict)
        
        Returns:
            Learned hypothesis representing user preference pattern
        """
        
        if not positive_examples:
            return {}
        
        # Initialize with first positive example (most specific)
        hypothesis = {
            'genres': set(positive_examples[0].get('genres', [])),
            'min_rating': positive_examples[0].get('rating', 5),
            'cast': set(positive_examples[0].get('cast', [])),
        }
        
        # Generalize based on positive examples
        for example in positive_examples[1:]:
            # Keep genres that appear in this example
            hypothesis['genres'] &= set(example.get('genres', []))
            if not hypothesis['genres']:  # If no common genres, use union
                hypothesis['genres'] = set(positive_examples[0].get('genres', [])) | set(example.get('genres', []))
            
            # Adjust minimum rating
            hypothesis['min_rating'] = min(hypothesis['min_rating'], example.get('rating', 5))
            
            # Update cast intersection
            hypothesis['cast'] &= set(example.get('cast', []))
        
        # Specialize based on negative examples
        for neg_example in negative_examples:
            if self._feature_matches(neg_example, hypothesis):
                # This hypothesis matches a negative example - need to specialize
                # By removing common features
                neg_genres = set(neg_example.get('genres', []))
                neg_cast = set(neg_example.get('cast', []))
                
                # Remove genres that appear in negative example
                hypothesis['genres'] -= neg_genres
                hypothesis['cast'] -= neg_cast
        
        # Convert sets to lists for serialization
        hypothesis['genres'] = list(hypothesis['genres'])
        hypothesis['cast'] = list(hypothesis['cast'])
        
        return hypothesis
    
    def predict_preference(self, movie_features: Dict, hypothesis: Dict) -> float:
        """
        Predict whether user will like a movie based on learned hypothesis.
        
        Returns:
            Confidence score between 0 and 1
        """
        
        if not hypothesis:
            return 0.5
        
        score = 0
        factors = 0
        
        # Genre matching
        if hypothesis.get('genres'):
            factors += 1
            if any(g in movie_features.get('genres', []) for g in hypothesis['genres']):
                score += 1
        
        # Rating threshold
        if 'min_rating' in hypothesis:
            factors += 1
            if movie_features.get('rating', 0) >= hypothesis['min_rating']:
                score += 0.8
        
        # Cast matching
        if hypothesis.get('cast'):
            factors += 1
            if any(c in movie_features.get('cast', []) for c in hypothesis['cast']):
                score += 1
        
        return score / factors if factors > 0 else 0.5
    
    def inductive_bias_assumption(self, learned_hypothesis: Dict) -> str:
        """
        Explains the inductive bias: assumption that users who liked certain movies
        will prefer similar movies with matching features.
        """
        bias_explanation = f"""
        Inductive Bias (Assumption):
        - Users with preference for genres {learned_hypothesis.get('genres', [])} 
          will prefer similar movies in these genres
        - Minimum acceptable rating threshold: {learned_hypothesis.get('min_rating', 'N/A')}
        - Preferred actors/cast: {learned_hypothesis.get('cast', [])}
        
        This bias justifies generalizing from observed positive examples to unseen movies.
        """
        return bias_explanation
