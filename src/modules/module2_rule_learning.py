"""
Module 2: Rule Learning & Analytical Reasoning

This module implements rule-based learning to extract explainable recommendation rules
from user preferences and movie features.

Algorithms: Sequential Covering, FOIL, Example-Based Rule Learning, EBL
"""

from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

@dataclass
class Rule:
    """Represents a recommendation rule."""
    conditions: Dict  # Feature conditions (e.g., {"genre": "Sci-Fi", "rating_min": 8})
    conclusion: str  # "RECOMMEND" or "DON'T_RECOMMEND"
    confidence: float  # Probability this rule is correct
    coverage: float  # Fraction of examples this rule covers
    
    def __str__(self):
        cond_str = " AND ".join([f"{k}={v}" for k, v in self.conditions.items()])
        return f"IF {cond_str} THEN {self.conclusion} (Confidence: {self.confidence:.2%}, Coverage: {self.coverage:.2%})"


class RuleLearner:
    """
    Learns recommendation rules using Sequential Covering and FOIL principles.
    Transforms examples into interpretable if-then rules.
    """
    
    def __init__(self, min_confidence: float = 0.7):
        self.rules = []
        self.min_confidence = min_confidence
    
    def _compute_rule_metrics(self, rule: Rule, examples: List[Dict], 
                             target_class: str) -> Tuple[float, float]:
        """Compute confidence and coverage for a rule."""
        matching = [e for e in examples if self._rule_matches(rule, e)]
        
        if not matching:
            return 0.0, 0.0
        
        correct = sum(1 for e in matching if e.get('class') == target_class)
        
        confidence = correct / len(matching) if matching else 0.0
        coverage = len(matching) / len(examples) if examples else 0.0
        
        return confidence, coverage
    
    def _rule_matches(self, rule: Rule, example: Dict) -> bool:
        """Check if an example matches all rule conditions."""
        for key, value in rule.conditions.items():
            ex_value = example.get(key)
            
            if isinstance(value, (list, set)):
                # Check if any condition value matches
                if not any(v in ex_value if isinstance(ex_value, (list, set)) else v == ex_value for v in value):
                    return False
            elif isinstance(value, str) and isinstance(ex_value, (list, set)):
                # Check if string is in list
                if value not in ex_value:
                    return False
            else:
                if ex_value != value:
                    return False
        return True
    
    def sequential_covering(self, positive_examples: List[Dict], 
                           negative_examples: List[Dict]) -> List[Rule]:
        """
        Sequential Covering algorithm: learns rules by repeatedly finding the best rule
        that covers uncovered positive examples.
        
        Args:
            positive_examples: Liked movies (should have a 'class' field set to 'LIKE')
            negative_examples: Disliked movies (should have a 'class' field set to 'DISLIKE')
        
        Returns:
            List of learned rules
        """
        
        # Label examples
        labeled_pos = [dict(e, **{'class': 'LIKE'}) for e in positive_examples]
        labeled_neg = [dict(e, **{'class': 'DISLIKE'}) for e in negative_examples]
        uncovered_pos = set(range(len(labeled_pos)))
        rules = []
        
        while uncovered_pos:
            best_rule = None
            best_new_coverage = 0
            
            # Generate candidate rules
            candidates = self._generate_candidate_rules(labeled_pos, labeled_neg)
            
            for candidate in candidates:
                # Count how many new positive examples it covers
                covered = {i for i in uncovered_pos if self._rule_matches(candidate, labeled_pos[i])}
                new_coverage = len(covered)
                
                # Check confidence on all data
                conf, cov = self._compute_rule_metrics(candidate, labeled_pos + labeled_neg, 'LIKE')
                
                if conf >= self.min_confidence and new_coverage > best_new_coverage:
                    best_rule = candidate
                    best_new_coverage = new_coverage
                    best_rule.confidence = conf
                    best_rule.coverage = cov
            
            if best_rule and best_new_coverage > 0:
                rules.append(best_rule)
                # Remove covered examples
                covered = {i for i in uncovered_pos if self._rule_matches(best_rule, labeled_pos[i])}
                uncovered_pos -= covered
            else:
                break  # No more good rules found
        
        self.rules = rules
        return rules
    
    def _generate_candidate_rules(self, pos_examples: List[Dict], 
                                 neg_examples: List[Dict]) -> List[Rule]:
        """Generate candidate rules from examples (FOIL principle)."""
        candidates = []
        
        if not pos_examples:
            return candidates
        
        # Extract feature combinations from positive examples
        feature_sets = set()
        
        for example in pos_examples:
            # Create rules based on individual features
            for key in ['genres', 'rating', 'cast', 'year']:
                if key in example:
                    value = example[key]
                    # Convert lists to tuples for hashability
                    if isinstance(value, list):
                        value = tuple(value)
                    conditions = {key: value}
                    rule = Rule(conditions, 'RECOMMEND', 0.0, 0.0)
                    # Use frozenset with hashable items
                    feature_tuple = tuple(sorted([(k, v if not isinstance(v, list) else tuple(v)) 
                                                   for k, v in conditions.items()]))
                    feature_sets.add(feature_tuple)
        
        # Create rules from feature combinations
        for features in feature_sets:
            conditions = {}
            for k, v in features:
                conditions[k] = v if not isinstance(v, tuple) else list(v)
            rule = Rule(conditions, 'RECOMMEND', 0.0, 0.0)
            candidates.append(rule)
        
        return candidates
    
    def explanation_based_learning(self, user_liked_movies: List[Dict], 
                                   domain_knowledge: Dict) -> List[str]:
        """
        Explanation-Based Learning (EBL): Explains why recommendations are made
        using domain knowledge.
        
        Args:
            user_liked_movies: Movies the user has rated positively
            domain_knowledge: Prior knowledge about movies and preferences
        
        Returns:
            Explanation strings for each recommendation
        """
        
        explanations = []
        
        if not user_liked_movies:
            return explanations
        
        # Extract common features
        common_genres = self._find_common_features(user_liked_movies, 'genres')
        common_cast = self._find_common_features(user_liked_movies, 'cast')
        avg_rating = sum(m.get('rating', 5) for m in user_liked_movies) / len(user_liked_movies)
        
        explanation = f"""
        Based on your preferences:
        - You tend to like movies in: {', '.join(common_genres) if common_genres else 'diverse genres'}
        - You appreciate actors: {', '.join(common_cast) if common_cast else 'various actors'}
        - You prefer movies with ratings around: {avg_rating:.1f}/10
        
        Therefore, we recommend similar movies matching these criteria.
        """
        explanations.append(explanation)
        
        return explanations
    
    def _find_common_features(self, movies: List[Dict], feature: str) -> List[str]:
        """Find features that appear in most movies."""
        feature_counts = {}
        
        for movie in movies:
            features = movie.get(feature, [])
            if isinstance(features, (list, set)):
                for f in features:
                    feature_counts[f] = feature_counts.get(f, 0) + 1
            else:
                feature_counts[features] = feature_counts.get(features, 0) + 1
        
        # Return features appearing in at least half the movies
        threshold = len(movies) / 2
        return [f for f, count in feature_counts.items() if count >= threshold]
    
    def explain_recommendation(self, movie: Dict, matching_rules: List[Rule]) -> str:
        """Generate human-readable explanation for why a movie is recommended."""
        if not matching_rules:
            return "Recommendation based on general preference patterns."
        
        best_rule = matching_rules[0]
        explanation = f"""
        RECOMMENDATION EXPLANATION:
        This movie matches your preferences because:
        {best_rule}
        
        Matching criteria:
        """
        
        for key, value in best_rule.conditions.items():
            if key in movie:
                explanation += f"\n  ✓ {key}: {movie[key]}"
        
        return explanation
