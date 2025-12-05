"""
Module 5: Probabilistic Modeling & Sequential Behavior

This module implements:
- Bayesian Networks for modeling relationships between preferences
- Hidden Markov Models for tracking preference state changes
- Kalman Filters for belief refinement
- MCMC Sampling for probabilistic inference
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm, multinomial
import warnings
warnings.filterwarnings('ignore')


class BayesianPreferenceNetwork:
    """
    Bayesian Network modeling relationships between:
    - User genre preference
    - User rating threshold
    - Recommendation probability
    """
    
    def __init__(self):
        # Prior probabilities
        self.genre_preference = {}  # genre -> P(prefer)
        self.actor_preference = {}  # actor -> P(prefer)
        self.rating_threshold = 6.5
        self.recommendation_prior = 0.5
    
    def learn_from_ratings(self, user_ratings: Dict[int, float], 
                          movies: Dict[int, Dict]):
        """Learn Bayesian network from user's rating history."""
        
        if not user_ratings:
            return
        
        # Calculate threshold
        ratings = list(user_ratings.values())
        self.rating_threshold = np.mean(ratings)
        
        # Learn genre preferences
        genre_likes = {}
        genre_total = {}
        
        for movie_id, rating in user_ratings.items():
            if movie_id in movies:
                movie = movies[movie_id]
                genres = movie.get('genres', [])
                
                for genre in genres:
                    genre_total[genre] = genre_total.get(genre, 0) + 1
                    if rating >= self.rating_threshold:
                        genre_likes[genre] = genre_likes.get(genre, 0) + 1
        
        # Calculate conditional probabilities P(like | genre)
        for genre, total in genre_total.items():
            likes = genre_likes.get(genre, 0)
            self.genre_preference[genre] = (likes + 1) / (total + 2)  # Laplace smoothing
        
        # Learn actor preferences
        actor_likes = {}
        actor_total = {}
        
        for movie_id, rating in user_ratings.items():
            if movie_id in movies:
                movie = movies[movie_id]
                cast = movie.get('cast', [])
                
                for actor in cast:
                    actor_total[actor] = actor_total.get(actor, 0) + 1
                    if rating >= self.rating_threshold:
                        actor_likes[actor] = actor_likes.get(actor, 0) + 1
        
        for actor, total in actor_total.items():
            likes = actor_likes.get(actor, 0)
            self.actor_preference[actor] = (likes + 1) / (total + 2)
    
    def infer_recommendation_probability(self, movie: Dict) -> float:
        """
        Use Bayesian inference to compute P(recommend | movie features)
        
        P(recommend | genres, cast) ∝ P(genres | recommend) * P(cast | recommend) * P(recommend)
        """
        
        genres = movie.get('genres', [])
        cast = movie.get('cast', [])
        rating = movie.get('rating', 5.0)
        
        # Start with prior
        prob = self.recommendation_prior
        
        # Update based on genres
        if genres:
            genre_probs = [self.genre_preference.get(g, 0.5) for g in genres]
            avg_genre_prob = np.mean(genre_probs)
            prob *= avg_genre_prob
        
        # Update based on cast
        if cast:
            cast_probs = [self.actor_preference.get(a, 0.5) for a in cast]
            avg_cast_prob = np.mean(cast_probs)
            prob *= avg_cast_prob
        
        # Update based on rating
        if rating >= self.rating_threshold:
            prob *= 0.8
        else:
            prob *= 0.2
        
        # Normalize
        return min(1.0, max(0.0, prob))


class HiddenMarkovModel:
    """
    HMM for tracking hidden preference states over time.
    Hidden states: {Enthusiastic, Moderate, Selective}
    Observable events: {like, dislike, neutral}
    
    Tracks how user's taste evolves (e.g., from romance -> thrillers)
    """
    
    def __init__(self):
        self.states = ['Enthusiastic', 'Moderate', 'Selective']
        self.observations = ['like', 'dislike', 'neutral']
        
        # Transition probabilities P(state_t+1 | state_t)
        self.transition_matrix = np.array([
            [0.7, 0.2, 0.1],   # From Enthusiastic
            [0.3, 0.4, 0.3],   # From Moderate
            [0.1, 0.3, 0.6]    # From Selective
        ])
        
        # Emission probabilities P(obs | state)
        self.emission_matrix = np.array([
            [0.8, 0.1, 0.1],   # Enthusiastic emits likes
            [0.4, 0.3, 0.3],   # Moderate mixed
            [0.2, 0.5, 0.3]    # Selective dislikes more
        ])
        
        # Initial state distribution
        self.initial_dist = np.array([0.33, 0.33, 0.34])
        
        self.current_state_dist = self.initial_dist.copy()
    
    def forward_algorithm(self, observations: List[int]) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm: compute P(observations | model)
        
        Args:
            observations: List of observation indices (0=like, 1=dislike, 2=neutral)
        
        Returns:
            (forward_values, likelihood)
        """
        n_states = len(self.states)
        n_obs = len(observations)
        
        forward = np.zeros((n_obs, n_states))
        
        # Initial step
        for state in range(n_states):
            forward[0, state] = self.initial_dist[state] * self.emission_matrix[state, observations[0]]
        
        # Recursive step
        for t in range(1, n_obs):
            for state in range(n_states):
                forward[t, state] = np.sum(
                    forward[t-1, :] * self.transition_matrix[:, state]
                ) * self.emission_matrix[state, observations[t]]
        
        likelihood = np.sum(forward[-1, :])
        return forward, likelihood
    
    def viterbi_algorithm(self, observations: List[int]) -> List[int]:
        """
        Viterbi algorithm: find most likely sequence of hidden states.
        
        Returns:
            Sequence of most likely state indices
        """
        n_states = len(self.states)
        n_obs = len(observations)
        
        viterbi = np.zeros((n_obs, n_states))
        path_tracker = np.zeros((n_obs, n_states), dtype=int)
        
        # Initialize
        for state in range(n_states):
            viterbi[0, state] = self.initial_dist[state] * self.emission_matrix[state, observations[0]]
        
        # Forward pass
        for t in range(1, n_obs):
            for state in range(n_states):
                temp = viterbi[t-1, :] * self.transition_matrix[:, state]
                path_tracker[t, state] = np.argmax(temp)
                viterbi[t, state] = np.max(temp) * self.emission_matrix[state, observations[t]]
        
        # Backtrack
        path = []
        state = np.argmax(viterbi[-1, :])
        path.append(state)
        
        for t in range(n_obs - 1, 0, -1):
            state = path_tracker[t, state]
            path.append(state)
        
        path.reverse()
        return path
    
    def baum_welch_learning(self, observations: List[int], iterations: int = 10):
        """
        Baum-Welch algorithm: Learn HMM parameters from observations.
        Updates transition and emission matrices based on data.
        """
        for _ in range(iterations):
            forward, _ = self.forward_algorithm(observations)
            # Backward pass and parameter update would go here
            # Simplified version for demonstration
            pass
    
    def update_preference_state(self, recent_ratings: List[float]):
        """Update hidden state based on recent ratings."""
        # Convert ratings to observations
        obs = []
        threshold = 7.0
        
        for rating in recent_ratings[-5:]:  # Look at last 5 ratings
            if rating >= threshold:
                obs.append(0)  # like
            elif rating < 5:
                obs.append(1)  # dislike
            else:
                obs.append(2)  # neutral
        
        if obs:
            # Run Viterbi to find most likely state
            state_path = self.viterbi_algorithm(obs)
            final_state = state_path[-1]
            
            # Update state distribution
            self.current_state_dist = np.zeros(len(self.states))
            self.current_state_dist[final_state] = 1.0


class KalmanFilter:
    """
    Kalman Filter for refining belief about user preferences as new ratings arrive.
    Tracks evolving preference state with uncertainty quantification.
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.state = np.array([0.5])  # Initial preference level [0, 1]
        self.covariance = np.array([[1.0]])
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def predict(self, dt: float = 1.0):
        """Predict next state (with process noise)."""
        # Assume state doesn't change much between observations
        self.covariance += self.process_noise
    
    def update(self, measurement: float, measurement_uncertainty: float = 0.1):
        """
        Update state based on new rating observation.
        
        Args:
            measurement: New rating value (0-1 scale)
            measurement_uncertainty: Confidence in this measurement
        """
        # Kalman gain
        K = self.covariance / (self.covariance + measurement_uncertainty)
        
        # Update state estimate
        self.state = self.state + K * (measurement - self.state)
        
        # Update covariance
        self.covariance = (1 - K) * self.covariance
    
    def get_belief(self) -> Tuple[float, float]:
        """Return current belief state and uncertainty."""
        return float(self.state[0]), float(np.sqrt(self.covariance[0, 0]))


class MCMCSampler:
    """
    MCMC (Markov Chain Monte Carlo) for approximate inference
    when exact probability computation is infeasible.
    """
    
    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
        self.samples = []
    
    def metropolis_hastings_sampling(self, likelihood_fn, 
                                     initial_state: Dict, 
                                     proposal_std: float = 0.1) -> List[Dict]:
        """
        Metropolis-Hastings MCMC sampling.
        
        Args:
            likelihood_fn: Function computing likelihood of a state
            initial_state: Starting point for chain
            proposal_std: Standard deviation of proposal distribution
        
        Returns:
            List of samples from posterior distribution
        """
        samples = []
        current_state = initial_state.copy()
        current_likelihood = likelihood_fn(current_state)
        
        for _ in range(self.n_samples):
            # Propose new state
            proposed_state = current_state.copy()
            for key in proposed_state:
                if isinstance(proposed_state[key], (int, float)):
                    proposed_state[key] += np.random.normal(0, proposal_std)
                    # Clamp to valid range
                    proposed_state[key] = np.clip(proposed_state[key], 0, 1)
            
            # Compute acceptance ratio
            proposed_likelihood = likelihood_fn(proposed_state)
            acceptance_ratio = proposed_likelihood / (current_likelihood + 1e-10)
            
            # Accept/reject
            if np.random.uniform() < acceptance_ratio:
                current_state = proposed_state
                current_likelihood = proposed_likelihood
            
            samples.append(current_state.copy())
        
        self.samples = samples
        return samples
    
    def get_posterior_mean(self) -> Dict:
        """Compute posterior mean from samples."""
        if not self.samples:
            return {}
        
        keys = self.samples[0].keys()
        posterior = {}
        
        for key in keys:
            values = [s[key] for s in self.samples[len(self.samples)//2:]]  # Discard burn-in
            posterior[key] = np.mean(values)
        
        return posterior
