"""
Comprehensive test suite for the Movie Recommendation System
"""

import unittest
import numpy as np
from src.data.dataset import generate_sample_dataset, MovieDataset, Movie
from src.modules.module1_concept_learning import ConceptLearner
from src.modules.module2_rule_learning import RuleLearner, Rule
from src.modules.module3_clustering_ensemble import MovieClusterer, EnsemblePredictor
from src.modules.module5_probabilistic_models import BayesianPreferenceNetwork, HiddenMarkovModel, KalmanFilter
from src.recommender_system import MovieRecommender


class TestModule1ConceptLearning(unittest.TestCase):
    """Test Module 1: Concept Learning"""
    
    def setUp(self):
        self.learner = ConceptLearner()
    
    def test_learn_preferences(self):
        """Test learning user preference concept"""
        positive_examples = [
            {'genres': ['Sci-Fi', 'Thriller'], 'rating': 8.5, 'cast': ['Actor1']},
            {'genres': ['Sci-Fi', 'Action'], 'rating': 8.0, 'cast': ['Actor2']},
        ]
        negative_examples = [
            {'genres': ['Romance'], 'rating': 6.0, 'cast': ['Actor3']},
        ]
        
        concept = self.learner.learn_preferences(positive_examples, negative_examples)
        
        self.assertIn('genres', concept)
        self.assertIn('min_rating', concept)
        self.assertGreater(len(concept.get('genres', [])), 0)
    
    def test_predict_preference(self):
        """Test preference prediction"""
        hypothesis = {
            'genres': ['Sci-Fi'],
            'min_rating': 7.0,
            'cast': []
        }
        
        movie = {'genres': ['Sci-Fi', 'Action'], 'rating': 8.0, 'cast': []}
        score = self.learner.predict_preference(movie, hypothesis)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.5)  # Should be high confidence


class TestModule2RuleLearning(unittest.TestCase):
    """Test Module 2: Rule Learning"""
    
    def setUp(self):
        self.learner = RuleLearner()
    
    def test_sequential_covering(self):
        """Test sequential covering algorithm"""
        positive_examples = [
            {'genres': ['Sci-Fi'], 'rating': 8.5, 'cast': []},
            {'genres': ['Sci-Fi'], 'rating': 8.0, 'cast': []},
        ]
        negative_examples = [
            {'genres': ['Romance'], 'rating': 6.0, 'cast': []},
        ]
        
        rules = self.learner.sequential_covering(positive_examples, negative_examples)
        
        self.assertIsInstance(rules, list)
        for rule in rules:
            self.assertIsInstance(rule, Rule)


class TestModule3Clustering(unittest.TestCase):
    """Test Module 3: Clustering & Ensemble"""
    
    def setUp(self):
        self.clusterer = MovieClusterer(n_clusters=2)
    
    def test_k_means_clustering(self):
        """Test K-means clustering"""
        movies = [
            {'movie_id': 1, 'rating': 8.5, 'genres': ['Sci-Fi'], 'cast': [], 'keywords': [], 'year': 2020, 'runtime': 120},
            {'movie_id': 2, 'rating': 8.0, 'genres': ['Sci-Fi'], 'cast': [], 'keywords': [], 'year': 2021, 'runtime': 130},
            {'movie_id': 3, 'rating': 5.0, 'genres': ['Romance'], 'cast': [], 'keywords': [], 'year': 2019, 'runtime': 100},
            {'movie_id': 4, 'rating': 4.5, 'genres': ['Romance'], 'cast': [], 'keywords': [], 'year': 2020, 'runtime': 110},
        ]
        
        clusters = self.clusterer.fit(movies)
        
        self.assertEqual(len(clusters), 2)
        self.assertEqual(sum(len(v) for v in clusters.values()), 4)
    
    def test_similarity_search(self):
        """Test finding similar movies"""
        movies = [
            {'movie_id': 1, 'rating': 8.5, 'genres': ['Sci-Fi'], 'cast': [], 'keywords': [], 'year': 2020, 'runtime': 120},
            {'movie_id': 2, 'rating': 8.0, 'genres': ['Sci-Fi'], 'cast': [], 'keywords': [], 'year': 2021, 'runtime': 130},
            {'movie_id': 3, 'rating': 5.0, 'genres': ['Romance'], 'cast': [], 'keywords': [], 'year': 2019, 'runtime': 100},
        ]
        
        self.clusterer.fit(movies)
        similar = self.clusterer.get_similar_movies(1, n_similar=2)
        
        self.assertIsInstance(similar, list)


class TestModule5ProbabilisticModels(unittest.TestCase):
    """Test Module 5: Probabilistic Models"""
    
    def test_bayesian_network(self):
        """Test Bayesian network learning"""
        bayes = BayesianPreferenceNetwork()
        
        user_ratings = {1: 8.0, 2: 8.5, 3: 5.0}
        movies = {
            1: Movie(1, "Movie1", ["Sci-Fi"], 8.5, ["Actor1"], [], 2020, 120),
            2: Movie(2, "Movie2", ["Sci-Fi"], 8.0, ["Actor2"], [], 2021, 130),
            3: Movie(3, "Movie3", ["Romance"], 5.0, ["Actor3"], [], 2019, 100),
        }
        
        bayes.learn_from_ratings(user_ratings, movies)
        
        # Test inference
        test_movie = Movie(4, "Movie4", ["Sci-Fi"], 8.0, ["Actor1"], [], 2022, 125)
        prob = bayes.infer_recommendation_probability(test_movie)
        
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
    
    def test_hidden_markov_model(self):
        """Test HMM"""
        hmm = HiddenMarkovModel()
        
        # Test forward algorithm
        observations = [0, 0, 1]  # like, like, dislike
        forward, likelihood = hmm.forward_algorithm(observations)
        
        self.assertEqual(forward.shape, (3, 3))
        self.assertGreater(likelihood, 0)
    
    def test_viterbi_algorithm(self):
        """Test Viterbi decoding"""
        hmm = HiddenMarkovModel()
        
        observations = [0, 0, 1]
        path = hmm.viterbi_algorithm(observations)
        
        self.assertEqual(len(path), 3)
        for state in path:
            self.assertIn(state, [0, 1, 2])
    
    def test_kalman_filter(self):
        """Test Kalman filter"""
        kf = KalmanFilter()
        
        # Simulate rating updates
        for rating in [0.7, 0.75, 0.8]:
            kf.predict()
            kf.update(rating)
        
        belief, uncertainty = kf.get_belief()
        
        self.assertGreaterEqual(belief, 0.0)
        self.assertLessEqual(belief, 1.0)
        self.assertGreater(uncertainty, 0.0)


class TestMovieRecommender(unittest.TestCase):
    """Test integrated recommender system"""
    
    def setUp(self):
        self.dataset = generate_sample_dataset()
    
    def test_initialization(self):
        """Test recommender initialization"""
        recommender = MovieRecommender(self.dataset, user_id=1)
        
        self.assertIsNotNone(recommender.learned_concept)
        self.assertIsNotNone(recommender.rules)
    
    def test_recommendations(self):
        """Test recommendation generation"""
        recommender = MovieRecommender(self.dataset, user_id=1)
        recs = recommender.recommend(n_recommendations=3)
        
        self.assertEqual(len(recs), 3)
        
        for rec in recs:
            self.assertGreaterEqual(rec.confidence, 0.0)
            self.assertLessEqual(rec.confidence, 1.0)
            self.assertIsNotNone(rec.movie_title)
            self.assertIsNotNone(rec.explanation)
    
    def test_feedback_integration(self):
        """Test adding user feedback"""
        recommender = MovieRecommender(self.dataset, user_id=1)
        
        initial_ratings = len(self.dataset.users[1].rated_movies)
        recommender.add_feedback(10, 9.0)
        new_ratings = len(self.dataset.users[1].rated_movies)
        
        self.assertEqual(new_ratings, initial_ratings + 1)
    
    def test_system_insights(self):
        """Test system insights"""
        recommender = MovieRecommender(self.dataset, user_id=1)
        insights = recommender.get_system_insights()
        
        self.assertIn('current_preference_state', insights)
        self.assertIn('preference_certainty', insights)
        self.assertIn('learned_genres', insights)


class TestDataset(unittest.TestCase):
    """Test dataset generation"""
    
    def test_sample_dataset_generation(self):
        """Test sample dataset generation"""
        dataset = generate_sample_dataset()
        
        self.assertGreater(len(dataset.movies), 0)
        self.assertGreater(len(dataset.users), 0)
        self.assertGreater(len(dataset.ratings_history), 0)
    
    def test_add_rating(self):
        """Test adding ratings"""
        dataset = MovieDataset()
        dataset.add_movie(Movie(1, "Test", ["Test"], 8.0, [], [], 2020, 120))
        dataset.add_user(type('User', (), {'user_id': 1, 'name': 'Test', 'rated_movies': {}})())
        
        dataset.add_rating(1, 1, 8.5)
        
        self.assertIn(1, dataset.users[1].rated_movies)
        self.assertEqual(dataset.users[1].rated_movies[1], 8.5)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModule1ConceptLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestModule2RuleLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestModule3Clustering))
    suite.addTests(loader.loadTestsFromTestCase(TestModule5ProbabilisticModels))
    suite.addTests(loader.loadTestsFromTestCase(TestMovieRecommender))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
