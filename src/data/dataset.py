"""
Data structures and sample dataset generation for the movie recommendation system.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Movie:
    """Represents a movie with its features."""
    movie_id: int
    title: str
    genres: List[str]
    rating: float  # 1-10
    cast: List[str]
    keywords: List[str]
    year: int
    runtime: int

@dataclass
class User:
    """Represents a user with their preference history."""
    user_id: int
    name: str
    rated_movies: Dict[int, float]  # movie_id -> rating given

class MovieDataset:
    """Manages movie and user data."""
    
    def __init__(self):
        self.movies = {}
        self.users = {}
        self.ratings_history = []
    
    def add_movie(self, movie: Movie):
        """Add a movie to the dataset."""
        self.movies[movie.movie_id] = movie
    
    def add_user(self, user: User):
        """Add a user to the dataset."""
        self.users[user.user_id] = user
    
    def add_rating(self, user_id: int, movie_id: int, rating: float):
        """Record a user's rating for a movie."""
        if user_id in self.users and movie_id in self.movies:
            self.users[user_id].rated_movies[movie_id] = rating
            self.ratings_history.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating
            })
    
    def get_user_preferences(self, user_id: int) -> Dict[int, float]:
        """Get all ratings from a user."""
        if user_id in self.users:
            return self.users[user_id].rated_movies
        return {}
    
    def get_movie_features(self, movie_id: int) -> Movie:
        """Get features of a movie."""
        return self.movies.get(movie_id)
    
    def to_dataframe(self):
        """Convert ratings to DataFrame format."""
        return pd.DataFrame(self.ratings_history)


def generate_sample_dataset():
    """Generate a sample movie dataset for testing."""
    
    dataset = MovieDataset()
    
    # Sample movies
    movies_data = [
        Movie(1, "Inception", ["Sci-Fi", "Thriller", "Action"], 8.8, ["Leonardo DiCaprio", "Marion Cotillard"], ["heist", "dreams", "mind-bending"], 2010, 148),
        Movie(2, "Interstellar", ["Sci-Fi", "Drama"], 8.6, ["Matthew McConaughey", "Anne Hathaway"], ["space", "time", "dimensions"], 2014, 169),
        Movie(3, "The Dark Knight", ["Action", "Crime", "Thriller"], 9.0, ["Christian Bale", "Heath Ledger"], ["superhero", "justice", "chaos"], 2008, 152),
        Movie(4, "The Shawshank Redemption", ["Drama"], 9.3, ["Tim Robbins", "Morgan Freeman"], ["prison", "hope", "redemption"], 1994, 142),
        Movie(5, "Titanic", ["Romance", "Drama"], 7.8, ["Leonardo DiCaprio", "Kate Winslet"], ["ship", "love", "disaster"], 1997, 194),
        Movie(6, "Avatar", ["Sci-Fi", "Action", "Adventure"], 7.8, ["Sam Worthington", "Zoe Saldana"], ["aliens", "nature", "3D"], 2009, 162),
        Movie(7, "The Notebook", ["Romance", "Drama"], 7.8, ["Ryan Gosling", "Rachel McAdams"], ["love", "memory", "emotions"], 2004, 123),
        Movie(8, "The Matrix", ["Sci-Fi", "Action"], 8.7, ["Keanu Reeves", "Laurence Fishburne"], ["virtual-reality", "reality", "awakening"], 1999, 136),
        Movie(9, "Forrest Gump", ["Drama", "Romance"], 8.8, ["Tom Hanks", "Gary Sinise"], ["life-story", "love", "destiny"], 1994, 142),
        Movie(10, "Gladiator", ["Action", "Adventure", "Drama"], 8.5, ["Russell Crowe", "Joaquin Phoenix"], ["ancient-rome", "revenge", "empire"], 2000, 155),
        Movie(11, "The Conjuring", ["Horror", "Mystery"], 7.5, ["Vera Farmiga", "Patrick Wilson"], ["haunting", "supernatural", "demons"], 2013, 112),
        Movie(12, "Dune", ["Sci-Fi", "Adventure", "Drama"], 8.0, ["Timothée Chalamet", "Zendaya"], ["desert", "politics", "destiny"], 2021, 166),
    ]
    
    for movie in movies_data:
        dataset.add_movie(movie)
    
    # Sample users with their ratings
    users_data = [
        (1, "Alice", {1: 9, 2: 8.5, 3: 8, 8: 9, 12: 7}),
        (2, "Bob", {4: 9, 9: 9, 5: 6, 7: 5, 11: 4}),
        (3, "Charlie", {1: 8.5, 2: 8, 3: 9, 6: 7, 10: 8}),
        (4, "Diana", {5: 8, 7: 8.5, 9: 7, 4: 8, 2: 6}),
        (5, "Eve", {1: 9, 2: 9, 8: 8.5, 12: 8, 3: 8}),
        (6, "Frank", {4: 9.2, 9: 8.8, 10: 8, 3: 7.5, 1: 6}),
    ]
    
    for user_id, name, ratings in users_data:
        user = User(user_id, name, {})
        dataset.add_user(user)
        for movie_id, rating in ratings.items():
            dataset.add_rating(user_id, movie_id, rating)
    
    return dataset
