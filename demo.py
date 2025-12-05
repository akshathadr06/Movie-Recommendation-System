"""
Command-line interface and demonstration of the Movie Recommendation System
"""

import sys
from typing import List
from src.data.dataset import generate_sample_dataset
from src.recommender_system import MovieRecommender


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_separator():
    """Print separator line."""
    print(f"{'-'*70}\n")


def demo_module1_concept_learning():
    """Demonstrate Module 1: Concept Learning"""
    print_header("MODULE 1: CONCEPT LEARNING & HYPOTHESIS SEARCH")
    
    dataset = generate_sample_dataset()
    
    # Show concept learning for User 1 (Alice - likes Sci-Fi)
    user_id = 1
    recommender = MovieRecommender(dataset, user_id)
    
    print(f"User ID: {user_id}")
    print(f"Movies they've rated:")
    for movie_id, rating in dataset.users[user_id].rated_movies.items():
        movie = dataset.movies[movie_id]
        print(f"  • {movie.title} ({rating}/10) - Genres: {', '.join(movie.genres)}")
    
    print_separator()
    print("Learned Concept (Preference Pattern):")
    concept = recommender.learned_concept
    print(f"  • Preferred Genres: {concept.get('genres', [])}")
    print(f"  • Minimum Rating: {concept.get('min_rating', 'N/A'):.1f}/10")
    print(f"  • Favorite Actors: {concept.get('cast', [])}")
    
    print_separator()
    print("Inductive Bias Explanation:")
    print(recommender.concept_learner.inductive_bias_assumption(concept))


def demo_module2_rule_learning():
    """Demonstrate Module 2: Rule Learning"""
    print_header("MODULE 2: RULE LEARNING & ANALYTICAL REASONING")
    
    dataset = generate_sample_dataset()
    user_id = 1
    recommender = MovieRecommender(dataset, user_id)
    
    print(f"User ID: {user_id}")
    print(f"\nLearned Recommendation Rules:")
    
    rules = recommender.rules
    if rules:
        for i, rule in enumerate(rules, 1):
            print(f"\n  Rule {i}:")
            print(f"    {rule}")
    else:
        print("  (No rules generated - need more diverse data)")
    
    print_separator()
    print("Explanation-Based Learning (Domain Theory Application):")
    user_movies = [dataset.movies[mid] for mid in dataset.users[user_id].rated_movies.keys()]
    explanations = recommender.rule_learner.explanation_based_learning(
        [recommender._movie_to_features(m) for m in user_movies],
        {}
    )
    for exp in explanations:
        print(exp)


def demo_module3_clustering_ensemble():
    """Demonstrate Module 3: Clustering & Ensemble"""
    print_header("MODULE 3: CLUSTERING, SIMILARITY & ENSEMBLE PREDICTION")
    
    dataset = generate_sample_dataset()
    
    print("Movie Clustering Results (K-Means):")
    movies_list = list(dataset.movies.values())
    clusters = dataset.movies
    
    user_id = 1
    recommender = MovieRecommender(dataset, user_id)
    
    print("\nClusters:")
    for cluster_id, movie_ids in recommender.movie_clusterer.clusters.items():
        movies_in_cluster = [dataset.movies[mid] for mid in movie_ids]
        print(f"\n  Cluster {cluster_id}:")
        for movie in movies_in_cluster:
            print(f"    • {movie.title} (Rating: {movie.rating}, Genres: {movie.genres})")
    
    print_separator()
    print("Ensemble Predictor - Feature Importance:")
    importance = recommender.ensemble_predictor.get_feature_importance()
    if importance:
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {feature}: {imp:.3f}")
    
    print_separator()
    print("Similarity-Based Recommendations (from clustering):")
    if dataset.users[user_id].rated_movies:
        first_liked = list(dataset.users[user_id].rated_movies.keys())[0]
        similar = recommender.movie_clusterer.get_similar_movies(first_liked, n_similar=3)
        if similar:
            print(f"  Similar to '{dataset.movies[first_liked].title}':")
            for sim_id in similar:
                print(f"    • {dataset.movies[sim_id].title}")


def demo_module5_probabilistic():
    """Demonstrate Module 5: Probabilistic Modeling"""
    print_header("MODULE 5: PROBABILISTIC MODELING & SEQUENTIAL BEHAVIOR")
    
    dataset = generate_sample_dataset()
    user_id = 1
    recommender = MovieRecommender(dataset, user_id)
    
    print("Bayesian Network Learning:")
    bayes_net = recommender.bayesian_network
    print(f"\n  Genre Preferences (P(like | genre)):")
    for genre, prob in sorted(bayes_net.genre_preference.items(), 
                             key=lambda x: x[1], reverse=True)[:5]:
        print(f"    • {genre}: {prob:.2%}")
    
    print(f"\n  Actor Preferences (P(like | actor)):")
    for actor, prob in sorted(bayes_net.actor_preference.items(), 
                             key=lambda x: x[1], reverse=True)[:3]:
        print(f"    • {actor}: {prob:.2%}")
    
    print_separator()
    print("Hidden Markov Model - Preference State Tracking:")
    print(f"  Current Hidden State: {recommender.hmm.states[np.argmax(recommender.hmm.current_state_dist)]}")
    print(f"  State Distribution: {recommender.hmm.current_state_dist}")
    
    print_separator()
    print("Kalman Filter - Preference Belief Tracking:")
    belief, uncertainty = recommender.kalman_filter.get_belief()
    print(f"  Current Belief (normalized): {belief:.3f}")
    print(f"  Uncertainty: {uncertainty:.3f}")
    print(f"  Confidence: {(1 - uncertainty)*100:.1f}%")


def demo_recommendations():
    """Demonstrate full recommendation pipeline"""
    print_header("FULL RECOMMENDATION PIPELINE")
    
    dataset = generate_sample_dataset()
    
    for user_id in [1, 2, 3]:
        user = dataset.users[user_id]
        print(f"\n{'*'*70}")
        print(f"GENERATING RECOMMENDATIONS FOR USER: {user_id}")
        print(f"{'*'*70}\n")
        
        recommender = MovieRecommender(dataset, user_id)
        
        print(f"User {user_id}'s Rating History:")
        for movie_id, rating in user.rated_movies.items():
            movie = dataset.movies[movie_id]
            stars = "⭐" * int(rating)
            print(f"  {stars} {movie.title} ({rating}/10)")
        
        print_separator()
        print("RECOMMENDATIONS WITH EXPLANATIONS:\n")
        
        recommendations = recommender.recommend(n_recommendations=3)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.movie_title}")
            print(f"   Confidence: {rec.confidence*100:.1f}%")
            print(f"   {rec.explanation}")
            
            print(f"\n   Reasoning Breakdown:")
            for module, score in rec.reasoning.items():
                print(f"     • {module}: {score:.3f}")
            print()


def demo_interactive():
    """Interactive demo where user can rate movies and get recommendations"""
    print_header("INTERACTIVE RECOMMENDATION SYSTEM")
    
    dataset = generate_sample_dataset()
    
    print("Available movies:")
    for movie in dataset.movies.values():
        print(f"  {movie.movie_id}. {movie.title} ({movie.rating}/10)")
    
    user_id = 6
    user = dataset.users[user_id]
    recommender = MovieRecommender(dataset, user_id)
    
    print(f"\nUsing test user: {user_id}")
    
    while True:
        print_separator()
        print("Options: 1=Get recommendations  2=Add rating  3=System insights  4=Exit")
        choice = input("Choose: ").strip()
        
        if choice == '1':
            print("\nGenerating recommendations...")
            recs = recommender.recommend(n_recommendations=5)
            
            if not recs:
                print("No new movies available to recommend.")
                continue
            
            for i, rec in enumerate(recs, 1):
                print(f"\n{i}. {rec.movie_title}")
                print(f"   Confidence: {rec.confidence*100:.1f}%")
                print(f"   {rec.explanation}")
        
        elif choice == '2':
            movie_id = input("Enter movie ID to rate: ").strip()
            try:
                movie_id = int(movie_id)
                rating = float(input("Enter rating (1-10): "))
                recommender.add_feedback(movie_id, rating)
                print("Rating added and models updated!")
            except:
                print("Invalid input")
        
        elif choice == '3':
            insights = recommender.get_system_insights()
            print("\nSystem Insights:")
            for key, value in insights.items():
                print(f"  • {key}: {value}")
        
        elif choice == '4':
            print("Exiting...")
            break


if __name__ == "__main__":
    import numpy as np
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MOVIE RECOMMENDATION SYSTEM DEMO" + " "*21 + "║")
    print("║" + " "*12 + "Integrating ML Algorithms Across All Modules" + " "*12 + "║")
    print("╚" + "="*68 + "╝")
    
    print("""
This system demonstrates:
✓ Module 1: Concept Learning - Identify user preference patterns
✓ Module 2: Rule Learning - Generate explainable recommendation rules
✓ Module 3: Clustering & Ensemble - Group similar movies and ensemble predictions
✓ Module 5: Probabilistic Models - Bayesian networks, HMM, Kalman filters, MCMC
    """)
    
    print("\nSelect a demonstration:")
    print("  1 - Module 1: Concept Learning")
    print("  2 - Module 2: Rule Learning")
    print("  3 - Module 3: Clustering & Ensemble")
    print("  4 - Module 5: Probabilistic Modeling")
    print("  5 - Full Recommendation Pipeline")
    print("  6 - Interactive Demo")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        demo_module1_concept_learning()
    elif choice == '2':
        demo_module2_rule_learning()
    elif choice == '3':
        demo_module3_clustering_ensemble()
    elif choice == '4':
        demo_module5_probabilistic()
    elif choice == '5':
        demo_recommendations()
    elif choice == '6':
        try:
            demo_interactive()
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
    else:
        print("Running full pipeline demonstration...")
        demo_module1_concept_learning()
        demo_module2_rule_learning()
        demo_module3_clustering_ensemble()
        demo_module5_probabilistic()
        demo_recommendations()
    
    print("\n" + "="*70)
    print("Thank you for using the Movie Recommendation System!")
    print("="*70 + "\n")
