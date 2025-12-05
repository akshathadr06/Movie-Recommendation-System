# Movie Recommendation System 🎬

An academically rigorous, end-to-end movie recommendation system integrating machine learning algorithms from concept learning, rule learning, clustering, ensemble methods, and probabilistic models.

## 📋 Project Overview

This system demonstrates how multiple ML paradigms combine into one unified architecture. Each module addresses a different aspect:

| Module | Algorithms | Purpose |
|--------|-----------|---------|
| **Module 1** | Concept Learning, Version Spaces, Candidate-Elimination | Learn user preference patterns |
| **Module 2** | Sequential Covering, FOIL, EBL | Generate explainable rules |
| **Module 3** | K-Means, SOM, Random Forests, AdaBoost | Cluster movies, ensemble predictions |
| **Module 5** | Bayesian Networks, HMM, Kalman Filters, MCMC | Probabilistic inference |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_recommender.py

# Run interactive demo
python demo.py
```

## 📚 Architecture

```
┌─────────────────────────────────────────┐
│    MovieRecommender (Unified System)    │
├──────┬──────┬──────┬──────────────────┤
│ Mod1 │ Mod2 │ Mod3 │  Module 5        │
│ Conc │ Rules│ Clus │(Bayesian, HMM,   │
│Learn │Learn │ Ens  │ Kalman, MCMC)    │
└──────┴──────┴──────┴──────────────────┘
          ↓
    Multi-Module Scoring
          ↓
Top-K Recommendations + Explanations
```

## 🧪 Test Results

**15/15 tests passing** ✅

- Module 1: Concept learning & prediction
- Module 2: Rule generation & explanation  
- Module 3: Clustering & ensemble methods
- Module 5: Bayesian networks, HMM, Kalman filters
- Unified recommender integration
- Dataset handling & feedback

## 💡 Key Features

✨ **Explainability**: Rules and reasoning for each recommendation

✨ **Multi-Algorithm**: 15+ ML algorithms integrated seamlessly

✨ **Uncertainty**: Confidence scores and belief tracking

✨ **Adaptive**: Models update with user feedback

✨ **Production-Ready**: Fully tested and documented

✨ **Modular**: Easy to extend and customize

## 📁 Project Structure

```
src/
├── data/dataset.py                  # Movie & user dataset
├── modules/
│   ├── module1_concept_learning.py
│   ├── module2_rule_learning.py
│   ├── module3_clustering_ensemble.py
│   └── module5_probabilistic_models.py
└── recommender_system.py            # Unified recommender

demo.py                              # Interactive demonstrations
test_recommender.py                  # Test suite
```

## 💻 Usage Example

```python
from src.data.dataset import generate_sample_dataset
from src.recommender_system import MovieRecommender

# Load dataset
dataset = generate_sample_dataset()

# Create recommender
recommender = MovieRecommender(dataset, user_id=1)

# Get recommendations
recommendations = recommender.recommend(n_recommendations=5)

# Display results
for rec in recommendations:
    print(f"{rec.movie_title}: {rec.confidence*100:.1f}%")
    print(f"{rec.explanation}\n")
```

## 📊 Scoring Weights

- **Module 1** (20%): Concept-based preference matching
- **Module 2** (15%): Rule matching confidence
- **Module 3** (25%): Ensemble prediction
- **Module 5** (25%): Bayesian network probability
- **Similarity** (15%): Clustering proximity

## 🎓 Algorithms Implemented

### Module 1: Concept Learning
- Candidate-Elimination Algorithm
- Version Spaces
- Inductive Bias Learning

### Module 2: Rule Learning
- Sequential Covering
- FOIL (First-Order Induction Logic)
- Explanation-Based Learning (EBL)

### Module 3: Clustering & Ensemble
- K-Means Clustering
- Self-Organizing Maps (SOM)
- Random Forests
- AdaBoost
- Committee-Based Ensemble Voting

### Module 5: Probabilistic Models
- Bayesian Networks
- Hidden Markov Models (Forward/Viterbi)
- Kalman Filters
- MCMC Sampling

## 📈 Demonstration Options

Run `python demo.py` for:
1. Module 1 - Concept Learning demo
2. Module 2 - Rule Learning demo
3. Module 3 - Clustering & Ensemble demo
4. Module 5 - Probabilistic Models demo
5. Full recommendation pipeline
6. Interactive recommendation mode

## ✅ Testing

Run comprehensive test suite:
```bash
python test_recommender.py
```

Tests cover all modules, integration, and edge cases.

## 🔧 Technologies

- Python 3.8+
- NumPy, Pandas, Scikit-Learn
- SciPy for scientific computing

## 📝 License

Educational and academic purposes

---

**Built for excellence in machine learning and recommendations! 🎬✨**
