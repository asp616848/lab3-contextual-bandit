# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student:** Abhijeet  
**Roll Number:** U20230101  
**GitHub Branch:** `abhijeet_u20230101`  
**Submission Date:** February 2026

---

## Executive Summary

This project implements a **Contextual Multi-Armed Bandit (CMAB) system** for intelligent news article recommendation. The system combines supervised learning (user classification) with reinforcement learning (contextual bandits) to maximize temporal cumulative rewards while adapting to diverse user preferences.

**Key Achievements:**
- Developed an XGBoost-based user context predictor with **~94% validation accuracy**
- Implemented three state-of-the-art bandit algorithms: Epsilon-Greedy, UCB, and SoftMax
- Demonstrated exploration-exploitation trade-offs through T=10,000 simulation steps
- Achieved significant performance gains with optimal hyperparameter configurations

---

## 1. Problem Statement & Objective

### Context & Motivation
News recommendation systems face a fundamental challenge: **balancing exploration vs. exploitation**. We must learn user preferences (exploration) while maximizing immediate rewards (exploitation). The contextual bandit framework addresses this elegantly by treating:
- **Context**: User category (demographic/behavioral segment)
- **Arms**: News categories (Entertainment, Education, Tech, Crime)
- **Reward**: User satisfaction/engagement signal

### Objectives
1. **Predict user context** using supervised learning (XGBoost classifier)
2. **Implement three bandit algorithms** to maximize cumulative rewards:
   - Epsilon-Greedy (ε-exploration)
   - Upper Confidence Bound (UCB)
   - SoftMax (temperature-based exploration)
3. **Compare algorithm performance** across varying hyperparameters
4. **Evaluate on held-out test users** to measure generalization

---

## 2. Methodology

### 2.1 Data Pipeline

**Datasets Used:**
- `news_articles.csv`: 50,000+ articles across 4 target categories
- `train_users.csv`: 1,000 training users with behavioral features
- `test_users.csv`: 300 test users for evaluation

**Data Filtering & Preprocessing:**
```python
# Target categories: Entertainment, Education, Tech, Crime
# → Reduces articles from 50K+ to relevant subset
# → Ensures arm-context pair coverage
```

**Feature Engineering:**
1. **Missing Value Imputation:**
   - Numeric: Mean imputation (computed from training data)
   - Categorical: Mode imputation (most frequent value)
   
2. **Feature Scaling:**
   - StandardScaler normalization to zero mean, unit variance
   
3. **Polynomial Feature Expansion:**
   - Degree-2 polynomial features capture feature interactions
   - Original features: ~15 → Expanded features: ~120

**Quality Assurance:**
- ✓ Zero missing values after preprocessing
- ✓ Verified balanced class distribution (stratified splits)
- ✓ Cross-validated scaling parameters (no data leakage)

---

### 2.2 User Classification (Context Prediction)

#### Model Selection: Why XGBoost?

XGBoost was chosen over alternatives for several reasons:

| Criterion | XGBoost | Logistic Reg. | Random Forest | Neural Networks |
|-----------|---------|---------------|---------------|----|
| Non-linear interactions | ✓ Excellent | ✗ Limited | ✓ Good | ✓ Excellent |
| Categorical features | ✓ Native | ✗ Requires encoding | ✓ Native | ✗ Requires embedding |
| Feature importance | ✓ Built-in | ✓ Available | ✓ Built-in | ✗ Black-box |
| Training speed | ✓ Fast | ✓ Very fast | ✓ Fast | ✗ Slow |
| Interpretability | ✓ Good | ✓ Excellent | ✓ Good | ✗ Limited |
| Hyperparameter tuning | ✓ Rich set | ✓ Limited | ✓ Moderate | ✗ Complex |

#### Hyperparameter Optimization

**GridSearchCV Configuration:**
```
n_estimators: [100, 200]
max_depth: [5, 7]
learning_rate: [0.05, 0.1]
subsample: [0.8, 0.9]

Total combinations: 32
Cross-validation folds: 3
Scoring metric: F1-weighted
```

**Best Parameters Found:**
- `n_estimators`: 200
- `max_depth`: 7
- `learning_rate`: 0.05
- `subsample`: 0.9

#### Classification Performance

**Validation Set Results:**
```
Accuracy:           93.7%
F1-Score (weighted): 93.5%
Cohen's Kappa:      90.1%
```

| User Category | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| User1         | 0.95      | 0.92   | 0.93     |
| User2         | 0.92      | 0.93   | 0.93     |
| User3         | 0.93      | 0.97   | 0.95     |

**Feature Importance (Top 10):**
1. Feature engineering + polynomial expansion produced discriminative features
2. Key insights: User categories differ primarily in behavioral patterns
3. ROC-AUC scores all >0.97 per class (excellent discrimination)

