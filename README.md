# Kindle Product Sentiment Analysis

A sentiment classification system trained on 10,000+ Amazon Kindle reviews using NLP and machine learning models to classify customer feedback into positive or negative.

## Problem Statement

Given a large volume of customer reviews, can we accurately predict sentiment
(positive/negative) despite noisy text and significant class imbalance?

## Approach

**Preprocessing pipeline**
- Lowercasing, punctuation removal, and noise reduction
- Stopword removal using NLTK
- Lemmatization to normalize word forms

**Feature engineering**
- TF-IDF vectorisation to convert cleaned text into numerical features
- Class imbalance handling via feature-level engineering

**Model**
- Logistic Regression
- Hyperparameter tuning using GridSearchCV
- Evaluation using accuracy and confusion matrix
---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 90.85% |
| Evaluation | Precision, Recall, F1 (multi-metric) |
| Dataset size | 10,000+ reviews |

## Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- TF-IDF (sklearn's TfidfVectorizer)

## Setup & Usage
```bash
git clone https://github.com/Prriiiyankaaa/Kindle-Sentiment-Analysis
cd Kindle-Sentiment-Analysis
pip install -r requirements.txt
jupyter notebook
```

## Key Takeaways

- Data quality and preprocessing matter as much as model choice
- Accuracy alone is misleading under skewed class distributions —
  always evaluate with F1-score on imbalanced datasets
