# Kindle Product Sentiment Analysis

A sentiment classification system trained on 12,000+ Amazon Kindle product reviews,
using a Random Forest classifier with TF-IDF text features.

## Problem Statement

Given a large volume of customer reviews, can we accurately predict sentiment
(positive/negative) despite noisy text and significant class imbalance?

## Approach

**Preprocessing pipeline**
- Lowercasing, punctuation removal, stopword filtering, tokenisation
- Categorical encoding for non-text features

**Feature engineering**
- TF-IDF vectorisation to convert cleaned text into numerical features
- Class imbalance handling via feature-level engineering

**Model**
- Random Forest classifier
- Evaluated on precision, recall, and F1-score — not just accuracy — to
  account for skewed class distribution

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 88% |
| Evaluation | Precision, Recall, F1 (multi-metric) |
| Dataset size | 12,000+ reviews |

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
- A well-tuned simple model (Random Forest + TF-IDF) can be
  production-ready without deep learning overhead
