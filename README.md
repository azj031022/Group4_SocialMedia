# Restaurant Review Sentiment Analysis

## Overview

This notebook performs sentiment analysis on restaurant review data using both rule-based and machine learning models. The workflow includes text preprocessing, sentiment labeling using VADER, feature extraction with TF-IDF, traditional machine learning classifiers, and a fine-tuned BERT model for advanced classification.

## Dataset

- File: `res_review.csv`
- Contains customer restaurant reviews with text and numeric star ratings (1-5).

## Workflow

### 1. Text Preprocessing

- Lowercasing all text.
- Removing URLs, special characters, and extra whitespace.
- Removing stopwords using NLTK.
- Lemmatization using WordNet Lemmatizer.

### 2. Sentiment Labeling with VADER

- Used VADER SentimentIntensityAnalyzer to assign sentiment scores.
- Reviews labeled as Positive, Neutral, or Negative based on compound scores.

### 3. Feature Extraction

- Applied TF-IDF vectorization (unigram and bigram) to transform text into numerical features.

### 4. Machine Learning Models

Trained and evaluated several classifiers using scikit-learn:

- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes (MultinomialNB)
- Random Forest

### 5. Transformer Model (BERT)

- Fine-tuned pre-trained `bert-base-uncased` model from Hugging Face.
- Converted dataset to Hugging Face `datasets` format for tokenization and training.
- Used Trainer API for model fine-tuning and evaluation.

### 6. Evaluation

- Calculated Accuracy, Precision, Recall, and F1-score.
- Visualized confusion matrices for both traditional ML models and BERT.

### 7. Visualization

- Generated bar plot for sentiment distribution (VADER results).
- Created WordCloud for positive reviews.
- Displayed confusion matrix heatmaps.

## Libraries Used

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `transformers (Hugging Face)`
- `datasets`
- `vaderSentiment`

## Notes

- No aspect-level sentiment extraction was performed.
- The workflow focuses on document-level sentiment classification.
- BERT model achieved higher accuracy than traditional models.

## Future Improvements

- Implement full aspect-based sentiment extraction using spaCy or transformer-based ABSA models.
- Extend dataset for better generalization.
- Deploy model into real-time inference or dashboards using Streamlit.

