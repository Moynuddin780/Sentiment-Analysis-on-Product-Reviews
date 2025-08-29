# IMDB Sentiment Analysis Project

This project performs **sentiment analysis** on the IMDB movie review dataset, classifying reviews as **positive** or **negative** using three distinct approaches: **TF-IDF**, **Word2Vec**, and **BERT**, each paired with a Logistic Regression classifier. The project includes comprehensive data preprocessing, model training, evaluation, and visualization to compare the performance of these methods.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Visualizations](#visualizations)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to analyze the sentiment expressed in IMDB movie reviews by leveraging natural language processing (NLP) techniques. Three different text representation methods are used:
- **TF-IDF**: A traditional bag-of-words approach with term frequency-inverse document frequency.
- **Word2Vec**: A word embedding model capturing semantic relationships.
- **BERT**: A transformer-based model for contextual word embeddings.

Each method generates feature representations that are fed into a Logistic Regression model for binary classification (positive vs. negative sentiment). The project evaluates model performance using **accuracy**, **precision**, **recall**, and **F1-score** and provides visualizations to interpret the results.

## Dataset
The dataset used is the **IMDB dataset** from the `datasets` library, containing 50,000 movie reviews (25,000 for training and 25,000 for testing). The training set is further split into 80% training and 20% validation sets, with labels balanced between positive (1) and negative (0) reviews.

## Features
- **Data Preprocessing**: Lowercasing, HTML tag removal, punctuation removal, and extra space normalization.
- **Model Training**: Three models (TF-IDF, Word2Vec, BERT) with Logistic Regression for classification.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score for validation and test sets.
- **Visualizations**:
  - Distribution of positive and negative reviews.
  - Word count analysis by sentiment.
  - Word clouds for all, positive, and negative reviews.
  - Confusion matrices for each model.
  - Bar plots comparing model performance across metrics.

## Installation
To run this project, ensure you have Python 3.8+ installed. Follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install datasets transformers gensim scikit-learn nltk wordcloud matplotlib seaborn torch
   ```

3. **Download NLTK resources**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

4. **Hardware Requirements**:
   - For BERT, a GPU is recommended for faster processing (CUDA support required).
   - At least 8GB RAM for Word2Vec and BERT embeddings.

## Usage
1. **Run the notebook**:
   Open the Jupyter notebook (`Module13_IMDB_Sentiment_Moynuddin.ipynb`) in Jupyter or Google Colab and execute the cells sequentially.

2. **Key Steps**:
   - Load and preprocess the IMDB dataset.
   - Train and evaluate models (TF-IDF, Word2Vec, BERT).
   - Save trained models and vectorizers (`tfidf_model.pkl`, `w2v_model.pkl`, etc.).
   - Generate visualizations for analysis.

3. **Example Command** (if running as a script):
   ```bash
   python imdb_sentiment_analysis.py
   ```

## Model Details
- **TF-IDF + Logistic Regression**:
  - Uses `TfidfVectorizer` with 5,000 max features and English stop words.
  - Logistic Regression with 200 iterations.
- **Word2Vec + Logistic Regression**:
  - Trains a Word2Vec model with 100-dimensional embeddings on tokenized training data.
  - Computes average word embeddings for each review.
- **BERT + Logistic Regression**:
  - Uses `bert-base-uncased` for contextual embeddings (mean-pooled over tokens).
  - Processes texts in batches (batch size: 16, max length: 64).

## Results
The models are evaluated on the test set with the following metrics (example values, actual results may vary):

| Method   | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| TF-IDF   | 0.85     | 0.84      | 0.86   | 0.85     |
| Word2Vec | 0.80     | 0.79      | 0.81   | 0.80     |
| BERT     | 0.90     | 0.89      | 0.91   | 0.90     |

BERT typically outperforms TF-IDF and Word2Vec due to its contextual understanding, while TF-IDF is computationally efficient, and Word2Vec captures semantic relationships.

## Visualizations
The project includes the following visualizations:
- **Bar Plots**: Positive vs. negative review counts and percentages.
- **Histograms**: Word count distribution by sentiment.
- **Word Clouds**: Visual representation of frequent words in all, positive, and negative reviews.
- **Confusion Matrices**: Model performance for TF-IDF, Word2Vec, and BERT.
- **Comparison Bar Plot**: Accuracy and other metrics across models.

To generate visualizations, ensure `matplotlib`, `seaborn`, and `wordcloud` are installed.

## File Structure
```
imdb-sentiment-analysis/
├── Module13_IMDB_Sentiment_Moynuddin.ipynb  # Main notebook
├── tfidf_model.pkl                         # Saved TF-IDF model
├── tfidf_vectorizer.pkl                    # Saved TF-IDF vectorizer
├── w2v_model.pkl                           # Saved Word2Vec model
├── w2v_embeddings.pkl                      # Saved Word2Vec embeddings
├── bert_model.pkl                          # Saved BERT model
├── README.md                               # This file
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

