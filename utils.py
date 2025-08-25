import re
import string
import joblib
import numpy as np
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch

# -------------------------------
# Load models
# -------------------------------
tfidf_model = joblib.load("models/tfidf_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

w2v_model = Word2Vec.load("models/w2v_model.pkl")
w2v_lr = joblib.load("models/w2v_model.pkl")  # if your LR is saved separately, load that instead

bert_model_lr = joblib.load("models/bert_model.pkl")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Word2Vec embedding
# -------------------------------
def get_w2v_embedding(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# -------------------------------
# BERT embedding
# -------------------------------
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# -------------------------------
# Prediction function
# -------------------------------
def predict_sentiment(text, method="bert"):
    text = preprocess_text(text)

    if method=="tfidf":
        X = tfidf_vectorizer.transform([text])
        return int(tfidf_model.predict(X)[0])

    elif method=="w2v":
        tokens = text.split()
        X = get_w2v_embedding(tokens, w2v_model).reshape(1,-1)
        return int(w2v_lr.predict(X)[0])

    elif method=="bert":
        X = get_bert_embedding(text)
        return int(bert_model_lr.predict(X)[0])
