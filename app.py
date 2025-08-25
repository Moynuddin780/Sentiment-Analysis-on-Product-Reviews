from fastapi import FastAPI
from pydantic import BaseModel
from utils import predict_sentiment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentiment Analysis API")

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class Review(BaseModel):
    text: str
    method: str = "bert"  # default

# Health check
@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

# Prediction endpoint
@app.post("/predict/")
def predict(review: Review):
    label = predict_sentiment(review.text, review.method)
    sentiment = "Positive" if label==1 else "Negative"
    return {"text": review.text, "method": review.method, "prediction": sentiment}
