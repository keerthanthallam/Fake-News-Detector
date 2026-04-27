from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Fake News Detection API")

# Setup CORS to allow the frontend to communicate with we
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load machine learning artifacts at app startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')

print("Loading ML model...")
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("ML model loaded successfully.")
except FileNotFoundError:
    print("Warning: Model artifacts not found. Please run train.py first.")
    model = None
    vectorizer = None

class NewsRequest(BaseModel):
    text: str

class NewsResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=NewsResponse)
def predict_news(req: NewsRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model is not trained/loaded yet.")
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")

    # Vectorize input text
    X_input = vectorizer.transform([req.text])
    
    # Predict
    prediction_raw = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    
    classification = "Fake News" if prediction_raw == 1 else "True News"
    confidence = max(probabilities) * 100
    
    return NewsResponse(prediction=classification, confidence=confidence)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fake News Detection API limits are operational."}
