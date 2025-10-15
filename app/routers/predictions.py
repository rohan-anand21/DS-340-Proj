import pandas as pd
import io
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse
from ..dependencies import get_models, templates

router = APIRouter()

@router.post("/predict", response_class=HTMLResponse)
async def predict_single_review(request: Request, review_text: str = Form(...), models: dict = Depends(get_models)):
    preprocessed_lstm_input = models["preprocess_lstm"](review_text)
    
    pred_simple = models["simple_lstm"].predict(preprocessed_lstm_input)
    pred_stacked = models["stacked_lstm"].predict(preprocessed_lstm_input)
    pred_bert = models["bert"].predict(review_text)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "review_text": review_text,
        "prediction_simple_lstm": pred_simple,
        "prediction_stacked_lstm": pred_stacked,
        "prediction_bert": pred_bert
    })
    
@router.post("/bulk-predict", response_class=HTMLResponse)
async def predict_bulk_reviews(request: Request, file: UploadFile = File(...), models: dict = Depends(get_models)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    results = []
    correct_counts = {"simple_lstm": 0, "stacked_lstm": 0, "bert": 0}

    for _, row in df.iterrows():
        text = str(row['reviews.text'])
        true_label = "Positive" if row['reviews.rating'] >= 4 else "Negative"
        
        lstm_input = models["preprocess_lstm"](text)
        pred_s = models["simple_lstm"].predict(lstm_input)
        pred_st = models["stacked_lstm"].predict(lstm_input)
        pred_b = models["bert"].predict(text)
        
        if pred_s == true_label: correct_counts["simple_lstm"] += 1
        if pred_st == true_label: correct_counts["stacked_lstm"] += 1
        if pred_b == true_label: correct_counts["bert"] += 1

        results.append({
            "review": text, "true_label": true_label,
            "pred_simple": pred_s, "pred_stacked": pred_st, "pred_bert": pred_b
        })

    total = len(df)
    accuracies = {model: f"{(count / total * 100):.2f}%" for model, count in correct_counts.items()}
    
    return templates.TemplateResponse("bulk_results.html", {
        "request": request, "results": results,
        "accuracies": accuracies, "filename": file.filename
    })
