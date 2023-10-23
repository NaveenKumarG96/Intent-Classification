from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch.nn as nn
from training.custom_classifier import SequenceClassifier
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import sys,os

full_dir = os.path.join(os.path.dirname(__file__), 'training')
sys.path.append(full_dir)

app = FastAPI() 

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

classifier = torch.load('./data/custom.pth').eval()


class QueryRequest(BaseModel):
    query: str

def classify_text(text):

    inputs = tokenizer(text, 
        padding=True,
        truncation=True,
        return_tensors="pt",)

    #BERT embeddings
    with torch.no_grad():
        bert_outputs = bert_model(**inputs)
        bert_embeddings = bert_outputs.last_hidden_state

    # Passing BERT embeddings through the custom classifier
    outputs = classifier(bert_embeddings)

    
    _, predicted_class = outputs.max(1)
    print(outputs)

    intent_labels = {0: "Churn", 1: "Escalation", 2: "Churn and Escalation" , 3:'No Intent Found'}
    predicted_label = intent_labels[predicted_class.item()]

    return predicted_label



@app.get("/")
async def get_index():
    return FileResponse("form.html")

# Route to do intent classifier post call
@app.post("/intent")
async def classify(query_data: QueryRequest):
    user_query = query_data.query
    predicted_intent = classify_text(user_query) 

    return {"intent": predicted_intent}

