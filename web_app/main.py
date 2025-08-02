from fastapi import FastAPI
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime
from pydantic import BaseModel
tokenizer = tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")

app = FastAPI()
class Body(BaseModel):
    phrase: str

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

@app.get("/")
def root():
    """
    curl localhost:8000
    """
    return {"message": "Welcome to the RoBERTa Sequence Classification API"}

@app.post("/predict")
def predict(body: Body):
    """
    curl -X POST --data '{"phrase": "This is a test"}' localhost:8000/predict
    """
    input_ids = torch.tensor(tokenizer.encode(body.phrase, add_special_tokens=True)).unsqueeze(0)
    inputs = {session.get_inputs()[0].name: to_numpy(input_ids)}
    out = session.run(None, inputs)
    predict = np.argmax(out)

    return {'positive': bool(predict)}