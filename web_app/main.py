from fastapi import FastAPI
import torch
import numpy as np
from transformers import RobertaTokenizer
from transformers import pipeline
import onnxruntime
from pydantic import BaseModel
from huggingface_hub import login
import os

# tokenizer = tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")


login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
# pipe = pipeline(
#     "image-text-to-text",
#     model="google/gemma-3n-e2b-it",
#     device="cuda",
#     torch_dtype=torch.bfloat16,
# )
model_name_or_path = "Qwen/Qwen3-8B"

generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)



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

# @app.post("/predict")
# def predict(body: Body):
#     """
#     curl -X POST --data '{"phrase": "This is a test"}' localhost:8000/predict
#     """
#     input_ids = torch.tensor(tokenizer.encode(body.phrase, add_special_tokens=True)).unsqueeze(0)
#     inputs = {session.get_inputs()[0].name: to_numpy(input_ids)}
#     out = session.run(None, inputs)
#     predict = np.argmax(out)

#     return {'positive': bool(predict)}

@app.post("/generate")
def generate(body: Body):
    """
    curl -X POST --data '{"phrase": "This is a test"}' localhost:8000/generate
    """

    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": body.phrase}
        ]
    }
    ]
    output = generator(text=messages, max_new_tokens=200)
    return {"generated_text": output[0]["generated_text"][-1]["content"]}