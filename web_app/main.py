from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from huggingface_hub import login
import os

# Login to Hugging Face Hub
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Define model name
model_name_or_path = "Qwen/Qwen2.5-0.5B"

# Initialize the text generation pipeline
# https://qwen.readthedocs.io/en/latest/inference/transformers.html
generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)

# Initialize FastAPI app
app = FastAPI()

# Simple data model for request body
class Body(BaseModel):
    phrase: str

# Root endpoint
@app.get("/")
def root():
    """
    curl localhost:8000
    """
    return {"message": "Welcome to the Qwen API"}


@app.post("/generate")
def generate(body: Body):
    """
    curl -X POST --data '{"phrase": "This is a test"}' localhost:8000/generate
    Generates text based on the input phrase.
    """
    # Prepare messages for the model
    messages = [
        {"role": "user", "content": body.phrase},
    ]

    # Generate text using the model    
    output = generator(
        messages, 
        max_new_tokens=512, 
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )[0]["generated_text"]

    # Return the generated text
    return {"generated_text": output[-1]["content"]}