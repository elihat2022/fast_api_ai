# FastAPI LLM Service

A lightweight FastAPI application that provides REST endpoints for text generation using Qwen2.5-0.5B and sentiment classification using RoBERTa models.

## 🚀 Features

- **Text Generation**: Generate human-like text responses using Qwen2.5-0.5B

- **Docker Support**: Containerized deployment with Docker
- **Auto-generated Documentation**:
  Interactive API docs with Swagger UI
- **Optimized Performance**: Models loaded once at startup for faster inference

## 📋 Requirements

- Python 3.13+
- Docker (optional, for containerized deployment)
- Hugging Face Hub token (for Qwen model access)

### Getting a Hugging Face Token

1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with read permissions
3. Copy the token (starts with `hf_`)

## 📖 API Documentation

Once the application is running, visit:

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
