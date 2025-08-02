FROM python:3.13

ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /web_app/requirements.txt

WORKDIR /web_app

RUN pip install --no-cache-dir -r requirements.txt

COPY web_app/* /web_app


ENTRYPOINT [ "uvicorn" ]

CMD [ "--host", "0.0.0.0", "main:app", "--port", "8000"]

