FROM python:3.13

COPY ./requirements.txt /web_app/requirements.txt

WORKDIR /web_app

RUN pip install -r requirements.txt

COPY web_app/* /web_app

COPY web_app/roberta-sequence-classification-9.onnx /web_app


ENTRYPOINT [ "uvicorn" ]

CMD [ "--host", "0.0.0.0", "main:app"]

