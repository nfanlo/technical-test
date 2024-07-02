FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/
COPY predict.py /app/
COPY ../models /app/models
COPY ../data/test /app/data/test

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "predict.py"]
