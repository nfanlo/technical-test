
FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/
COPY predict.py /app/
COPY ../models /app/models

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "predict.py", "test_image.jpg"]

