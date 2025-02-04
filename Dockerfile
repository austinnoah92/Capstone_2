# Dockerfile

FROM python:3.13-slim

RUN pip install pipenv gunicorn

WORKDIR /app

COPY . .

RUN pipenv install --system --deploy

COPY 'predict.py' 'best_model.pkl' ./

EXPOSE 9990

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9990", "predict:app" ]
