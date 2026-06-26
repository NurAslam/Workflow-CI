FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.18.0 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    matplotlib==3.10.0 \
    seaborn==0.13.2 \
    joblib==1.4.2

COPY modelling.py .
COPY heart_disease_preprocessing.csv .

RUN mkdir -p artifacts

CMD ["python", "modelling.py"]