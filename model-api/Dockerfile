FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

ENV MODEL_PATH="/home/jovyan/work/modelos/best_pipeline.joblib"
ENV API_PORT=${API_PORT}

EXPOSE ${API_PORT}

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${API_PORT}"]