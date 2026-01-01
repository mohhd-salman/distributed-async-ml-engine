FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tasks.py .

CMD ["celery", "-A", "tasks.celery_app", "worker", "--loglevel=INFO", "--concurrency=2"]
