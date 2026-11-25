# One image for every service of the system; docker-compose.yml chooses the command.

# stage 1: build the React dashboard
FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# stage 2: Python models, services and API (the API also serves the built dashboard)
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.lock .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.lock

COPY wq_model ./wq_model
COPY wq_system ./wq_system
COPY models ./models
COPY assets ./assets
COPY *.py ./
COPY --from=frontend /frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1 DESKTOP_NOTIFY=0
EXPOSE 8000
CMD ["python", "ingest_service.py"]
