FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt && useradd -m app
COPY backend backend
COPY frontend frontend
COPY models models
COPY app/feature_stats.json app/feature_stats.json
COPY results/final_metrics.json results/scenario_presets.json results/
RUN mkdir -p data && chown app data
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
