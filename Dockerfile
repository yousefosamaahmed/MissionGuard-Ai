FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --upgrade -r requirements.txt \
    && python -c "from typing_extensions import TypeVar; import psycopg"

COPY . .

RUN test -f models/opssat_model.joblib || python scripts/train_opssat.py \
    && chmod +x docker/entrypoint.sh \
    && mkdir -p data/opssat/uploads \
    && useradd --create-home --uid 10001 missionguard \
    && chown -R missionguard:missionguard /app

USER missionguard

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["./docker/entrypoint.sh"]
