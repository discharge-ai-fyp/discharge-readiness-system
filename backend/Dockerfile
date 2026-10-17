FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser \
    && mkdir /runtime \
    && chown appuser:appuser /runtime
WORKDIR /srv

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend/app /srv/backend/app
COPY criteria /srv/criteria
COPY ml/__init__.py /srv/ml/__init__.py
COPY ml/src /srv/ml/src

ENV PYTHONPATH=/srv/backend:/srv \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health')"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
