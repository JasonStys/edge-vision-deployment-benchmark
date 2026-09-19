# File: Builds a non-root CPU validation image from the locked Python environment and artifacts.
FROM python:3.14.7-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-runtime.lock pyproject.toml README.md LICENSE ./
COPY src ./src
COPY artifacts ./artifacts
COPY docs/reports/generated ./docs/reports/generated

RUN python -m pip install --upgrade pip==26.2.1 \
    && python -m pip install -r requirements-runtime.lock \
    && python -m pip install --no-build-isolation --no-deps . \
    && useradd --create-home --uid 10001 benchmark \
    && chown -R benchmark:benchmark /app

USER benchmark
ENTRYPOINT ["edge-vision-benchmark"]
CMD ["validate", "--root", "/app"]
