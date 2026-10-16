# syntax=docker/dockerfile:1
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir build==1.6.1 && python -m build --wheel --outdir /dist

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
LABEL org.opencontainers.image.title="VeraMem" \
      org.opencontainers.image.description="Verifiable, outcome-aware, share-safe memory for AI agents" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/Thebinary110/veramem"
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[mcp]" && rm /tmp/*.whl \
 && useradd --create-home --uid 10001 veramem && mkdir /data && chown veramem /data
USER veramem
ENV VERAMEM_HOME=/data PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys;sys.exit(urllib.request.urlopen('http://127.0.0.1:8765/health').status!=200)"
# Listening on 0.0.0.0 inside the container requires VERAMEM_API_KEY (enforced by the CLI).
CMD ["veramem", "serve", "--host", "0.0.0.0", "--port", "8765"]
