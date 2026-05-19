ARG GDAL_IMAGE=ghcr.io/osgeo/gdal:ubuntu-small-latest
FROM ${GDAL_IMAGE}
LABEL org.opencontainers.image.created="2026-05-19"
LABEL org.opencontainers.image.description="mktehokkuus - reproducible GIS Streamlit app"
RUN apt-get update && \
    apt-get install -y python3-pip python3-venv software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get -y install git
WORKDIR /app
COPY ./app /app
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /app/.venv/bin/pip install --no-cache-dir -r /app/requirements.txt
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["/app/.venv/bin/streamlit", "run", "app.py"]