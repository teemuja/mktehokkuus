mktehokkuus
===============

Reproducible Docker build and run instructions for the Streamlit GIS app.

Build (example):

```bash
# pin the GDAL image at build time if you want a specific one
docker build --build-arg GDAL_IMAGE=ghcr.io/osgeo/gdal:ubuntu-small-latest -t mktehokkuus:latest -f Dockerfile .
```

Run:

```bash
docker run --rm -p 8501:8501 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  -e MAPBOX_TOKEN="your_mapbox_token" \
  -e MAPBOX_STYLE="your_mapbox_style" \
  mktehokkuus:latest
```

Local dev (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r app/requirements.txt
cd app
streamlit run app.py
```

Notes
- `app/requirements.txt` now pins package versions for reproducible installs.
- Dockerfile exposes an ARG `GDAL_IMAGE` so future builds can pin the base GDAL image.
- Keep `MAPBOX_TOKEN` and `MAPBOX_STYLE` in `~/.streamlit/secrets.toml` or pass as env vars.
