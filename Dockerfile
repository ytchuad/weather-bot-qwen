FROM node:22-slim AS frontend-build

WORKDIR /src/app/frontend
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    APP_MODE=streamlit
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
COPY --from=frontend-build --chown=user /src/app/frontend/dist ./app/frontend/dist

EXPOSE 7860

CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"api\" ]; then uvicorn app.api.server:app --host 0.0.0.0 --port 7860; else streamlit run streamlit_app.py --server.port=7860 --server.address=0.0.0.0; fi"]
