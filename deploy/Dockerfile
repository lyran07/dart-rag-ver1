# API + 웹 화면 (서빙용, CPU). GPU 임베딩은 별도 머신에서 embed 단계까지 수행.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-optional.txt
COPY . .
ENV DART_DATA_ROOT=/data
EXPOSE 8080
CMD ["uvicorn", "dart_rag.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
