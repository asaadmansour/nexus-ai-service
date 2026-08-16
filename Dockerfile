FROM python:3.12-alpine
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apk add --no-cache bash build-base git nodejs npm
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade \
      "pip>=26.1.2" "setuptools>=78.1.1" \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN adduser -D -u 1000 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
