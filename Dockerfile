FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade \
      "pip>=26.1.2" "setuptools>=78.1.1" \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m pip uninstall -y setuptools pip
COPY . .
RUN useradd -m appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
