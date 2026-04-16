FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV PYTHONPATH=/app

EXPOSE 8520

CMD ["python", "server.py"]
