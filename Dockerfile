FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install CPU-only PyTorch
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    torch==2.14.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]