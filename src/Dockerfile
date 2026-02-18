FROM python:3.10-slim

WORKDIR /app
#open-cv libaries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000


CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]