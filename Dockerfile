FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Set environment
ENV FLASK_APP=app.py

# Expose port (Koyeb uses 8000 by default)
EXPOSE 8000

# Run with gunicorn - use PORT env var that Koyeb sets
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "120", "app:app"]
