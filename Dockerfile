FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Set environment
ENV FLASK_APP=app.py
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run with gunicorn on port 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
