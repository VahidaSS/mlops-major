FROM python:3.9-slim

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=artifacts/savedmodel.pth

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]