FROM python:3.9-slim

# Install Java (required for PySpark)
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /app

# Copy requirements and install
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code
COPY src/api /app/src/api

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
