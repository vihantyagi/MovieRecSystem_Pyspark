# docker/dashboard/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy source code
COPY src/dashboard /app/src/dashboard

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.baseUrlPath=/", \
     "--browser.gatherUsageStats=false"]
