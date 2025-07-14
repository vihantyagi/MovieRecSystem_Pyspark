FROM python:3.9

# Install Java
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Install Python packages including PySpark
# Install dependencies
COPY requirements-spark.txt .
RUN pip install --no-cache-dir -r requirements-spark.txt

WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
