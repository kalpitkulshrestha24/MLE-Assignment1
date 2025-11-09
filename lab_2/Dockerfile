# last updated Oct 27 2025, 6:55pm SGT
FROM python:3.12-slim

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# Install Java 21 (trixie ships 21) and procps
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jdk-headless procps && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME explicitly (Java 21 path on Debian)
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Use bash for subsequent RUN/CMD (safer than replacing /bin/sh symlink)
SHELL ["/bin/bash", "-lc"]

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
# Ensure your requirements.txt includes jupyterlab; add/adjust as needed.
RUN pip install --no-cache-dir -r requirements.txt
# If you need to pin PySpark, verify compatibility with Java 21 (Spark 3.5.x is usually Java 17)
# RUN pip install --no-cache-dir pyspark==3.5.1

# Expose the default JupyterLab port
EXPOSE 8888

# Create a volume mount point for notebooks
VOLUME /app

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes

# Set up the command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
