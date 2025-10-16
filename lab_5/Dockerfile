# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1

# Use bash with strict flags for RUN steps
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java 17 (headless) and procps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Java 17 and prepend to PATH
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# (Optional) Verify Java is available at build time
RUN java -version

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks (optional)
VOLUME ["/app"]
