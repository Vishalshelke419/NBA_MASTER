# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only Python code
COPY ["Datasets/Python files", "/app/code"]

# Create mount point for data
RUN mkdir -p /app/data_files

# Run the script
CMD ["python", "/app/code/test.py"]