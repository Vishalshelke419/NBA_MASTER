# Use the official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all Python files and requirements.txt
COPY ./*.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Run your script (adjust if needed)
CMD ["python", "dataset.py"]
