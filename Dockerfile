FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything to the container
COPY . .

# Set the command to run your script from the subdirectory
CMD ["python", "Datasets/Python files/test.py"]
