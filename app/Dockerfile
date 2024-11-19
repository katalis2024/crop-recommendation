# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV OUTPUT_DIR=/app

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy model files into the container (copy only the specific files)
COPY pscaler_fixmodel.pkl /app/
COPY plabel_encoder_fixmodel.pkl /app/
COPY PStacked_model.pkl /app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
