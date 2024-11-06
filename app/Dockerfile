# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Copy model files into the container (copy only the specific files)
COPY ./scaler_fixmodel.pkl /app/scaler_fixmodel.pkl
COPY ./label_encoder_fixmodel.pkl /app/label_encoder_fixmodel.pkl
COPY ./SStacked_model.pkl /app/SStacked_model.pkl

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
