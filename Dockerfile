# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD *.py /app/
ADD models/ /app/models/
RUN ls -la /app/models/*
ADD requirements_deploy.txt /app/

# Install any needed packages specified in requirements_deploy.txt
RUN pip install --no-cache-dir --compile  -r requirements_deploy.txt

# Run predict.py when the container launches
ENTRYPOINT  ["python", "/app/predict.py"]
