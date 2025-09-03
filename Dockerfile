# Use the official Python image as the base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy and install the dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your application code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 7860

# Command to run your FastAPI application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
