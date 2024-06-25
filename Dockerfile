# Use a base image that includes Anaconda
FROM continuumio/miniconda3:latest

# Set environment variables for Conda
ENV PATH /opt/conda/envs/your_environment_name/bin:$PATH

# Create a directory for your application
WORKDIR /app

# Copy the contents of the current directory to the container
COPY . .

# Set the default command to run your application
CMD ["python", "app.py"]
