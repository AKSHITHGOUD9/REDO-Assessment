# Use official Python image
FROM python:3.13-slim

# IMPORTANT: When building, use '.' as the build context, e.g.:
#   docker build -t redo-assessment .

# Set working directory
WORKDIR /app

# Copy requirements (create if not present)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY app/ ./

# Expose Streamlit port
EXPOSE 8502

# Set Streamlit to run the app
CMD ["streamlit", "run", "main.py", "--server.port=8502", "--server.address=0.0.0.0"]