# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . /app

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 8501 (default Streamlit)
EXPOSE 8501

# Set environment variable (optional)
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_TELEMETRY_ENABLED=false

# Default command to run Streamlit app
CMD ["streamlit", "run", "app_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
