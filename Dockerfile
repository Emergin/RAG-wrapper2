# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && \
    pip config set global.timeout 1000

# Create and work in a temporary directory
WORKDIR /install

# Copy and install requirements in smaller batches
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install streamlit==1.30.0 python-dotenv==1.0.0
RUN pip install --no-cache-dir --prefix=/install openai==1.5.0 PyPDF2==3.0.1
RUN pip install --no-cache-dir --prefix=/install pandas==2.1.3 markdown==3.5.1
RUN pip install --no-cache-dir --prefix=/install python-docx==1.0.1 plotly==5.17.0
RUN pip install --no-cache-dir --prefix=/install scikit-learn==1.3.2
RUN pip install --no-cache-dir --prefix=/install faiss-cpu==1.7.4
RUN pip install --no-cache-dir --prefix=/install --index-url https://pypi.org/simple/ pyarrow==20.0.0

# Stage 2: Runtime image
FROM python:3.9-slim

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Update and install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /RAG-wrapper

# Copy the application
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]