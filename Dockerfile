FROM python:3.11-slim

# Install system libraries required by MediaPipe and OpenCV
# libgles2 provides libGLESv2.so.2, libgl1 provides libGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/render/project/src

# Copy and install Python dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the full project
COPY . .

# Set working directory to backend for uvicorn
WORKDIR /opt/render/project/src/backend

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
