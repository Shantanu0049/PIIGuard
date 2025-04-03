# Install the application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg libavcodec-extra
RUN pip install pydub
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound-dev \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Copy the application code into the container
COPY main.py .

# Set the entrypoint for the container
CMD ["functions-framework", "--target", "pii_audio_main", "--port", "8080"]

# Expose port 8080
EXPOSE 8080
