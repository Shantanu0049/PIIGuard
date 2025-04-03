# README: PII Audio Processing with Google Cloud Functions

## Overview
This project implements a Google Cloud Function to process audio files uploaded to a Google Cloud Storage (GCS) bucket. The function performs the following operations:

1. Extracts audio from the uploaded file (supports both audio and video files).
2. Transcribes the audio using Google Cloud's Speech-to-Text API.
3. Identifies Personally Identifiable Information (PII) using Gemini AI.
4. Applies a beep sound over detected PII in the audio.
5. Saves the modified audio file back to a GCS bucket.

## Prerequisites
To use this project, you need:
- A Google Cloud Platform (GCP) project with billing enabled.
- Google Cloud Storage (GCS) bucket for input and output.
- Enabled APIs:
  - Google Cloud Speech-to-Text API
  - Google Cloud Storage API
  - Google Gemini API (Generative AI API)
- Python 3.x

## Project Structure
```
pii_audio_processing/
├── main.py  # Contains the Cloud Function logic
├── requirements.txt  # Lists the dependencies
└── README.md  # This file
```

## Setup and Deployment

### Step 1: Clone the Repository
```sh
git clone <repository_url>
cd pii_audio_processing
```

### Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Configure Google Cloud
- Set up authentication:
```sh
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-service-account.json"
```
- Update the `PROJECT_ID` and `OUTPUT_BUCKET_NAME` in `main.py`.

### Step 4: Deploy the Cloud Function
```sh
gcloud functions deploy pii_audio_main     --runtime python310     --trigger-event google.storage.object.finalize     --trigger-resource <your-input-bucket-name>     --entry-point pii_audio_main     --timeout 300s
```

## How It Works
1. When a file is uploaded to the input GCS bucket, the function is triggered.
2. It extracts audio (if the file is a video) and transcribes the speech.
3. The transcript is analyzed for sensitive information.
4. A beep sound is applied over the detected PII.
5. The processed file is saved in the output GCS bucket.

## API Usage Details
- **Google Cloud Speech-to-Text**: Converts speech into text with timestamps.
- **Google Gemini AI**: Detects sensitive information such as names, phone numbers, and addresses.
- **Pydub**: Processes audio and applies beeps.

## Example Output
Beeped audio files are saved in the `pii_protectors_output` bucket with the prefix `final_beeped_`.

## Logging and Debugging
Logs can be viewed using:
```sh
gcloud functions logs read pii_audio_main
```
