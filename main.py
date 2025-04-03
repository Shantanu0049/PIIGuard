import functions_framework
import os
import io
import logging
from google.cloud import speech, storage
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.utils import mediainfo
import google.generativeai as genai
import tempfile

PROJECT_ID = 'geometric-wall-455607-h8'  # Your Google Cloud Project ID
OUTPUT_BUCKET_NAME = "pii_protectors_output"  # Output bucket where beeped audio will be stored
API_KEY = "AIzaSyC95x_CV0nSeIJen2KnrPOU5hC6P6NHJMo"  # Replace with your actual API key
genai.configure(api_key=API_KEY)  # Configure Gemini API with the key

# Initialize Google Cloud clients for speech and storage
storage_client = storage.Client(project=PROJECT_ID)
speech_client = speech.SpeechClient()


def transcribe_audio(gcs_uri, input_bucket_name, file_name):
    """
    Transcribes speech from an audio file stored in Cloud Storage using Google's Speech-to-Text API.

    Args:
        gcs_uri (str): The URI of the audio file in Google Cloud Storage.
        input_bucket_name (str): The name of the input bucket.
        file_name (str): The name of the audio file to transcribe.

    Returns:
        tuple: A tuple containing:
            - transcript_data (list): A list of dictionaries with word timestamps and word info.
            - full_transcript (str): The full transcribed text as a single string.
    """
    # Download the audio file from GCS bucket
    audio_blob = storage_client.bucket(input_bucket_name).blob(file_name)
    audio_bytes = audio_blob.download_as_bytes()

    # Save the audio bytes to a temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name

    # Get audio file information (e.g., number of channels) using mediainfo
    audio_info = mediainfo(temp_audio_file_path)
    num_channels = int(audio_info['channels'])

    # Configure speech-to-text request
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=24000,  # Adjust based on your audio file
        language_code="en-CA",
        use_enhanced=True,
        model="telephony",
        audio_channel_count=num_channels,
        enable_word_confidence=True,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True
    )

    # Send the request to Google's speech-to-text service
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=300)  # 5 minutes timeout

    # Process the transcribed results into timestamped data and full transcript
    transcript_data = []
    full_transcript = []

    for result in response.results:
        transcript = result.alternatives[0].transcript
        full_transcript.append(transcript)

        for word_info in result.alternatives[0].words:
            transcript_data.append({
                "word": word_info.word,
                "start_time": word_info.start_time.total_seconds(),
                "end_time": word_info.end_time.total_seconds()
            })

    logging.info(f"Transcription complete. Found {len(transcript_data)} words.")
    return transcript_data, " ".join(full_transcript)


def process_sensitive_info(transcript_data):
    """
    Analyzes the transcribed text to extract sensitive information using the Gemini API.

    Args:
        transcript_data (list): A list of dictionaries containing word timestamps and word info.

    Returns:
        list: A list of dictionaries containing detected sensitive information along with their timestamps.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Configuration for the generative model
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 8192
    }

    # Format the transcript data into a string
    formatted_transcript = "\n".join(
        [f"[{item['start_time']:.2f} - {item['end_time']:.2f}] {item['word']}" for item in transcript_data]
    )

    # Prepare the prompt for Gemini model to extract sensitive information
    prompt = """
            Analyze the transcript and extract sensitive information, including:
            Full names
            Emails
            Phone numbers
            Social Security Numbers
            Complete Addresses (MUST include house/building number, street name, city, and postal code)**  
            Medical conditions
            Appointment times
            Car models
            Customer numbers
            Agent names
            Greeted names (if spoken)
            year of birth
            Dates of birth
            CVV
            Any numeric identifiers like 84097

            ### **Rules for Address Extraction (Critical Updates)**  
            1. **Extract the ENTIRE address in one phrase**  
              - The address **must start with the first number mentioned** (e.g., "1255 Maple Street").  
              - The address **must end with the last part**, including postal codes like "84097".  
              - **Do NOT break the address into separate parts.** Return it as one **continuous phrase**.
            2. **Names:** Extract full names exactly as spoken. If a name appears in a greeting (e.g., "Thank you, John"), extract "John" separately.  
            3. **Social Security Numbers:** Extract **only** the number, not surrounding words.  
            4. **Car Models & Numbers:** Extract the **entire model name**, even if split into parts.  
            5. **Timestamp Accuracy:** The timestamp **must cover the entire detected phrase**, starting from the first word and ending at the last word.

            ### **Expected Output Format:**
            TIMESTAMP_START - TIMESTAMP_END : "DETECTED_INFORMATION" (Type: CATEGORY)

            **Examples:**
            00:12 - 00:14 : "Julian Simon" (Type: Full Name)
            00:18 - 00:20 : "John" (Type: Greeted Name)
            00:30 - 00:32 : "1255 Maple Street, Springfield 84097" (Type: Address)
            00:45 - 00:47 : "123-45-6789" (Type: Social Security Number)
              """

    # Send the request to Gemini for PII extraction
    response = model.generate_content(f"{prompt}\n\n{formatted_transcript}", generation_config=generation_config)

    extracted_info = []
    for part in response.parts:
        lines = part.text.strip().split("\n")
        for line in lines:
            if " - " in line:
                try:
                    timestamps, pii_info = line.split(" : ")
                    start_time, end_time = map(float, timestamps.strip().split(" - "))
                    extracted_info.append({"timestamp": f"{start_time} - {end_time}", "info": pii_info.strip('"')})
                except:
                    pass

    return extracted_info


def apply_beep_gcs(input_bucket, input_audio_filename, sensitive_info, output_bucket, output_audio_filename,
                   temp_audio_file_path):
    """
    Applies a beep sound to an audio file in GCS, covering regions that contain sensitive information.

    Args:
        input_bucket (str): The name of the input GCS bucket containing the original audio file.
        input_audio_filename (str): The name of the audio file to process.
        sensitive_info (list): A list of dictionaries containing detected sensitive information and timestamps.
        output_bucket (str): The name of the output GCS bucket to store the beeped audio.
        output_audio_filename (str): The name of the output audio file to be created.
        temp_audio_file_path (str): The local path to the temporary audio file.

    Returns:
        str: The GCS URI of the final beeped audio file.
    """
    bucket = storage_client.bucket(input_bucket)
    blob = bucket.blob(input_audio_filename)

    # Load the audio file using pydub
    audio = AudioSegment.from_file(temp_audio_file_path, format="mp3")
    logging.info(f"Loaded audio file: duration = {len(audio)}ms")

    # Generate a beep sound for replacing sensitive info
    def generate_beep(duration_ms=600, frequency=500):
        """Generates a beep sound for a given duration."""
        return Sine(frequency).to_audio_segment(duration=duration_ms)

    # Process the detected sensitive information and apply beeps
    for info in sensitive_info:
        try:
            start_time, end_time = map(float, info["timestamp"].split(" - "))
            start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
            beep = generate_beep(duration_ms=max(300, end_ms - start_ms))

            logging.info(f"Beeping: {start_time}s - {end_time}s for '{info['info']}'")
            audio = audio[:start_ms] + beep + audio[end_ms:]  # Insert beep into audio

        except Exception as e:
            logging.error(f"Error processing {info['timestamp']}: {str(e)}")

    # Save the processed audio to GCS
    output_audio_data = io.BytesIO()
    audio.export(output_audio_data, format="mp3")

    # Upload the beeped audio file to the output GCS bucket
    output_blob = storage_client.bucket(output_bucket).blob(output_audio_filename)
    output_blob.upload_from_string(output_audio_data.getvalue(), content_type="audio/mp3")

    logging.info(f"Beeped audio saved to: gs://{output_bucket}/{output_audio_filename}")
    return f"gs://{output_bucket}/{output_audio_filename}"


def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file and saves it to a specified path.

    Args:
        video_path (str): The path to the video file.
        audio_path (str): The path to save the extracted audio file.
    """
    # Convert the video to an audio file
    audio = AudioSegment.from_file(video_path, format=video_path.split(".")[-1])
    audio.export(audio_path, format="mp3")


def extract_audio_acc_to_ext(blob, file_name):
    """
    Extracts audio from a file based on its extension (audio or video).

    Args:
        blob (Blob): The GCS blob representing the file.
        file_name (str): The name of the file to extract audio from.

    Returns:
        bytes: The audio content extracted from the file.
    """
    file_extension = file_name.split('.')[-1].lower()
    if file_extension in ["mp4", "avi", "mpeg4"]:
        # Extract audio from a video file
        temp_audio_path = "/tmp/temp_audio.mp3"
        video_bytes = blob.download_as_bytes()
        temp_video_path = "/tmp/temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)
        extract_audio_from_video(temp_video_path, temp_audio_path)
        with open(temp_audio_path, "rb") as f:
            return f.read()
    else:
        # Assume it's an audio file and return the raw bytes
        return blob.download_as_bytes()


@functions_framework.cloud_event
def pii_audio_main(cloud_event):
    """
    Cloud Function that processes an audio file from Google Cloud Storage to detect and mask sensitive information.

    This function is triggered by an event (such as a new file being uploaded) to a Google Cloud Storage bucket.
    It extracts audio from the uploaded file, transcribes the audio to text using Google Cloud's Speech-to-Text API,
    identifies sensitive information in the transcript using Gemini AI, and applies a beep over sensitive content in the audio file.

    Args:
        cloud_event (CloudEvent): The event object containing details about the uploaded file in GCS.

    Returns:
        None: The function performs actions like transcribing, detecting sensitive data, and saving the processed audio.
    """
    # Extract event data (e.g., file name and bucket)
    data = cloud_event.data
    file_name = data["name"]  # The name of the file uploaded to GCS
    bucket_name = data['bucket']  # The GCS bucket where the file is stored

    # Create a reference to the input GCS bucket and file URI
    bucket = storage_client.bucket(bucket_name)
    gcs_uri = f"gs://{bucket.name}/{file_name}"

    # Get the blob (object) reference for the uploaded file in the bucket
    blob = bucket.get_blob(file_name)

    # Extract audio from the uploaded file, handling both audio and video formats
    audio_bytes = extract_audio_acc_to_ext(blob, file_name)

    # Save the extracted audio bytes to a temporary file for further processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name  # Path to the temporary audio file

    # Perform speech-to-text transcription on the audio file from GCS
    transcript_data, full_transcript = transcribe_audio(gcs_uri, bucket.name, file_name)

    # Analyze the transcription to detect sensitive information
    sensitive_info = process_sensitive_info(transcript_data)

    # Apply a beep sound over sensitive information in the audio
    output_gcs_uri = apply_beep_gcs(bucket.name, file_name, sensitive_info, OUTPUT_BUCKET_NAME,
                                    f"final_beeped_{file_name}", temp_audio_file_path)

    # Log the completion message and provide the URI of the processed audio
    print("\n--- BEEP APPLICATION COMPLETE ---")
    print(f"Beeped audio saved to: {output_gcs_uri}")