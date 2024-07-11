import os
import shutil
import warnings
from pathlib import Path
import whisper
from openai import OpenAI, OpenAIError
import cv2
import pytesseract
from dotenv import load_dotenv
import yt_dlp
import logging
from tqdm import tqdm
import json
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import requests
import subprocess
import time

# Define SAVED_RESULTS_DIR
SAVED_RESULTS_DIR = Path(__file__).resolve().parent / 'saved_results'
SAVED_RESULTS_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file if it exists
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

if not client.api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

WHISPER_MODEL = 'base'
OUTPUT_VIDEO = Path(__file__).resolve().parent.joinpath('data', 'video.mp4')
OUTPUT_AUDIO = Path(__file__).resolve().parent.joinpath('data', 'audio.mp4')
DOWNSAMPLED_AUDIO = Path(__file__).resolve().parent.joinpath('data', 'downsampled_audio.mp3')
#FRAMES_DIR = Path(__file__).resolve().parent.joinpath('static/data/frames')
#FRAMES_DIR = Path(__file__).resolve().parent.joinpath('/frames')
FRAMES_DIR = Path('/root/deployment/directory/a/src/static/data/frames')



COOKIES_PATH = Path(__file__).resolve().parent.joinpath('cookies.txt')

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Configure Tesseract OCR
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    raise FileNotFoundError("Tesseract not found")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

def download_youtube_video(url, output_video, retries=3):
    """Downloads video from a YouTube URL using yt_dlp with cookies."""
    try:
        output_video.parent.mkdir(parents=True, exist_ok=True)

        if output_video.exists():
            output_video.unlink()  # Remove the existing file

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_video),
            'cookiefile': str(COOKIES_PATH),
        }

        for attempt in range(retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                logger.info(f"Downloaded video to {output_video} using yt_dlp")
                return True
            except OSError as e:
                logger.error(f"OSError occurred while downloading video: {e}. Attempt {attempt + 1} of {retries}")
                if attempt == retries - 1:
                    raise  # Raise exception if last attempt fails
            except Exception as e:
                logger.error(f"An error occurred while downloading video: {e}. Attempt {attempt + 1} of {retries}")
                if attempt == retries - 1:
                    raise  # Raise exception if last attempt fails

        return False
    except Exception as e:
        logger.error(f"An error occurred while setting up the download: {e}")
        return False

def extract_audio_from_video(video_path, audio_output_path):
    """Extracts audio from the downloaded video."""
    if audio_output_path.exists():
        audio_output_path.unlink()
    os.system(f"ffmpeg -y -i {video_path} -q:a 0 -map a {audio_output_path}")
    logger.info(f"Extracted audio to {audio_output_path}")

def downsample_audio(input_path, output_path):
    """Downsamples the audio to a lower bitrate and mono channel."""
    os.system(f"ffmpeg -y -i {input_path} -ac 1 -ar 16000 {output_path}")
    logger.info(f"Downsampled audio saved to {output_path}")

def segment_audio(audio_path, segment_length=40):
    """Segment audio into smaller chunks."""
    logger.info(f"Starting segmentation of audio: {audio_path}")
    os.system(f"ffmpeg -i {audio_path} -f segment -segment_time {segment_length} -c copy output%03d.mp3")
    segments = [f for f in os.listdir('.') if f.startswith('output') and f.endswith('.mp3')]
    logger.info(f"Segmentation completed. Number of segments: {len(segments)}")
    return segments

def transcribe_segment(segment_path):
    """Transcribe a single audio segment."""
    try:
        start_time = time.time()
        logger.info(f"Starting transcription for segment: {segment_path} at {start_time}")
        model = whisper.load_model("base")
        result = model.transcribe(segment_path)
        end_time = time.time()
        logger.info(f"Completed transcription for segment: {segment_path} at {end_time}, duration: {end_time - start_time} seconds")
        return result['text']
    except Exception as e:
        logger.error(f"Error transcribing segment {segment_path}: {e}")
        return ""

def transcribe_audio_in_parallel(audio_path):
    """Transcribe audio by segmenting and processing in parallel."""
    logger.info("Starting segmentation of audio file.")
    segments = segment_audio(audio_path)
    logger.info(f"Segmentation complete. Number of segments: {len(segments)}")

    transcriptions = []
    logger.info("Starting transcription of audio segments in parallel.")

    with ThreadPoolExecutor(max_workers=3) as executor:  # Adjust the number of workers as needed
        futures = [executor.submit(transcribe_segment, segment) for segment in segments]
        logger.info("All transcription tasks submitted.")

        for future in tqdm(as_completed(futures), total=len(futures), desc="Transcribing segments"):
            logger.info("In for future in")
            try:
                start_time = time.time()
                result = future.result()
                end_time = time.time()
                logger.info(f"Transcription result received at {end_time}, duration: {end_time - start_time} seconds, result: {result[:50]}...")  # Log the first 50 characters of the result
                transcriptions.append(result)
            except Exception as e:
                logger.error(f"Error retrieving transcription result: {e}")

    logger.info("Transcription of all segments completed.")
    
    # Combine transcriptions
    full_transcript = " ".join(transcriptions)
    return full_transcript

def extract_frames_from_video(video_path, frames_dir, frame_rate=0.05, session_id=None):
    """Extracts frames from the video at a specified frame rate (frames per second)."""
    def save_frame(image, count):
        frame_path = frames_dir.joinpath(f"frame_{session_id}_{count:04d}.jpg")
        cv2.imwrite(str(frame_path), image)
        logger.info(f"Saved frame {count} to {frame_path}")

    try:
        frames_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"In cv2.VideoCapture: Before")

        vidcap = cv2.VideoCapture(str(video_path))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise ValueError("Frame rate (FPS) is zero. Unable to process the video.")
        interval = int(fps / frame_rate)
        success, image = vidcap.read()
        frames = []
        count = 0
        frame_count = 0
        if session_id is None:
            session_id = uuid.uuid4()  # Generate a unique session ID

        logger.info(f"In cv2.VideoCapture: Before while loop")
        
        # Preallocate a list of tuples (image, frame_number)
        frames = []

        while True:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, image = vidcap.read()
            if not success:
                break
            frames.append((image, frame_count))
            frame_count += interval


#        while success:
#            if frame_count % interval == 0:
#                frames.append((image.copy(), count))
#                count += 1
#            success, image = vidcap.read()
#            frame_count += 1

        logger.info(f"In cv2.VideoCapture: After")

        logger.info(f"Collected {count} frames. Saving them to {frames_dir}...")

        # Save frames in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(save_frame, frame, count) for frame, count in frames]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()

        logger.info(f"Extracted {count} frames to {frames_dir}")
        return session_id  # Return the session ID
    except Exception as e:
        logger.error(f"An error occurred while extracting frames: {e}")
        return None  # Ensure to return None if an error occurs


def process_audio_file(audio_path, custom_prompt, look_for):
    """Process an audio file and summarize its transcript using OpenAI's GPT model."""
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path)
    transcript = result['text']
    summary = summarize_text(transcript, custom_prompt, look_for)
    highlighted_transcript = highlight_transcript(transcript, look_for)
    return summary, transcript

def summarize_text(transcript, frame_analysis, custom_prompt, look_for, language):
    """Summarizes the transcript using OpenAI's GPT-4 model."""
    # Limit the length of the transcript to avoid token overflow
    max_transcript_length = 10000
    if (len(transcript) > max_transcript_length):
        transcript = transcript[:max_transcript_length] + "..."

    # Summarize frame analysis
    frame_analysis_summary = summarize_frame_analysis(frame_analysis, look_for, language)

    system_prompt = f"Be an expert in the topic you are summarizing. The user is interested in any one of the following topics: {look_for}."

    user_prompt = f"""{custom_prompt}
    Use the following transcript and frame analysis to write a summary of the video.
    Keep the total length of the summary to 200 words or less.

    Transcript: {transcript}

    Frame Analysis Summary: {frame_analysis_summary}

    Extract parts of the transcript and frame analysis related to {look_for} and include them in the summary with "" around them.

    Add a title to the summary.

    Make sure your summary has only the key information and true information about the main points of the topic.
    Begin with a one-sentence punchline to highlight the main point, list important details, highlight and translate the relevant parts of the transcript and frame analysis, and finish your summary with a concluding sentence."""

    if (language == 'japanese'):
        system_prompt = f"要約するトピックの専門家であってください。ユーザーは次のことに興味があります: {look_for}。"
        user_prompt = f"""{custom_prompt}
        次のトランスクリプトとフレーム分析を使用して、200語未満でビデオの要約を作成します。

        トランスクリプト: {transcript}

        フレーム分析の要約: {frame_analysis_summary}

        {look_for} に関連するトランスクリプトとフレーム分析の一部を日本語に抽出し、それらを「」で囲んで要約に含めます。
        要約にタイトルを追加してください。

        要約には、トピックの主要なポイントに関する有用で真実の情報が含まれていることを確認してください。
        主要なポイントを強調する1文のパンチラインで始め、重要な詳細をリストし、関連するトランスクリプトとフレーム分析の部分を強調し翻訳し、締めくくりの文で要約を終了してください。"""

    logger.info('Summarizing ... ')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=4096  # Adjust as needed to control the maximum length of the response
    )

    return response.choices[0].message.content

def summarize_frame_analysis(frame_analysis, look_for, language):
    """Summarizes the frame analysis using OpenAI's GPT-4 model."""
    system_prompt = "Summarize the frame-by-frame analysis to describe what is happening in the video."

    user_prompt = f"""
    The user is interested in: {look_for}.
    
    Frame Analysis: {frame_analysis}
    
    Provide a summary of what is happening in the video based on the frame analysis."""

    if language == 'japanese':
        system_prompt = "フレームごとの分析を要約して、ビデオで何が起こっているかを説明してください。"

        user_prompt = f"""
        結果を日本語で書いてください.  ユーザーは次のことに興味があります: {look_for}。
        
        フレーム分析: {frame_analysis}
        
        フレーム分析に基づいて、ビデオで何が起こっているかの要約を提供してください。"""

    logger.info('Summarizing frame analysis ... ')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=200  # Adjust as needed to control the maximum length of the response
    )

    return response.choices[0].message.content


def highlight_transcript(transcript, keywords):
    """Highlights key parts of the transcript based on keywords."""
    highlighted_transcript = transcript
    for keyword in keywords.split(','):
        highlighted_transcript = highlighted_transcript.replace(keyword.strip(), f"<mark>{keyword.strip()}</mark>")
    return highlighted_transcript

def prepare_frames_dir(frames_dir):
    """Prepare the frames directory without deleting any files."""
    if frames_dir.exists():
        logger.info(f"Frames directory exists: {frames_dir}")
    else:
        frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Frames directory created: {frames_dir}")

def validate_image(image_path):
    """Validates if the image can be opened and is not corrupted."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating image {image_path}: {e}")
        return False

def analyze_frame(frame_path, transcript, public_url_base, keywords, language):
    """Analyze a single frame using GPT-4 Vision API."""
    try:
        image_url = f"{public_url_base}/frames/{frame_path.name}"

        logger.info(f"Analyzing frame: {frame_path}, Image URL: {image_url}")  # Log message with the file name and image URL
        logger.info(f"Transcript used for frame analysis: {transcript[:500]}...")  # Log first 500 chars of transcript

        if language == 'japanese':
            prompt_text = "80語以内でフレームの要点を要約してください。このフレームはこのストーリーの一部です: {transcript}"
        else:
            prompt_text = "using 80 words or less summarize the point of the frame. this frame is part of this story : {transcript}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ],
                }
            ],
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return {"frame": frame_path.name, "description": description}
    except client.openai.APIError as e:
        logger.error(f"Invalid image error for frame {frame_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing frame {frame_path.name}: {e}")
        return None

def analyze_frames_with_gpt4(frames_dir, transcript, public_url_base, keywords, session_id, language):
    """Analyze frames using GPT-4 Vision API in parallel."""
    logger.info(f"Analyzing frames in {frames_dir} ...")
    frame_pattern = f"frame_{session_id}_*.jpg"
    frames = sorted(frames_dir.glob(frame_pattern))  # Filter by session ID

    if not frames:
        logger.warning(f"No frames found for session ID: {session_id} with pattern: {frame_pattern}")
    else:
        logger.info(f"Found {len(frames)} frames for session ID: {session_id} with pattern: {frame_pattern}")
        for frame in frames:
            logger.info(f"Frame found: {frame}")

    analysis_results = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(analyze_frame, frame, transcript, public_url_base, keywords, language) for frame in frames]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                result["frame"] = f"/frames/{result['frame']}"
                analysis_results.append(result)

    matched_frames = []
    for result in analysis_results:
        for keyword in keywords:
            logger.info(f"Comparing keyword: '{keyword.lower()}' with description: '{result['description'].lower()}'")  # Log the comparison
            similarity_score = fuzz.partial_ratio(keyword.lower(), result["description"].lower())
            if similarity_score > 60:  # Adjust threshold as needed
                matched_frames.append(result)
                break

    if not matched_frames:
        matched_frames = analysis_results  # Return all frames if no matches are found

    logger.info(f"Total matched frames: {len(matched_frames)}")
    return matched_frames



def process_youtube_video(youtube_url, custom_prompt, look_for, language, public_url_base):
    keywords = look_for.split(',')
    logger.info("Starting process for YouTube URL: %s", youtube_url)

    if not download_youtube_video(youtube_url, OUTPUT_VIDEO):
        raise Exception("An error occurred while downloading the video. Please check the URL or try another video.")
    logger.info("Video downloaded successfully")

    vidcap = cv2.VideoCapture(str(OUTPUT_VIDEO))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps != 0:
        duration = frame_count / fps
    else:
        duration = 0
        raise Exception("Error: Frame rate (FPS) is zero. Unable to process the video.")

    if duration > 2400:
        raise Exception("Video too long. Please upload a video shorter than 1hr.")

    extract_audio_from_video(OUTPUT_VIDEO, OUTPUT_AUDIO)
    downsample_audio(OUTPUT_AUDIO, DOWNSAMPLED_AUDIO)
    prepare_frames_dir(FRAMES_DIR)
    session_id = extract_frames_from_video(OUTPUT_VIDEO, FRAMES_DIR, frame_rate=0.05)

    # Transcribe the audio and log the result
    transcript = transcribe_audio_in_parallel(DOWNSAMPLED_AUDIO.as_posix())
    logger.info(f'Transcript generated: \n{transcript}')

    # Check if transcript is correctly passed
    assert transcript is not None and len(transcript) > 0, "Transcript is empty or None"

    matched_frames = analyze_frames_with_gpt4(FRAMES_DIR, transcript, public_url_base, keywords, session_id, language)

    frame_analysis_text = "\n\n".join([f"Frame: {frame['frame']}, Description: {frame['description']}" for frame in matched_frames])

    summary_text = summarize_text(transcript, frame_analysis_text, custom_prompt, keywords, language)
    summary = {"Section": summary_text}  # Ensure summary is returned as a dictionary

    highlighted_transcript = highlight_transcript(transcript, look_for)

    result = {
        "summary": summary,
        "transcript": transcript,
        "highlighted_transcript": highlighted_transcript,
        "matched_frames": matched_frames,
        "language": language,
        "youtube_url": youtube_url  # Include youtube_url in the result
    }

    logger.info(f"Matched Frames: {json.dumps(matched_frames, indent=2)}")

    read_text_data = {
        'text': summary_text,
        'language': language
    }
    response = requests.post(f"{public_url_base}/read_text", json=read_text_data)
    if response.status_code == 200:
        logger.info("Audio summary generated successfully")
    else:
        logger.error("Failed to generate audio summary")

    logger.info("Returning result: %s", result)
    return result



if __name__ == "__main__":
    # This block allows you to test the script independently
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    custom_prompt = "Summarize the main points of this video"
    look_for = "music,dance"
    language = "english"
    public_url_base = "http://localhost:5000"
    
    result = process_youtube_video(youtube_url, custom_prompt, look_for, language, public_url_base)
    print(json.dumps(result, indent=2))

