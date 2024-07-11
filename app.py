import os
import uuid
import json
from pathlib import Path
from flask import Flask, render_template, request, flash, jsonify, session, url_for, send_file, send_from_directory, redirect
from summarize_youtube_videos import process_youtube_video, process_audio_file  # Ensure this import is correct
from dotenv import load_dotenv
from rq import Queue
from rq.job import Job
from redis import Redis
import markdown2
import tempfile
from openai import OpenAI, OpenAIError
import hashlib
import logging

app = Flask(__name__)

# Load environment variables from a .env file if it exists
load_dotenv()

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

if not client.api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Define directories for frames and saved results
FRAMES_DIR = Path('/root/deployment/directory/a/src/static/data/frames')
SAVED_RESULTS_DIR = Path(__file__).resolve().parent.joinpath('saved_results')
SAVED_RESULTS_DIR.mkdir(exist_ok=True)

# Set the public URL base for generating correct URLs
app.config['PUBLIC_URL_BASE'] = os.getenv('PUBLIC_URL_BASE', 'http://aivon.co')

# Connect to Redis and initialize the queue
redis_conn = Redis()
queue = Queue(connection=redis_conn)

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.template_filter('markdown')
def markdown_filter(text):
    return markdown2.markdown(text)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/summarizer', methods=['GET', 'POST'])
def summarizer():
    results = []

    if 'previous_requests' not in session:
        session['previous_requests'] = []

    if request.method == 'POST':
        youtube_urls = request.form['youtube_urls'].splitlines()
        custom_prompt = request.form['custom_prompt']
        look_for = request.form['look_for']
        language = request.form['language']
        public_url_base = app.config['PUBLIC_URL_BASE']

        job_ids = []

        for youtube_url in youtube_urls:
            try:
                # Enqueue the processing task with the selected language
                job = queue.enqueue(
                    'summarize_youtube_videos.process_youtube_video',
                    youtube_url, custom_prompt, look_for, language, public_url_base,
                    job_timeout=1200
                )
                previous_request = {
                    'youtube_url': youtube_url,
                    'custom_prompt': custom_prompt,
                    'look_for': look_for,
                    'language': language,
                    'job_id': job.get_id()
                }

                session['previous_requests'].insert(0, previous_request)
                if len(session['previous_requests']) > 3:
                    session['previous_requests'].pop()

                app.logger.info(f"Job {job.get_id()} enqueued successfully")
                flash(f"Job {job.get_id()} enqueued successfully", 'info')
                job_ids.append(job.get_id())

            except Exception as e:
                flash(str(e), 'error')

        session['job_ids'] = job_ids

    if 'job_ids' in session:
        for job_id in session['job_ids']:
            try:
                job = Job.fetch(job_id, connection=redis_conn)
                if job.is_finished:
                    summary_link = url_for('view_saved_summary', unique_id=job_id, _external=True)

                    results.append({
                        'job_id': job_id,
                        'youtube_url': next(req['youtube_url'] for req in session['previous_requests'] if req['job_id'] == job_id),
                        'summary_link': summary_link
                    })
                elif job.is_failed:
                    flash(f'Job {job_id} failed. Please try again.', 'error')
                else:
                    flash(f"Job {job_id} is still in progress", 'info')
            except Exception as e:
                flash(f"Error fetching job {job_id}: {str(e)}", 'error')

    # Clean up session data to remove outdated job references
    session['previous_requests'] = [req for req in session['previous_requests'] if Job.exists(req['job_id'], connection=redis_conn)]

    return render_template('index.html', results=results, previous_requests=session['previous_requests'])

@app.route('/saved_summaries')
def get_saved_summaries():
    saved_summaries = []
    for result_file in sorted(SAVED_RESULTS_DIR.glob('*.json'), key=os.path.getmtime, reverse=True)[:20]:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
            app.logger.info(f"Result data structure: {result_data}")

            try:
                full_summary = result_data['summary']['Section']
            except KeyError as e:
                app.logger.error(f"KeyError: {e} in file {result_file}")
                continue
            except TypeError as e:
                app.logger.error(f"TypeError: {e} in file {result_file}")
                continue

            truncated_summary = ' '.join(full_summary.split()[:10]) + '...'  # Get the first 5 words
            saved_summaries.append({
                'id': result_file.stem,
                'summary': truncated_summary
            })
    return jsonify({'status': 'success', 'saved_summaries': saved_summaries})

@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as e:
        app.logger.error(f"Error fetching job {job_id}: {str(e)}")
        return jsonify({'status': 'error', 'message': f"No such job: {job_id}"}), 404

    if job.is_finished:
        result = job.result
        app.logger.info(f"Job finished with result: {result}")

        try:
            summary = {section: markdown2.markdown(content) for section, content in result["summary"].items()}
            transcript = result["transcript"]
            highlighted_transcript = result.get("highlighted_transcript", "")
            matched_frames = result["matched_frames"]
            for frame in matched_frames:
                frame["frame_url"] = f"{app.config['PUBLIC_URL_BASE']}/frames/{Path(frame['frame']).name}"

        except Exception as e:
            app.logger.error(f"Error processing job result: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

        return jsonify({'status': 'finished', 'result': {
            'summary': summary,
            'transcript': transcript,
            'highlighted_transcript': highlighted_transcript,
            'matched_frames': matched_frames,
            'youtube_url': result.get('youtube_url')
        }})
    elif job.is_failed:
        app.logger.error(f"Job failed with exception: {job.exc_info}")
        return jsonify({'status': 'failed', 'result': str(job.exc_info)})
    else:
        progress = job.meta.get('progress', 0) * 100
        app.logger.info(f"Job status: {job.get_status()}, progress: {progress}")
        return jsonify({'status': job.get_status(), 'progress': progress})

@app.route('/frames/<filename>')
def serve_frame(filename):
    try:
        app.logger.info(f"Serving frame from path: {FRAMES_DIR / filename}")
        return send_from_directory(FRAMES_DIR, filename)
    except Exception as e:
        app.logger.error(f"Error serving frame {filename}: {e}")
        return '', 404

@app.route('/serve_summary_audio/<job_id>', methods=['GET'])
def serve_summary_audio(job_id):
    return send_file('path/to/audio/file.mp3', mimetype='audio/mpeg')

@app.route('/podcast', methods=['GET', 'POST'])
def podcast():
    summary, transcript = None, None

    if request.method == 'POST':
        if 'youtube_url' in request.form:
            youtube_url = request.form['youtube_url']
            try:
                job = queue.enqueue('summarize_youtube_videos.process_youtube_video', youtube_url, "", "")
            except Exception as e:
                flash(str(e), 'error')
        elif 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            audio_path = os.path.join('uploads', audio_file.filename)
            audio_file.save(audio_path)
            try:
                job = queue.enqueue('summarize_youtube_videos.process_audio_file', audio_path, "", "")
            except Exception as e:
                flash(str(e), 'error')

    return render_template('podcast.html', summary=summary, transcript=transcript)

@app.route('/save_summary', methods=['POST'])
def save_summary():
    data = request.json
    job_id = data.get('job_id')
    summary = data.get('summary')
    transcript = data.get('transcript')
    matched_frames = data.get('matched_frames')
    youtube_url = data.get('youtube_url')

    if not job_id or not summary or not transcript or not matched_frames or not youtube_url:
        return jsonify({'status': 'error', 'message': 'Invalid data provided'}), 400

    unique_id = job_id
    result_path = SAVED_RESULTS_DIR.joinpath(f'{unique_id}.json')

    result_data = {
        'job_id': job_id,
        'summary': summary,
        'transcript': transcript,
        'matched_frames': matched_frames,
        'youtube_url': youtube_url
    }

    try:
        with open(result_path, 'w') as f:
            json.dump(result_data, f)

        sharable_link = url_for('view_saved_summary', unique_id=unique_id, _external=True)
        return jsonify({'status': 'success', 'link': sharable_link})
    except Exception as e:
        app.logger.error(f"Error saving summary: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to save summary'}), 500

@app.route('/read_text', methods=['POST'])
def read_text():
    data = request.json
    text = data.get('text')
    language = data.get('language', 'en')
    app.logger.info(f"In read_text(): {text}")

    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    unique_id = hashlib.md5(f"{text}-{language}".encode()).hexdigest()
    speech_file_path = Path(tempfile.gettempdir()) / f"speech_{unique_id}.mp3"

    if speech_file_path.exists():
        app.logger.info(f"Returning existing audio file {speech_file_path}")
        return send_file(speech_file_path, mimetype='audio/mp3')

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3"
        )

        with open(speech_file_path, 'wb') as audio_file:
            audio_file.write(response.content)

        if not speech_file_path.exists():
            raise ValueError("No audio file generated from OpenAI")

        app.logger.info(f"Audio file saved to {speech_file_path}")
        return send_file(speech_file_path, mimetype='audio/mp3')

    except OpenAIError as e:
        app.logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    except Exception as e:
        app.logger.error(f"Exception: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/summary/<unique_id>', methods=['GET'])
def view_saved_summary(unique_id):
    result_path = SAVED_RESULTS_DIR.joinpath(f'{unique_id}.json')
    app.logger.info(f"Fetching summary for unique_id: {unique_id}")

    if not result_path.exists():
        app.logger.error(f"Summary not found for unique_id: {unique_id}")
        return jsonify({'status': 'error', 'message': 'Summary not found'}), 404

    with open(result_path, 'r') as f:
        result_data = json.load(f)
        app.logger.info(f"Result data loaded for unique_id: {unique_id}")

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # If the request is an AJAX request, return JSON
        return jsonify({'status': 'success', 'result': result_data})
    else:
        # Otherwise, render the HTML template
        try:
            summary = {section: markdown2.markdown(content) for section, content in result_data["summary"].items()}
            transcript = result_data["transcript"]
            highlighted_transcript = result_data.get("highlighted_transcript", "")
            matched_frames = result_data["matched_frames"]
            language = result_data.get("language", "english")

            youtube_url = result_data.get('youtube_url')
            if not youtube_url:
                flash("YouTube URL not found.", 'error')
                return redirect(url_for('summarizer'))

            for frame in matched_frames:
                frame["frame_url"] = f"{app.config['PUBLIC_URL_BASE']}/frames/{Path(frame['frame']).name}"

            return render_template('summary.html', summary=summary, transcript=transcript, highlighted_transcript=highlighted_transcript, matched_frames=matched_frames, language=language, youtube_url=youtube_url)
        except Exception as e:
            app.logger.error(f"Error processing summary for unique_id: {unique_id}: {e}")
            return jsonify({'status': 'error', 'message': 'Error processing summary'}), 500


@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(SAVED_RESULTS_DIR, filename, as_attachment=False)

@app.route('/saved_summaries_page')
def saved_summaries_page():
    return render_template('saved_summaries.html')

@app.route('/summarizer_jp', methods=['GET', 'POST'])
def summarizer_jp():
    results = []

    if 'previous_requests' not in session:
        session['previous_requests'] = []

    if request.method == 'POST':
        youtube_urls = request.form['youtube_urls'].splitlines()
        custom_prompt = request.form.get('custom_prompt', '')
        look_for = request.form.get('look_for', '')
        language = request.form['language']
        public_url_base = app.config['PUBLIC_URL_BASE']

        job_ids = []

        for youtube_url in youtube_urls:
            try:
                # Enqueue the processing task with the selected language
                job = queue.enqueue('summarize_youtube_videos.process_youtube_video', youtube_url, custom_prompt, look_for, language, public_url_base, job_timeout=1200)
                previous_request = {
                    'youtube_url': youtube_url,
                    'custom_prompt': custom_prompt,
                    'look_for': look_for,
                    'language': language,
                    'job_id': job.get_id()
                }

                session['previous_requests'].insert(0, previous_request)
                if len(session['previous_requests']) > 3:
                    session['previous_requests'].pop()

                app.logger.info(f"Job {job.get_id()} enqueued successfully")
                flash(f"Job {job.get_id()} enqueued successfully", 'info')
                job_ids.append(job.get_id())

            except Exception as e:
                flash(str(e), 'error')

        session['job_ids'] = job_ids

    if 'job_ids' in session:
        for job_id in session['job_ids']:
            try:
                job = Job.fetch(job_id, connection=redis_conn)
                if job.is_finished:
                    summary_link = url_for('view_saved_summary', unique_id=job_id, _external=True)

                    results.append({
                        'job_id': job_id,
                        'youtube_url': next(req['youtube_url'] for req in session['previous_requests'] if req['job_id'] == job_id),
                        'summary_link': summary_link
                    })
                elif job.is_failed:
                    flash(f'Job {job_id} failed. Please try again.', 'error')
                else:
                    flash(f"Job {job_id} is still in progress", 'info')
            except Exception as e:
                flash(f"Error fetching job {job_id}: {str(e)}", 'error')

    # Clean up session data to remove outdated job references
    session['previous_requests'] = [req for req in session['previous_requests'] if Job.exists(req['job_id'], connection=redis_conn)]

    return render_template('summarizer_jp.html', results=results, previous_requests=session['previous_requests'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
