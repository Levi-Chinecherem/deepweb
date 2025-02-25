from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
import cv2
import numpy as np
import librosa
import tempfile
import logging
import mimetypes
from django.views.decorators.csrf import csrf_protect
import ffmpeg
import time

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_metadata(video_path):
    """Extract video metadata using OpenCV."""
    logger.debug(f"Extracting metadata for {video_path}")
    start_time = time.time()
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError("Invalid video file")
        
        metadata = {
            'size': os.path.getsize(video_path),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        }
        cap.release()
        logger.info(f"Metadata extracted: {metadata}")
        logger.debug(f"Metadata extraction took {time.time() - start_time:.2f} seconds")
        return metadata
    except Exception as e:
        logger.error(f"Metadata extraction failed: {str(e)}", exc_info=True)
        raise

def extract_emotion_consistency(video_path):
    """Analyze video for deepfake detection with optimized audio extraction."""
    logger.info(f"Starting deepfake analysis for {video_path}")
    start_time = time.time()
    
    try:
        # Video processing
        logger.debug("Opening video with OpenCV")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error("Failed to open video file")
            raise ValueError("Could not open video file")
            
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video metadata: FPS={fps}, Frame Count={frame_count}")
        
        logger.debug("Initializing face cascade classifier")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            raise ValueError("Failed to load face cascade classifier")
        
        visual_emotions = []
        frame_interval = max(1, int(fps / 2))
        logger.debug(f"Processing frames with interval: {frame_interval}")
        
        for i in range(0, frame_count, frame_interval):
            logger.debug(f"Processing frame {i}")
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if not ret:
                logger.warning(f"Failed to read frame {i}, stopping video analysis")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            logger.debug(f"Detected {len(faces)} faces in frame {i}")
            visual_emotions.append(1 if len(faces) > 0 else 0)
        
        video.release()
        logger.info(f"Video analysis complete. Frames processed: {len(visual_emotions)}")
        
        # Audio extraction with ffmpeg-python
        logger.debug("Starting audio extraction with ffmpeg")
        audio_start_time = time.time()
        audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream, 
                audio_temp.name, 
                format='wav', 
                acodec='pcm_s16le',  # 16-bit PCM for compatibility
                ac=1,                # Mono
                ar='16k',            # 16kHz sample rate
                vn=True,             # No video
                loglevel='info'      # Log FFmpeg output
            )
            logger.debug(f"FFmpeg command: {' '.join(stream.compile())}")
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            logger.info(f"Audio extracted to {audio_temp.name} in {time.time() - audio_start_time:.2f} seconds")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise ValueError(f"Audio extraction failed: {e.stderr.decode() if e.stderr else str(e)}")
        
        # Audio processing
        logger.debug("Loading audio with librosa")
        y, sr = librosa.load(audio_temp.name, sr=None)
        logger.info(f"Audio loaded: Sample rate={sr}, Duration={len(y)/sr}s")
        os.unlink(audio_temp.name)
        logger.debug(f"Cleaned up temporary audio file: {audio_temp.name}")
        
        logger.debug("Analyzing audio features")
        pitch = librosa.pitch_tuning(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        audio_segments = librosa.effects.split(y)
        audio_emotion = len(audio_segments)
        energy = np.mean(librosa.feature.rms(y=y))
        logger.info(f"Audio analysis: Pitch={pitch}, Tempo={tempo}, Segments={audio_emotion}, Energy={energy}")
        
        # Calculate scores
        total_duration = frame_count / fps
        visual_score = (sum(visual_emotions) / len(visual_emotions)) * 100 if visual_emotions else 0
        audio_score = (audio_emotion / (len(y) / sr)) * 100 * (1 + energy)
        consistency_diff = abs(visual_score - audio_score)
        is_fake = (consistency_diff > 30) or (visual_score < 20 and audio_score > 80)
        
        logger.info(f"Results: Visual={visual_score}%, Audio={audio_score}%, Diff={consistency_diff}, Fake={is_fake}")
        
        end_time = time.time()
        logger.info(f"Deepfake analysis completed in {end_time - start_time:.2f} seconds")
        
        return {
            'visual_consistency': round(visual_score, 2),
            'audio_consistency': round(audio_score, 2),
            'decision': 'Fake' if is_fake else 'Real',
            'status': 'success',
            'confidence': round(100 - (consistency_diff / 100) * 50, 2)
        }
    except Exception as e:
        logger.error(f"Error in deepfake analysis: {str(e)}", exc_info=True)
        return {
            'visual_consistency': 0,
            'audio_consistency': 0,
            'decision': 'Error',
            'status': 'error',
            'error_message': str(e),
            'confidence': 0
        }
    finally:
        if 'video' in locals():
            video.release()
            logger.debug("Video capture released")

@csrf_protect
def home(request):
    logger.info(f"Received {request.method} request to home")
    
    if request.method == 'POST':
        logger.debug("Processing POST request")
        if 'videos' not in request.FILES:
            logger.warning("No videos uploaded in POST request")
            return render(request, 'home.html', {'error': 'No videos uploaded'})
            
        temp_files = []
        results = []
        uploaded_files = request.FILES.getlist('videos')
        logger.info(f"Received {len(uploaded_files)} video files")
        
        max_size = 100 * 1024 * 1024
        allowed_types = {'video/mp4', 'video/avi', 'video/mov'}
        
        for video in uploaded_files:
            logger.info(f"Processing video: {video.name}")
            content_type = mimetypes.guess_type(video.name)[0] or video.content_type
            logger.debug(f"Content type: {content_type}, Size: {video.size}")
            
            if video.size > max_size:
                logger.warning(f"File {video.name} exceeds 100MB")
                results.append({'name': video.name, 'error': 'File exceeds 100MB'})
                continue
            if content_type not in allowed_types:
                logger.warning(f"Unsupported format for {video.name}: {content_type}")
                results.append({'name': video.name, 'error': 'Unsupported format'})
                continue
                
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                logger.debug(f"Writing video to temporary file: {tmp.name}")
                for chunk in video.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
                temp_files.append(tmp_path)
            
            try:
                metadata = extract_metadata(tmp_path)
                results.append({'name': video.name, 'metadata': metadata, 'analysis': None})
            except Exception as e:
                logger.error(f"Metadata extraction failed for {video.name}: {str(e)}", exc_info=True)
                results.append({'name': video.name, 'error': f'Metadata extraction failed: {str(e)}'})
        
        request.session['temp_files'] = temp_files
        logger.info("Redirecting to metadata page with initial results")
        return render(request, 'metadata.html', {'results': results})
    
    logger.debug("Rendering home page")
    return render(request, 'home.html')

@csrf_protect
def analyze_deepfake(request):
    logger.info("Received request to analyze deepfake")
    
    if request.method != 'POST' or 'temp_files' not in request.session:
        logger.warning("Invalid request or no temp files in session")
        return redirect('home')
    
    temp_files = request.session.get('temp_files', [])
    results = []
    
    for tmp_path in temp_files:
        logger.info(f"Analyzing deepfake for {tmp_path}")
        try:
            video_name = os.path.basename(tmp_path)  # Simplified
            metadata = extract_metadata(tmp_path)
            analysis = extract_emotion_consistency(tmp_path)
            results.append({'name': video_name, 'metadata': metadata, 'analysis': analysis})
        except Exception as e:
            logger.error(f"Deepfake analysis failed for {tmp_path}: {str(e)}", exc_info=True)
            results.append({'name': video_name, 'error': f'Analysis failed: {str(e)}'})
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
    
    del request.session['temp_files']
    logger.info("Rendering final results page")
    return render(request, 'results.html', {'results': results})

def process_status(request):
    logger.info("Received request for process status")
    return JsonResponse({'status': 'complete'})