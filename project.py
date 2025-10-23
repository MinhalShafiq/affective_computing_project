import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from dataclasses import dataclass
import sys
import os
import json
import tempfile
import uuid
from datetime import datetime
import yt_dlp
import subprocess

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add EmotionCLIP src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'EmotionCLIP', 'src')))

try:
    from models.base import CLIP, CLIPVisionCfg, CLIPTextCfg
    from models.tokenizer import tokenize
except ImportError as e:
    logger.error("Failed to import EmotionCLIP modules.")
    raise e

# Models
@dataclass
class EmotionFrame:
    frame_idx: int
    timestamp: float
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    bbox: Tuple[int, int, int, int] = None

class AnalyzeRequest(BaseModel):
    video_url: str

class JobResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    status: str
    result: Optional[Dict[str, float]] = None
    error: Optional[str] = None

# Emotion Detector
class EmotionCLIPDetector:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.emotion_labels = [
            "happy", "sad", "angry", "fear", "surprise", 
            "disgust", "contempt", "neutral"
        ]
        
        templates = [
            "a photo of a {} person",
            "a person expressing {}",
            "a face showing {} emotion",
            "someone feeling {}",
            "{} facial expression"
        ]
        self.emotion_prompts = []
        self.prompt_to_emotion = []
        for emotion in self.emotion_labels:
            for template in templates:
                self.emotion_prompts.append(template.format(emotion))
                self.prompt_to_emotion.append(emotion)
        
        self.emotion_embeddings = self._encode_prompts()
        logger.info(f"✓ EmotionCLIP model loaded on {device}")
        
    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        logger.info(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        config_path = os.path.join(os.path.dirname(__file__), 'EmotionCLIP', 'src', 'models', 'model_configs', 'ViT-B-32.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                'embed_dim': 512,
                'vision_cfg': {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 32},
                'text_cfg': {'context_length': 77, 'vocab_size': 49408, 'width': 512, 'heads': 8, 'layers': 12}
            }
        
        vision_cfg = CLIPVisionCfg(**config['vision_cfg'])
        text_cfg = CLIPTextCfg(**config['text_cfg'])
        model = CLIP(embed_dim=config['embed_dim'], vision_cfg=vision_cfg, text_cfg=text_cfg, quick_gelu=False)
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}")
        return model.to(self.device)
    
    def _encode_prompts(self) -> torch.Tensor:
        with torch.no_grad():
            text_tokens = tokenize(self.emotion_prompts).to(self.device)
            embeddings = self.model.encode_text(text_tokens)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    def extract_frames(self, video_path: str, fps: int = 8) -> List[Tuple[np.ndarray, float]]:
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                frames.append((frame, timestamp))
            frame_count += 1
        cap.release()
        return frames
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return list(faces)
    
    def preprocess_frame(self, frame: np.ndarray, target_size: int = 224) -> torch.Tensor:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (target_size, target_size))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)
    
    def detect_emotions(self, frame: np.ndarray) -> Tuple[Dict[str, float], str, float]:
        with torch.no_grad():
            frame_tensor = self.preprocess_frame(frame)
            batch_size = frame_tensor.shape[0]
            height = frame_tensor.shape[2]
            width = frame_tensor.shape[3]
            mask = torch.ones(batch_size, height, width, dtype=torch.bool, device=self.device)
            image_features = self.model.encode_image(frame_tensor, mask)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ self.emotion_embeddings.t()) * 100
            probabilities = F.softmax(logits, dim=-1)[0]
        
        emotion_scores = {emotion: [] for emotion in self.emotion_labels}
        for idx, prob in enumerate(probabilities):
            emotion = self.prompt_to_emotion[idx]
            emotion_scores[emotion].append(prob.item())
        
        emotions = {emotion: np.mean(scores) for emotion, scores in emotion_scores.items()}
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        else:
            emotions = {k: 1.0/len(emotions) for k in emotions.keys()}
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        return emotions, dominant_emotion, confidence
    
    def process_video(self, video_path: str, fps: int = 8, use_face_detection: bool = True) -> List[EmotionFrame]:
        frames_data = self.extract_frames(video_path, fps)
        if not frames_data:
            logger.warning("No frames extracted from video.")
            return []
        
        results = []
        total_frames = len(frames_data)
        logger.info(f"Processing {total_frames} frames...")
        
        for idx, (frame, timestamp) in enumerate(frames_data):
            # Log progress every 10% or at least every frame if fewer than 10 frames
            if idx % max(1, total_frames // 10) == 0:
                logger.info(f"Progress: {idx + 1}/{total_frames} frames ({(idx + 1) / total_frames * 100:.1f}%)")
            
            process_frame = frame
            bbox = None
            if use_face_detection:
                faces = self.detect_faces(frame)
                if faces:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    y_start = max(0, y - 10)
                    y_end = min(frame.shape[0], y + h + 10)
                    x_start = max(0, x - 10)
                    x_end = min(frame.shape[1], x + w + 10)
                    process_frame = frame[y_start:y_end, x_start:x_end]
                    bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
            
            emotions, dominant, confidence = self.detect_emotions(process_frame)
            results.append(EmotionFrame(idx, timestamp, emotions, dominant, confidence, bbox))
        
        logger.info(f"✓ Processing complete: {total_frames} frames analyzed")
        return results
    
    def smooth_emotions(self, results: List[EmotionFrame], window_size: int = 9) -> List[EmotionFrame]:
        if not results:
            return []
        smoothed = []
        weights = np.exp(-0.5 * np.linspace(-2, 2, window_size)**2)
        weights = weights / weights.sum()
        for i, frame in enumerate(results):
            start = max(0, i - window_size // 2)
            end = min(len(results), i + window_size // 2 + 1)
            window = results[start:end]
            w = weights[-(end-start):]
            w = w / w.sum() if w.sum() > 0 else w
            avg_emotions = {}
            for emotion in self.emotion_labels:
                avg_emotions[emotion] = sum(f.emotions[emotion] * wi for f, wi in zip(window, w))
            dominant_emotion = max(avg_emotions, key=avg_emotions.get)
            confidence = avg_emotions[dominant_emotion]
            smoothed.append(EmotionFrame(
                frame.frame_idx, frame.timestamp, avg_emotions, dominant_emotion, confidence, frame.bbox
            ))
        return smoothed
    
    def generate_report(self, results: List[EmotionFrame]) -> Dict:
        total = len(results)
        if total == 0:
            # Fallback: all neutral
            return {
                "total_frames": 0,
                "overall_mood": "neutral",
                "emotion_distribution": {e: 0.0 for e in self.emotion_labels},
                "average_confidence": 0.0,
            }
        counts = {e: 0 for e in self.emotion_labels}
        for f in results:
            counts[f.dominant_emotion] += 1
        dist = {e: (c / total * 100) for e, c in counts.items()}
        mood = max(dist, key=dist.get)
        avg_conf = np.mean([f.confidence for f in results])
        return {
            "total_frames": total,
            "overall_mood": mood,
            "emotion_distribution": dist,
            "average_confidence": float(avg_conf),
        }

# Video Downloader
def download_video_from_url(video_url: str, temp_dir: str) -> str:
    output_template = os.path.join(temp_dir, 'input_video.%(ext)s')
    
    # Path to your exported cookies file (export from Chrome using Get cookies.txt extension)
    cookies_file = os.path.join(os.path.dirname(__file__), "cookies.txt")
    
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '-o', output_template,
        '--no-playlist',
        video_url
    ]
    
    # Only add cookies if file exists
    if os.path.exists(cookies_file):
        cmd.insert(1, '--cookies')
        cmd.insert(2, cookies_file)
        logger.info(f"Using cookies from: {cookies_file}")
    else:
        logger.warning(f"Cookies file not found at {cookies_file}, attempting without cookies")
    
    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Download output: {result.stdout}")
        
        # Find the downloaded file
        files = os.listdir(temp_dir)
        if files:
            video_path = os.path.join(temp_dir, files[0])
            logger.info(f"Downloaded video: {video_path}")
            return video_path
        else:
            raise FileNotFoundError("No video file found after download")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed with exit code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        logger.error(f"stdout: {e.stdout}")
        raise Exception(f"Failed to download video: {e.stderr}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

# Job System
app = FastAPI(title="Emotion Analysis API")
JOBS: Dict[str, dict] = {}
MODEL_CHECKPOINT = "EmotionCLIP/preprocessing/emotionclip_latest.pt"

def process_job(job_id: str, video_url: str):
    try:
        JOBS[job_id] = {"status": "processing"}
        logger.info(f"Starting job {job_id} for URL: {video_url}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing EmotionCLIP detector on {device}...")
        detector = EmotionCLIPDetector(model_path=MODEL_CHECKPOINT, device=device)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Downloading video...")
            video_path = download_video_from_url(video_url, temp_dir)
            logger.info(f"✓ Video downloaded to {video_path}")
            
            logger.info("Starting emotion detection...")
            results = detector.process_video(video_path, fps=8, use_face_detection=True)
            
            logger.info("Smoothing emotion predictions over time...")
            smoothed = detector.smooth_emotions(results, window_size=9)
            logger.info("✓ Smoothing complete")
            
            logger.info("Generating final report...")
            report = detector.generate_report(smoothed)
            logger.info("✓ Report generated")
            
            emotion_scores = {
                emotion: float(score / 100.0)
                for emotion, score in report['emotion_distribution'].items()
            }
            
            JOBS[job_id] = {
                "status": "completed",
                "result": emotion_scores,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✓ Job {job_id} completed successfully")
            logger.info(f"Overall mood: {report['overall_mood']}")
            logger.info(f"Average confidence: {report['average_confidence']:.2%}")
            logger.info(f"Emotion distribution: {emotion_scores}")
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Job {job_id} failed: {error_msg}", exc_info=True)
        JOBS[job_id] = {
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.utcnow().isoformat()
        }

# Endpoints
@app.post("/analyze", response_model=JobResponse)
async def submit_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(MODEL_CHECKPOINT):
        raise HTTPException(status_code=500, detail="Model checkpoint not found")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued"}
    background_tasks.add_task(process_job, job_id, request.video_url)
    return {"job_id": job_id}

@app.get("/result/{job_id}", response_model=JobStatusResponse)
async def get_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    return {
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error")
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if os.path.exists(MODEL_CHECKPOINT) else "unhealthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }