import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch.nn.functional as F
from dataclasses import dataclass
import sys
import os
import json

# Add EmotionCLIP src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'EmotionCLIP', 'src')))

from models.base import CLIP, CLIPVisionCfg, CLIPTextCfg
from models.tokenizer import tokenize

@dataclass
class EmotionFrame:
    """Container for frame-level emotion data"""
    frame_idx: int
    timestamp: float
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    bbox: Tuple[int, int, int, int] = None

class EmotionCLIPDetector:
    """Main emotion detection pipeline using EmotionCLIP"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize EmotionCLIP detector
        
        Args:
            model_path: Path to pretrained EmotionCLIP checkpoint (.pt file)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Emotion labels for CLIP text encoding
        self.emotion_labels = [
            "happy", "sad", "angry", "calm", "fearful", 
            "surprised", "disgusted", "neutral"
        ]
        self.emotion_prompts = [
            f"A person who is {emotion}" for emotion in self.emotion_labels
        ]
        
        # Encode emotion prompts once
        self.emotion_embeddings = self._encode_prompts()
        
        print(f"✓ EmotionCLIP model loaded on {device}")
        
    def _load_model(self, model_path: str):
        """Load pretrained EmotionCLIP model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        print(f"Loading checkpoint from {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        
        # Load model config from JSON
        config_path = os.path.join(os.path.dirname(__file__), 'EmotionCLIP', 'src', 'models', 'model_configs', 'ViT-B-32.json')
        
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}...")
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print("Warning: Config file not found, using default ViT-B/32 settings")
            config = {
                'embed_dim': 512,
                'vision_cfg': {
                    'image_size': 224,
                    'layers': 12,
                    'width': 768,
                    'patch_size': 32
                },
                'text_cfg': {
                    'context_length': 77,
                    'vocab_size': 49408,
                    'width': 512,
                    'heads': 8,
                    'layers': 12
                }
            }
        
        # Create config objects
        vision_cfg = CLIPVisionCfg(
            layers=config['vision_cfg']['layers'],
            width=config['vision_cfg']['width'],
            patch_size=config['vision_cfg']['patch_size'],
            image_size=config['vision_cfg']['image_size'],
        )
        
        text_cfg = CLIPTextCfg(
            context_length=config['text_cfg']['context_length'],
            vocab_size=config['text_cfg']['vocab_size'],
            width=config['text_cfg']['width'],
            heads=config['text_cfg']['heads'],
            layers=config['text_cfg']['layers'],
        )
        
        # Initialize CLIP model
        model = CLIP(
            embed_dim=config['embed_dim'],
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=False
        )
        
        print("✓ CLIP model initialized")
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Model state dict loaded successfully (strict)")
        except RuntimeError as e:
            print(f"Attempting flexible loading: {e}")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        
        model = model.to(self.device)
        return model
    
    def _encode_prompts(self) -> torch.Tensor:
        """Encode emotion prompts using CLIP text encoder"""
        prompts = self.emotion_prompts
        
        with torch.no_grad():
            # Tokenize prompts
            text_tokens = tokenize(prompts).to(self.device)
            
            # Encode using model's text encoder
            embeddings = self.model.encode_text(text_tokens)
        
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def extract_frames(self, video_path: str, fps: int = 8) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract (default 8 for efficiency)
            
        Returns:
            List of tuples (frame, timestamp)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        frame_count = 0
        
        print(f"Video FPS: {video_fps}, extracting every {frame_interval} frames (~{fps} FPS)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                frames.append((frame, timestamp))
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame using OpenCV Haar Cascade
        
        Args:
            frame: Input frame
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return list(faces)
    
    def preprocess_frame(self, frame: np.ndarray, target_size: int = 224) -> torch.Tensor:
        """
        Preprocess frame for CLIP model
        
        Args:
            frame: Input frame (BGR)
            target_size: Target size for CLIP (typically 224)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        resized = cv2.resize(rgb_frame, (target_size, target_size))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor and reshape (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.unsqueeze(0).to(self.device)
    
    def detect_emotions(self, frame: np.ndarray) -> Tuple[Dict[str, float], str, float]:
        """
        Detect emotions in a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (emotion_dict, dominant_emotion, confidence)
        """
        with torch.no_grad():
            # Preprocess frame
            frame_tensor = self.preprocess_frame(frame)
            
            # Create mask (all ones - no masking needed for single frame)
            batch_size = frame_tensor.shape[0]
            mask = torch.ones(batch_size, frame_tensor.shape[2], frame_tensor.shape[3], 
                            dtype=torch.bool, device=self.device)
            
            # Get image embeddings
            image_features = self.model.encode_image(frame_tensor, mask)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity with emotion prompts
            logits = (image_features @ self.emotion_embeddings.t()) * 100
            probabilities = F.softmax(logits, dim=-1)[0]
        
        # Create emotion dictionary
        emotions = {
            label: prob.item() 
            for label, prob in zip(self.emotion_labels, probabilities)
        }
        
        # Get dominant emotion
        dominant_idx = probabilities.argmax().item()
        dominant_emotion = self.emotion_labels[dominant_idx]
        confidence = probabilities[dominant_idx].item()
        
        return emotions, dominant_emotion, confidence
    
    def process_video(self, video_path: str, fps: int = 8, use_face_detection: bool = True) -> List[EmotionFrame]:
        """
        Process entire video frame-by-frame
        
        Args:
            video_path: Path to video file
            fps: Frames per second to analyze
            use_face_detection: Whether to detect and crop faces
            
        Returns:
            List of EmotionFrame objects
        """
        frames_data = self.extract_frames(video_path, fps)
        results = []
        
        print(f"\nProcessing {len(frames_data)} frames...")
        
        for idx, (frame, timestamp) in enumerate(frames_data):
            if idx % max(1, len(frames_data) // 10) == 0:
                print(f"  Progress: {idx + 1}/{len(frames_data)} frames")
            
            process_frame = frame
            bbox = None
            
            if use_face_detection:
                # Detect faces
                faces = self.detect_faces(frame)
                
                if len(faces) > 0:
                    # Use largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    # Ensure we don't go out of bounds
                    y_start = max(0, y - 10)
                    y_end = min(frame.shape[0], y + h + 10)
                    x_start = max(0, x - 10)
                    x_end = min(frame.shape[1], x + w + 10)
                    process_frame = frame[y_start:y_end, x_start:x_end]
                    bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
            
            # Detect emotions
            emotions, dominant, confidence = self.detect_emotions(process_frame)
            
            # Create emotion frame
            emotion_frame = EmotionFrame(
                frame_idx=idx,
                timestamp=timestamp,
                emotions=emotions,
                dominant_emotion=dominant,
                confidence=confidence,
                bbox=bbox
            )
            results.append(emotion_frame)
        
        return results
    
    def smooth_emotions(self, results: List[EmotionFrame], 
                       window_size: int = 5) -> List[EmotionFrame]:
        """
        Smooth emotion predictions over time using moving average
        
        Args:
            results: List of EmotionFrame objects
            window_size: Smoothing window size
            
        Returns:
            Smoothed results
        """
        smoothed = []
        
        for i, frame in enumerate(results):
            start = max(0, i - window_size // 2)
            end = min(len(results), i + window_size // 2 + 1)
            
            window = results[start:end]
            
            # Average emotion probabilities
            avg_emotions = {}
            for emotion in self.emotion_labels:
                avg_emotions[emotion] = np.mean([
                    f.emotions[emotion] for f in window
                ])
            
            dominant_emotion = max(avg_emotions, key=avg_emotions.get)
            confidence = avg_emotions[dominant_emotion]
            
            smoothed_frame = EmotionFrame(
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
                emotions=avg_emotions,
                dominant_emotion=dominant_emotion,
                confidence=confidence,
                bbox=frame.bbox
            )
            smoothed.append(smoothed_frame)
        
        return smoothed
    
    def generate_report(self, results: List[EmotionFrame]) -> Dict:
        """
        Generate summary report
        
        Args:
            results: List of EmotionFrame objects
            
        Returns:
            Summary report dictionary
        """
        total_frames = len(results)
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        
        for frame in results:
            emotion_counts[frame.dominant_emotion] += 1
        
        emotion_percentages = {
            emotion: (count / total_frames * 100)
            for emotion, count in emotion_counts.items()
        }
        
        overall_mood = max(emotion_percentages, key=emotion_percentages.get)
        
        report = {
            "total_frames": total_frames,
            "overall_mood": overall_mood,
            "emotion_distribution": emotion_percentages,
            "emotion_counts": emotion_counts,
            "average_confidence": np.mean([f.confidence for f in results]),
            "summary": f"Overall mood is {overall_mood}. "
                      f"Dominant emotion appears in {emotion_percentages[overall_mood]:.1f}% of frames."
        }
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize detector with your pretrained model
    model_checkpoint = "EmotionCLIP/preprocessing/emotionclip_latest.pt"
    
    try:
        detector = EmotionCLIPDetector(
            model_path=model_checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Process video
        video_path = "test_1.mp4"
        if os.path.exists(video_path):
            print(f"\nProcessing video: {video_path}")
            results = detector.process_video(video_path, fps=8, use_face_detection=True)
            
            # Smooth results
            print("\nSmoothing predictions over time...")
            smoothed_results = detector.smooth_emotions(results, window_size=5)
            
            # Generate report
            report = detector.generate_report(smoothed_results)
            
            print("\n" + "="*60)
            print("EMOTION DETECTION REPORT")
            print("="*60)
            print(f"Overall Mood: {report['overall_mood'].upper()}")
            print(f"Average Confidence: {report['average_confidence']:.2%}")
            print(f"Total Frames Analyzed: {report['total_frames']}")
            print("\nEmotion Distribution:")
            for emotion, percentage in sorted(report['emotion_distribution'].items(), 
                                            key=lambda x: x[1], reverse=True):
                bar = "█" * int(percentage / 2)
                print(f"  {emotion:12} {bar:25} {percentage:5.1f}%")
            print("\n" + report['summary'])
            print("="*60)
        else:
            print(f"Video file not found: {video_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()