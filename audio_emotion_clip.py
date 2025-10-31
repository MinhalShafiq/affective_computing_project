import cv2
import torch
import numpy as np
import librosa
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import your existing EmotionCLIPDetector
from project import EmotionCLIPDetector, EmotionFrame


@dataclass
class AudioEmotionFrame:
    """Emotion detection result for audio segment"""
    segment_idx: int
    timestamp: float
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    audio_features: Optional[np.ndarray] = None


class AudioEmotionCLIPDetector(EmotionCLIPDetector):
    """
    Extended EmotionCLIP detector with audio input support
    Combines visual and audio features for more robust emotion detection
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__(model_path, device)
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.hop_length = 512
        self.n_mels = 128
        self.n_fft = 2048
        
        print(f"✓ AudioEmotionCLIP initialized with audio support")
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio track from video file
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for audio file
            
        Returns:
            Path to extracted audio file
        """
        import subprocess
        import tempfile
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(self.sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    
    def extract_audio_features(self, audio_path: str, segment_duration: float = 2.0) -> List[Tuple[np.ndarray, float]]:
        """
        Extract audio features from audio file
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of (features, timestamp) tuples
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Calculate segment length
        segment_samples = int(segment_duration * sr)
        
        features_list = []
        
        # Process audio in segments
        for i in range(0, len(y), segment_samples):
            segment = y[i:i + segment_samples]
            
            # Pad if necessary
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            
            # Extract features
            features = self._extract_segment_features(segment)
            timestamp = i / sr
            features_list.append((features, timestamp))
        
        return features_list
    
    def _extract_segment_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract features from a single audio segment
        
        Args:
            audio_segment: Audio samples
            
        Returns:
            Feature vector
        """
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio_segment,
            sr=self.sample_rate,
            n_mfcc=13
        )
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_segment,
            sr=self.sample_rate
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_segment,
            sr=self.sample_rate
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_segment)
        
        # Aggregate features
        features = np.concatenate([
            np.mean(mel_spec_db, axis=1),
            np.std(mel_spec_db, axis=1),
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate),
            np.mean(rms),
            np.std(rms)
        ])
        
        return features
    
    def detect_emotions_from_audio(self, audio_features: np.ndarray) -> Tuple[Dict[str, float], str, float]:
        """
        Detect emotions from audio features
        
        Args:
            audio_features: Extracted audio features
            
        Returns:
            (emotions dict, dominant emotion, confidence)
        """
        # Simple heuristic-based emotion detection from audio features
        # You would typically use a trained model here
        
        # Normalize features
        features_normalized = (audio_features - np.mean(audio_features)) / (np.std(audio_features) + 1e-8)
        
        # Extract key indicators
        energy = np.mean(audio_features[-2:])  # RMS mean & std
        pitch_variation = audio_features[self.n_mels * 2 + 13]  # Spectral centroid std
        
        # Simple rule-based classification (replace with trained model)
        emotions = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # High energy + high variation -> excited emotions
        if energy > 0.1 and pitch_variation > 0.1:
            emotions['happy'] = 0.3
            emotions['surprise'] = 0.25
            emotions['angry'] = 0.2
        # High energy + low variation -> angry
        elif energy > 0.1:
            emotions['angry'] = 0.4
            emotions['fear'] = 0.2
        # Low energy -> sad/neutral
        elif energy < 0.05:
            emotions['sad'] = 0.3
            emotions['neutral'] = 0.25
        else:
            emotions['neutral'] = 0.5
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        else:
            emotions = {k: 1.0 / len(emotions) for k in emotions.keys()}
        
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        return emotions, dominant_emotion, confidence
    
    def process_video_with_audio(
        self,
        video_path: str,
        fps: int = 8,
        use_face_detection: bool = True,
        audio_weight: float = 0.3
    ) -> List[EmotionFrame]:
        """
        Process video with both visual and audio emotion detection
        
        Args:
            video_path: Path to video file
            fps: Frames per second to analyze
            use_face_detection: Whether to detect faces
            audio_weight: Weight for audio emotions (0-1), visual weight = 1 - audio_weight
            
        Returns:
            List of EmotionFrame results with combined predictions
        """
        print("\n" + "="*60)
        print("PROCESSING WITH AUDIO + VIDEO")
        print("="*60)
        
        # Process video (visual)
        print("Extracting visual emotions...")
        visual_results = self.process_video(video_path, fps, use_face_detection)
        
        # Extract and process audio
        print("Extracting audio from video...")
        try:
            audio_path = self.extract_audio_from_video(video_path)
            print(f"✓ Audio extracted to: {audio_path}")
            
            print("Analyzing audio emotions...")
            audio_features_list = self.extract_audio_features(audio_path, segment_duration=1.0/fps)
            
            # Process audio features
            audio_results = []
            for idx, (features, timestamp) in enumerate(audio_features_list):
                emotions, dominant, confidence = self.detect_emotions_from_audio(features)
                audio_results.append(AudioEmotionFrame(
                    idx, timestamp, emotions, dominant, confidence, features
                ))
            
            print(f"✓ Processed {len(audio_results)} audio segments")
            
            # Combine visual and audio predictions
            print(f"Combining predictions (audio weight: {audio_weight:.1%})...")
            combined_results = []
            
            for i, visual_frame in enumerate(visual_results):
                # Find matching audio frame
                audio_frame = None
                if i < len(audio_results):
                    audio_frame = audio_results[i]
                
                if audio_frame:
                    # Weighted combination
                    combined_emotions = {}
                    for emotion in self.emotion_labels:
                        visual_score = visual_frame.emotions[emotion]
                        audio_score = audio_frame.emotions[emotion]
                        combined_emotions[emotion] = (
                            (1 - audio_weight) * visual_score + audio_weight * audio_score
                        )
                    
                    # Normalize
                    total = sum(combined_emotions.values())
                    combined_emotions = {k: v / total for k, v in combined_emotions.items()}
                    
                    dominant = max(combined_emotions, key=combined_emotions.get)
                    confidence = combined_emotions[dominant]
                    
                    combined_results.append(EmotionFrame(
                        visual_frame.frame_idx,
                        visual_frame.timestamp,
                        combined_emotions,
                        dominant,
                        confidence,
                        visual_frame.bbox
                    ))
                else:
                    # Use visual only if no audio
                    combined_results.append(visual_frame)
            
            print("✓ Combined audio + visual predictions")
            return combined_results
            
        except Exception as e:
            print(f"Warning: Audio processing failed: {e}")
            print("Falling back to visual-only processing")
            return visual_results


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize detector with audio support
    model_checkpoint = "EmotionCLIP/preprocessing/emotionclip_latest.pt"
    
    try:
        detector = AudioEmotionCLIPDetector(
            model_path=model_checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        video_path = sys.argv[1] if len(sys.argv) > 1 else "test_2.mp4"
        
        # Process with audio + video
        results = detector.process_video_with_audio(
            video_path,
            fps=8,
            use_face_detection=True,
            audio_weight=0.3  # 30% audio, 70% visual
        )
        
        # Apply smoothing
        smoothed = detector.smooth_emotions(results, window_size=9)
        
        # Generate report
        report = detector.generate_report(smoothed)
        
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        print(f"Overall Mood: {report['overall_mood'].upper()}")
        print(f"Average Confidence: {report['average_confidence']:.2%}")
        print("\nEmotion Distribution:")
        for emotion, percentage in sorted(report['emotion_distribution'].items(),
                                        key=lambda x: x[1], reverse=True):
            bar = "█" * int(percentage / 2)
            print(f"  {emotion:12} {bar:25} {percentage:5.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
