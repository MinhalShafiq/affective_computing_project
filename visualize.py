import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Dict
import sys
import torch

# Import your audio-enhanced detector
from audio_emotion_clip import AudioEmotionCLIPDetector
from project import EmotionFrame


class AudioEmotionVideoVisualizer:
    """Visualize video with audio-enhanced emotion detection"""
    
    def __init__(self, detector: AudioEmotionCLIPDetector):
        self.detector = detector
        self.results = []
        self.frames_data = []
        
        # Color map for emotions
        self.color_map = {
            'happy': (255, 215, 0),      # Gold
            'sad': (65, 105, 225),       # Royal Blue
            'angry': (255, 69, 0),       # Orange Red
            'fear': (153, 50, 204),      # Purple
            'surprise': (255, 105, 180), # Hot Pink
            'disgust': (139, 69, 19),    # Brown
            'contempt': (105, 105, 105), # Dim Gray
            'neutral': (144, 238, 144)   # Light Green
        }
    
    def visualize_video_with_audio(
        self,
        video_path: str,
        fps: int = 8,
        use_face_detection: bool = True,
        display_fps: int = 15,
        audio_weight: float = 0.3
    ):
        """
        Display video with audio-enhanced emotion visualization
        
        Args:
            video_path: Path to video file
            fps: Frames per second to analyze
            use_face_detection: Whether to detect faces
            display_fps: FPS for display playback
            audio_weight: Weight for audio in combined prediction
        """
        print(f"\nProcessing video with audio: {video_path}")
        
        # Process with audio + video
        self.results = self.detector.process_video_with_audio(
            video_path,
            fps=fps,
            use_face_detection=use_face_detection,
            audio_weight=audio_weight
        )
        
        if not self.results:
            print("Error: No frames were processed.")
            return
        
        # Apply smoothing
        print("Smoothing predictions...")
        self.results = self.detector.smooth_emotions(self.results, window_size=9)
        
        # Extract frames
        print("Extracting frames...")
        self.frames_data = self.detector.extract_frames(video_path, fps)
        
        # Generate report
        report = self.detector.generate_report(self.results)
        self._print_report(report, audio_weight)
        
        # Display
        self._display_opencv(display_fps, audio_weight)
    
    def _print_report(self, report: Dict, audio_weight: float):
        """Print summary report"""
        print("\n" + "="*70)
        print("AUDIO-ENHANCED EMOTION DETECTION REPORT")
        print("="*70)
        print(f"Mode: Combined (Audio: {audio_weight:.0%}, Visual: {1-audio_weight:.0%})")
        print(f"Overall Mood: {report['overall_mood'].upper()}")
        print(f"Average Confidence: {report['average_confidence']:.2%}")
        print(f"Total Frames Analyzed: {report['total_frames']}")
        print("\nEmotion Distribution:")
        for emotion, percentage in sorted(report['emotion_distribution'].items(),
                                        key=lambda x: x[1], reverse=True):
            bar = "█" * int(percentage / 2)
            print(f"  {emotion:12} {bar:25} {percentage:5.1f}%")
        print("="*70)
    
    def _create_emotion_chart(self, emotions: Dict[str, float], width: int = 400, height: int = 600):
        """Create emotion bar chart"""
        fig = plt.Figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        emotion_names = [e[0] for e in sorted_emotions]
        emotion_values = [e[1] for e in sorted_emotions]
        
        colors = [tuple(c/255 for c in self.color_map.get(e, (128, 128, 128))[::-1])
                 for e in emotion_names]
        
        y_pos = np.arange(len(emotion_names))
        ax.barh(y_pos, emotion_values, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([e.capitalize() for e in emotion_names], fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_title('Emotion Distribution\n(Audio + Visual)', fontsize=14, fontweight='bold', pad=20)
        
        for i, (emotion, value) in enumerate(zip(emotion_names, emotion_values)):
            ax.text(value + 0.02, i, f'{value:.2%}', va='center', fontsize=10, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        return img
    
    def _display_opencv(self, display_fps: int, audio_weight: float):
        """Display video with OpenCV"""
        print(f"\nDisplaying video (press 'q' to quit, SPACE to pause, 'r' to restart)...")
        
        frame_delay = int(1000 / display_fps)
        paused = False
        frame_idx = 0
        
        while True:
            if not paused:
                if frame_idx >= len(self.frames_data):
                    print("\nVideo finished. Press 'r' to restart or 'q' to quit.")
                    paused = True
                    frame_idx = len(self.frames_data) - 1
                
                if frame_idx < len(self.frames_data):
                    frame, timestamp = self.frames_data[frame_idx]
                    result = self.results[frame_idx]
                    display_frame = self._create_display_frame(frame, result, timestamp, frame_idx, audio_weight)
                    cv2.imshow('EmotionCLIP - Audio+Video Analysis', display_frame)
                    frame_idx += 1
            else:
                if frame_idx < len(self.frames_data):
                    frame, timestamp = self.frames_data[frame_idx]
                    result = self.results[frame_idx]
                    display_frame = self._create_display_frame(frame, result, timestamp, frame_idx, audio_weight)
                    cv2.imshow('EmotionCLIP - Audio+Video Analysis', display_frame)
            
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):
                frame_idx = 0
                paused = False
                print("Restarting...")
        
        cv2.destroyAllWindows()
    
    def _create_display_frame(self, frame: np.ndarray, result: EmotionFrame,
                             timestamp: float, frame_idx: int, audio_weight: float) -> np.ndarray:
        """Create combined display frame"""
        video_height = 720
        aspect_ratio = frame.shape[1] / frame.shape[0]
        video_width = int(video_height * aspect_ratio)
        frame_resized = cv2.resize(frame, (video_width, video_height))
        
        chart_width = 500
        emotion_chart = self._create_emotion_chart(result.emotions, chart_width, video_height)
        emotion_chart = cv2.resize(emotion_chart, (chart_width, video_height))
        
        combined = np.hstack([frame_resized, emotion_chart])
        
        overlay = frame_resized.copy()
        cv2.rectangle(overlay, (10, 10), (video_width - 10, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame_resized, 0.5, 0, frame_resized)
        combined[:, :video_width] = frame_resized
        
        dominant_emotion = result.dominant_emotion
        color_bgr = self.color_map.get(dominant_emotion, (255, 255, 255))
        
        # Emotion name
        cv2.putText(combined, dominant_emotion.upper(), (20, 60),
                   cv2.FONT_HERSHEY_DUPLEX, 1.8, color_bgr, 3)
        
        # Confidence
        cv2.putText(combined, f"Confidence: {result.confidence:.1%}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mode indicator
        mode_text = f"Mode: Audio({audio_weight:.0%}) + Visual({1-audio_weight:.0%})"
        cv2.putText(combined, mode_text, (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        # Timestamp
        cv2.putText(combined, f"Time: {timestamp:.2f}s", (20, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(combined, f"Frame: {frame_idx + 1}/{len(self.frames_data)}", (20, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Face bbox
        if result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(combined, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(combined, "Face", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        
        # Progress bar
        bar_height = 30
        progress_bar = np.zeros((bar_height, combined.shape[1], 3), dtype=np.uint8)
        progress_width = int((frame_idx / len(self.frames_data)) * combined.shape[1])
        cv2.rectangle(progress_bar, (0, 0), (progress_width, bar_height), (0, 255, 0), -1)
        cv2.rectangle(progress_bar, (0, 0), (combined.shape[1], bar_height), (255, 255, 255), 2)
        
        combined = np.vstack([combined, progress_bar])
        
        return combined


if __name__ == "__main__":
    import torch
    
    model_checkpoint = "EmotionCLIP/preprocessing/emotionclip_latest.pt"
    
    try:
        # Initialize audio-enhanced detector
        detector = AudioEmotionCLIPDetector(
            model_path=model_checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create visualizer
        visualizer = AudioEmotionVideoVisualizer(detector)
        
        # Get video path
        video_path = sys.argv[1] if len(sys.argv) > 1 else "test_2.mp4"
        
        # Get audio weight (optional)
        audio_weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
        
        # Process and visualize
        visualizer.visualize_video_with_audio(
            video_path,
            fps=8,
            use_face_detection=True,
            display_fps=15,
            audio_weight=audio_weight  # 0.0 = visual only, 1.0 = audio only
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()