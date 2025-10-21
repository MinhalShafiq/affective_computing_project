import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Dict
import time

from project import EmotionCLIPDetector, EmotionFrame

class EmotionVideoVisualizer:
    """Visualize video with real-time emotion detection using OpenCV"""
    
    def __init__(self, detector: EmotionCLIPDetector):
        """
        Initialize visualizer
        
        Args:
            detector: EmotionCLIPDetector instance
        """
        self.detector = detector
        self.results = []
        self.frames_data = []
        
        # Color map for emotions
        self.color_map = {
            'happy': (255, 215, 0),      # Gold
            'sad': (65, 105, 225),       # Royal Blue
            'angry': (255, 69, 0),       # Orange Red
            'calm': (144, 238, 144),     # Light Green
            'fearful': (153, 50, 204),   # Purple
            'surprised': (255, 105, 180), # Hot Pink
            'disgusted': (139, 69, 19),  # Brown
            'neutral': (128, 128, 128)   # Gray
        }
        
    def visualize_video(self, video_path: str, fps: int = 8, use_face_detection: bool = True, 
                       display_fps: int = 15):
        """
        Display video with emotion visualization using OpenCV
        
        Args:
            video_path: Path to video file
            fps: Frames per second to analyze
            use_face_detection: Whether to detect and crop faces
            display_fps: FPS for display playback
        """
        print(f"\nProcessing video: {video_path}")
        
        # Process video
        self.results = self.detector.process_video(video_path, fps, use_face_detection)
        
        # Smooth results
        print("Smoothing predictions...")
        self.results = self.detector.smooth_emotions(self.results, window_size=5)
        
        # Extract frames
        self.frames_data = self.detector.extract_frames(video_path, fps)
        
        # Generate report
        report = self.detector.generate_report(self.results)
        self._print_report(report)
        
        # Display video with emotions
        self._display_opencv(display_fps)
    
    def _print_report(self, report: Dict):
        """Print summary report"""
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
    
    def _create_emotion_chart(self, emotions: Dict[str, float], width: int = 400, height: int = 600):
        """
        Create emotion bar chart as numpy array
        
        Args:
            emotions: Dictionary of emotion probabilities
            width: Chart width
            height: Chart height
            
        Returns:
            Numpy array of chart image
        """
        # Create matplotlib figure
        fig = plt.Figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        # Sort emotions by probability
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        emotion_names = [e[0] for e in sorted_emotions]
        emotion_values = [e[1] for e in sorted_emotions]
        
        # Create colors (convert from BGR to RGB for matplotlib)
        colors = [tuple(c/255 for c in self.color_map.get(e, (128, 128, 128))[::-1]) 
                 for e in emotion_names]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(emotion_names))
        ax.barh(y_pos, emotion_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels([e.capitalize() for e in emotion_names], fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_title('Emotion Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (emotion, value) in enumerate(zip(emotion_names, emotion_values)):
            ax.text(value + 0.02, i, f'{value:.2%}', va='center', fontsize=10, fontweight='bold')
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        
        # Convert to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        return img
    
    def _display_opencv(self, display_fps: int = 15):
        """
        Display video with OpenCV window
        
        Args:
            display_fps: Display frame rate
        """
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
                    
                    # Create combined display
                    display_frame = self._create_display_frame(frame, result, timestamp, frame_idx)
                    
                    # Show frame
                    cv2.imshow('EmotionCLIP - Video Analysis', display_frame)
                    
                    frame_idx += 1
            else:
                # Keep displaying current frame when paused
                if frame_idx < len(self.frames_data):
                    frame, timestamp = self.frames_data[frame_idx]
                    result = self.results[frame_idx]
                    display_frame = self._create_display_frame(frame, result, timestamp, frame_idx)
                    cv2.imshow('EmotionCLIP - Video Analysis', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):  # Restart
                frame_idx = 0
                paused = False
                print("Restarting video...")
        
        cv2.destroyAllWindows()
    
    def _create_display_frame(self, frame: np.ndarray, result: EmotionFrame, 
                             timestamp: float, frame_idx: int) -> np.ndarray:
        """
        Create combined display frame with video and emotions
        
        Args:
            frame: Video frame
            result: Emotion detection result
            timestamp: Current timestamp
            frame_idx: Current frame index
            
        Returns:
            Combined display frame
        """
        # Resize video frame
        video_height = 720
        aspect_ratio = frame.shape[1] / frame.shape[0]
        video_width = int(video_height * aspect_ratio)
        frame_resized = cv2.resize(frame, (video_width, video_height))
        
        # Create emotion chart
        chart_width = 500
        emotion_chart = self._create_emotion_chart(result.emotions, chart_width, video_height)
        emotion_chart = cv2.resize(emotion_chart, (chart_width, video_height))
        
        # Combine video and chart horizontally
        combined = np.hstack([frame_resized, emotion_chart])
        
        # Add info overlay on video section
        overlay = frame_resized.copy()
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (video_width - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame_resized, 0.5, 0, frame_resized)
        
        # Update combined frame with overlay
        combined[:, :video_width] = frame_resized
        
        # Add text information
        dominant_emotion = result.dominant_emotion
        color_bgr = self.color_map.get(dominant_emotion, (255, 255, 255))
        
        # Emotion name (large)
        cv2.putText(combined, dominant_emotion.upper(), (20, 60), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.8, color_bgr, 3)
        
        # Confidence
        cv2.putText(combined, f"Confidence: {result.confidence:.1%}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Timestamp
        cv2.putText(combined, f"Time: {timestamp:.2f}s", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(combined, f"Frame: {frame_idx + 1}/{len(self.frames_data)}", (20, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw bounding box if face was detected
        if result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(combined, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(combined, "Face", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        
        # Add progress bar at bottom
        bar_height = 30
        progress_bar = np.zeros((bar_height, combined.shape[1], 3), dtype=np.uint8)
        progress_width = int((frame_idx / len(self.frames_data)) * combined.shape[1])
        cv2.rectangle(progress_bar, (0, 0), (progress_width, bar_height), (0, 255, 0), -1)
        cv2.rectangle(progress_bar, (0, 0), (combined.shape[1], bar_height), (255, 255, 255), 2)
        
        # Combine with progress bar
        combined = np.vstack([combined, progress_bar])
        
        return combined


# Standalone script version
if __name__ == "__main__":
    import torch
    import sys
    from project import EmotionCLIPDetector
    
    # Initialize detector
    model_checkpoint = "EmotionCLIP/preprocessing/emotionclip_latest.pt"
    
    try:
        detector = EmotionCLIPDetector(
            model_path=model_checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create visualizer
        visualizer = EmotionVideoVisualizer(detector)
        
        # Get video path from command line or use default
        video_path = sys.argv[1] if len(sys.argv) > 1 else "test_1.mp4"
        
        # Process and visualize video
        visualizer.visualize_video(
            video_path, 
            fps=8,  # Analysis FPS
            use_face_detection=True,
            display_fps=15  # Display FPS
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()