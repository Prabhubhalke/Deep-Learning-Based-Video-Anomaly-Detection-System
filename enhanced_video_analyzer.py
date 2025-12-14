import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

# Import GPT-5 analyzer for explanations
try:
    from github_gpt5_analyzer import GitHubModelsAnalyzer
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    print("Warning: GitHub GPT-5 Analyzer not available.")

class EnhancedVideoAnalyzer:
    """
    Enhanced video analyzer using YOLO for fast object detection and analysis.
    Optimized for speed with minimal processing overhead.
    """
    
    def __init__(self, 
                 yolo_model: str = "yolov8n.pt",
                 use_pose: bool = False,
                 caption_model: str = None,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.45):
        """
        Initialize the enhanced video analyzer.
        
        Args:
            yolo_model: YOLO model to use (yolov8n.pt for speed, yolov8m.pt for accuracy)
            use_pose: Whether to use pose estimation
            caption_model: Caption model to use (None for no captions)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        print(f"Initializing Enhanced Video Analyzer with {yolo_model}...")
        
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO class names
        self.coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Activity indicators for anomaly detection
        self.activity_indicators = {
            'physical_abuse': ['person', 'hit', 'push', 'fight'],
            'violence': ['person', 'weapon', 'gun', 'knife', 'fight'],
            'theft': ['person', 'bag', 'backpack', 'laptop', 'phone', 'steal'],
            'vandalism': ['break', 'damage', 'spray', 'paint'],
            'suspicious_behavior': ['loiter', 'watch', 'follow', 'hide']
        }
    
    def process_video_complete(self, 
                              video_path: str, 
                              frames_per_second: float = 0.5,
                              anomaly_threshold: float = 0.5) -> Dict:
        """
        Process video completely with optimized analysis for speed.
        
        Args:
            video_path: Path to the video file
            frames_per_second: Number of frames to analyze per second (0.5 = 1 frame every 2 seconds for faster analysis)
            anomaly_threshold: Threshold for anomaly detection
            
        Returns:
            Dictionary with complete analysis results
        """
        print(f"[FAST] Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"[FAST] Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Calculate frame interval for speed optimization
        # For faster analysis, we increase the frames per second
        frame_interval = max(1, int(fps / frames_per_second))
        
        # Limit total frames processed for very long videos to maintain speed
        max_frames_to_process = 50  # Maximum frames to process regardless of video length
        frames_processed = 0
        
        # Process frames with minimal overhead
        events = []
        frame_count = 0
        processed_frames = 0
        anomaly_score = 0.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame at specified interval for speed
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps if fps > 0 else frame_count
                event = self._analyze_frame_fast(frame, timestamp, frame_count)
                if event:
                    events.append(event)
                    # Accumulate anomaly score
                    if event.get('anomaly_flag', False):
                        anomaly_score += 1.0
                processed_frames += 1
                frames_processed += 1
                
                # Progress indicator (every 5 processed frames)
                if processed_frames % 5 == 0:
                    print(f"[FAST] Processed {processed_frames} frames...")
                
                # Limit total frames for speed
                if frames_processed >= max_frames_to_process:
                    print(f"[FAST] Reached maximum frame limit ({max_frames_to_process}) for speed optimization")
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"[FAST] Finished processing {processed_frames} frames")
        
        # Calculate overall anomaly score
        anomaly_count = sum(1 for event in events if event.get('anomaly_flag', False))
        overall_anomaly_score = anomaly_score / max(1, len(events)) if events else 0.0
        
        # Generate fast timeline report
        timeline_report = self._generate_fast_timeline_report(events, video_path)
        
        # Save report to file
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        report_filename = f"data/{video_name}_timeline_report.txt"
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(timeline_report)
            print(f"[FAST] Timeline report saved to: {report_filename}")
        except Exception as e:
            print(f"[FAST] Failed to save timeline report: {e}")
            report_filename = None
        
        # Generate GPT explanation for the report
        gpt_explanation = self.generate_gpt_explanation(timeline_report, video_path)
        
        return {
            'video_path': video_path,
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'frame_width': width,
            'frame_height': height,
            'events': events,
            'events_count': len(events),
            'anomaly_count': anomaly_count,
            'anomaly_score': overall_anomaly_score,
            'report': timeline_report,
            'gpt_explanation': gpt_explanation,
            'output_path': report_filename
        }
    
    def _analyze_frame_fast(self, frame: np.ndarray, timestamp: float, frame_number: int) -> Dict:
        """
        Fast frame analysis with YOLO detection optimized for speed.
        
        Args:
            frame: Video frame as numpy array
            timestamp: Timestamp in seconds
            frame_number: Frame number
            
        Returns:
            Dictionary with frame analysis
        """
        # Run YOLO detection with optimized parameters for speed
        results = self.yolo_model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Extract detections quickly
        detections = []
        person_count = 0
        objects = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if cls_id < len(self.coco_names):
                        class_name = self.coco_names[cls_id]
                        
                        # Count persons
                        if class_name == 'person':
                            person_count += 1
                        
                        # Collect all objects
                        objects.append(class_name)
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        # Quick anomaly detection based on object combinations
        anomaly_flag = False
        anomaly_type = None
        
        # Simple heuristic for anomaly detection
        if person_count >= 2:  # Person-person interaction (abuse detection)
            anomaly_flag = True
            anomaly_type = "physical_abuse"
        elif person_count > 5:  # Crowded scene
            anomaly_flag = True
            anomaly_type = "crowded_scene"
        elif any(obj in ['knife', 'gun', 'baseball bat'] for obj in objects):  # Weapon detection
            anomaly_flag = True
            anomaly_type = "weapon_detected"
        elif person_count > 0 and any(obj in ['backpack', 'laptop', 'phone'] for obj in objects):  # Potential theft
            anomaly_flag = True
            anomaly_type = "suspicious_activity"
        
        # Format timestamp
        timestamp_formatted = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}.{int((timestamp % 1) * 1000):03d}"
        
        return {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'timestamp_formatted': timestamp_formatted,
            'person_count': person_count,
            'objects': objects[:5],  # Limit to first 5 objects for speed
            'detections': detections[:3],  # Limit to first 3 detections for speed
            'anomaly_flag': anomaly_flag,
            'anomaly_type': anomaly_type
        }
    
    def _generate_fast_timeline_report(self, events: List[Dict], video_path: str) -> str:
        """
        Generate a fast timeline report from events.
        
        Args:
            events: List of event dictionaries
            video_path: Path to the video file
            
        Returns:
            Formatted timeline report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FAST VIDEO ANALYSIS REPORT (WITH YOLO)")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Video File: {os.path.basename(video_path)}")
        report_lines.append(f"Analysis Date: {datetime.now()}")
        report_lines.append(f"Total Events Analyzed: {len(events)}")
        report_lines.append(f"Anomalies Detected: {sum(1 for e in events if e.get('anomaly_flag', False))}")
        report_lines.append("")
        report_lines.append("TIMELINE OF EVENTS:")
        report_lines.append("-" * 40)
        
        for i, event in enumerate(events[:20]):
            event_line = f"[{event['timestamp_formatted']}] "
            if event.get('anomaly_flag'):
                event_line += f"⚠ NOT NORMAL ({event.get('anomaly_type', 'unknown')}) - "
            else:
                event_line += "✓ NORMAL - "
            
            # Add person count and objects
            if event['person_count'] > 0:
                event_line += f"{event['person_count']} person(s)"
            if event['objects']:
                objects_str = ", ".join(event['objects'][:3])  # Limit to first 3 objects
                event_line += f", Objects: {objects_str}"
            
            report_lines.append(event_line)
        
        # Add summary
        report_lines.extend([
            "",
            "ANALYSIS SUMMARY:",
            "-" * 20,
            f"This is a fast analysis report generated using YOLO object detection.",
            f"The system processed {len(events)} key frames for rapid anomaly detection.",
            f"Any anomalies detected are flagged with detailed timestamps above."
        ])
        
        return "\n".join(report_lines)
    
    def generate_gpt_explanation(self, report_text: str, video_path: str) -> str:
        """
        Generate a GPT-5 explanation for the video analysis report.
        
        Args:
            report_text: The analysis report text
            video_path: Path to the video file
            
        Returns:
            Plain English explanation of the analysis
        """
        if not GPT_AVAILABLE:
            return "GPT explanation not available - analyzer not installed."
        
        try:
            gpt = GitHubModelsAnalyzer()
            
            # Prepare the prompt with the report
            prompt = f"""
Explain this video analysis report in simple human language:

{report_text}

Explain:
- What happened
- Whether abuse was detected
- Why the video is classified as normal or not normal
"""
            
            print(f"[DEBUG] Sending prompt to GPT-5 (length: {len(prompt)} chars)")
            
            explanation = gpt._call_gpt(prompt, max_tokens=500)
            
            return explanation
        except Exception as e:
            print(f"[DEBUG] GPT explanation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Failed to generate GPT explanation: {e}"

    def _frames_to_text(self, events, max_frames=6):
        """
        Convert frame events into natural language descriptions for GPT
        """
        lines = []
        for e in events[-max_frames:]:
            lines.append(
                f"At {e['timestamp_formatted']}, "
                f"{e['person_count']} person(s) detected, "
                f"objects: {', '.join(e['objects']) if e['objects'] else 'none'}, "
                f"abnormal: {'yes' if e['anomaly_flag'] else 'no'}."
            )
        return "\n".join(lines)

    def generate_gpt_frame_explanation(self, events, video_path):
        """
        Use GPT to explain what is happening in the video
        based on frame-by-frame analysis
        
        This method now handles the case where the deep learning model
        classifies a video as "Not Normal" but YOLO doesn't detect anomalies
        """
        if not events:
            return "No frame data available for explanation."

        # Check if we have any anomalies detected by YOLO
        yolo_anomalies = sum(1 for e in events if e.get('anomaly_flag', False))
        
        # If no YOLO anomalies but we're still here, it means the deep learning
        # model classified this as "Not Normal", so we need to provide context
        if yolo_anomalies == 0:
            frame_text = self._frames_to_text(events)
            prompt = f"""
You are analyzing a surveillance video that has been flagged as containing potential abuse.
However, the frame analysis shows mostly normal activity with no clear anomalies detected.

Frame observations:
{frame_text}

Despite the lack of clear anomalies in this frame analysis, the video has been classified 
as containing potential abuse by another detection system. Please analyze:

1. What is happening in the video based on the frame data
2. Why this might still be considered a potential abuse case despite no clear anomalies
3. Any subtle signs of concern in the scene
4. Recommendations for further investigation if needed

Focus on human behavior and interactions.
Do not guess beyond the given data.
"""
        else:
            # Normal case with YOLO-detected anomalies
            frame_text = self._frames_to_text(events)
            prompt = f"""
You are analyzing a surveillance video using frame-by-frame observations.

Frame observations:
{frame_text}

Explain clearly what is happening in the video.
Focus on human behavior and interactions.
Do not guess beyond the given data.
"""

        try:
            gpt = GitHubModelsAnalyzer()
            return gpt._call_gpt(prompt, max_tokens=350)
        except Exception as e:
            return f"GPT explanation could not be generated: {e}"

# Example usage
if __name__ == "__main__":
    # Test the analyzer with a sample video
    print("Testing Enhanced Video Analyzer...")
    try:
        analyzer = EnhancedVideoAnalyzer(yolo_model="yolov8n.pt")  # Use nano model for speed
        print("✓ Analyzer initialized successfully")
        print("Ready for fast video analysis with YOLO!")
    except Exception as e:
        print(f"✗ Error initializing analyzer: {e}")