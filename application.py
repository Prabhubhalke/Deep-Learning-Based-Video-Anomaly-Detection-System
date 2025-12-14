import sys
import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Dict
from tensorflow.keras.models import load_model
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, QFrame, QGraphicsDropShadowEffect, QTextEdit, QScrollArea, QSplitter, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtGui import QFont, QCursor, QPixmap, QPainter, QColor, QPalette, QBrush, QTextCursor
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

# Import video analyzer
try:
    from video_analyzer import VideoAnalyzer
    VIDEO_ANALYZER_AVAILABLE = True
except ImportError:
    VIDEO_ANALYZER_AVAILABLE = False
    print("Warning: Video analyzer not available.")

# Import GPT Video Analyzer
try:
    from github_gpt5_analyzer import GitHubModelsAnalyzer
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: GitHub Models Analyzer not available. Install azure-ai-inference package and set GITHUB_TOKEN.")

# Import Video Explainer for GPT-based video explanations
try:
    from video_explainer import VideoExplainer
    VIDEO_EXPLAINER_AVAILABLE = True
except ImportError:
    VIDEO_EXPLAINER_AVAILABLE = False
    print("Warning: Video explainer not available.")

# Import enhanced video analyzer with YOLO (optional)
try:
    from enhanced_video_analyzer import EnhancedVideoAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYZER_AVAILABLE = False
    print("Warning: Enhanced video analyzer (YOLO) not available. Install ultralytics package.")

# Import optimized video analyzer (faster alternative)
try:
    from optimized_video_analyzer import OptimizedVideoAnalyzer
    OPTIMIZED_ANALYZER_AVAILABLE = True
except ImportError:
    OPTIMIZED_ANALYZER_AVAILABLE = False
    print("Warning: Optimized video analyzer not available.")

# Paths
model_path = "data/prepared_data/anomaly_detection_model.h5"  # Update the path if needed
label_encoder_path = "data/prepared_data/label_encoder.npy"

# Load the model
print("Loading the model...")
model = load_model(model_path)

# Load the label encoder
print("Loading label encoder...")
label_classes = np.load(label_encoder_path, allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Dynamically find the label for "Normal"
normal_class = None
for idx, label in enumerate(label_encoder.classes_):
    if label.lower() == 'normal':  # Case-insensitive match
        normal_class = idx
        break

if normal_class is None:
    print("Error: 'Normal' label not found in label encoder.")
    sys.exit()

print(f"'Normal' class label is: {normal_class}")

# EfficientNetB0 model as feature extractor
# Explicitly set channels_last and 3-channel RGB input to match ImageNet weights
K.set_image_data_format('channels_last')
try:
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
except Exception as e:
    print(f"Warning: Failed to load EfficientNetB0 ImageNet weights ({e}). Falling back to uninitialized weights.")
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor_model = Model(inputs=base_model.input, outputs=x)


class VideoProcessor(QThread):
    """
    A QThread to process the video and classify it as 'Normal' or 'Not Normal'.
    """
    progress_updated = Signal(float)  # Signal to update progress bar
    classification_done = Signal(str)  # Signal to display the final result
    analysis_done = Signal(dict)  # Signal for video analysis results
    realtime_update = Signal(str)  # Signal for real-time GPT explanations

    def __init__(self, video_path, sensitivity_mode, analysis_mode="balanced", enable_analysis=True, parent=None):
        super(VideoProcessor, self).__init__(parent)
        self.video_path = video_path
        self.sensitivity_mode = sensitivity_mode  # Sensitivity mode passed when the thread is created
        self.analysis_mode = analysis_mode  # Analysis mode: quick, balanced, or detailed
        # Enable analysis if either analyzer is available
        self.enable_analysis = enable_analysis and (VIDEO_ANALYZER_AVAILABLE or ENHANCED_ANALYZER_AVAILABLE)
        print(f"VideoProcessor initialized with sensitivity mode: {self.sensitivity_mode}, Analysis mode: {self.analysis_mode}, Analysis: {self.enable_analysis}")  # Debugging statement

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        if fps is None or fps <= 0:
            fps = 24  # Fallback FPS to avoid divide-by-zero
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames is None or total_frames <= 0:
            total_frames = 1
        predictions = []

        print(f"Processing video: {self.video_path}, FPS: {fps}, Total Frames: {total_frames}")

        # Initialize GPT analyzer for real-time explanations if available
        gpt_analyzer = None
        if GEMINI_AVAILABLE:
            try:
                from github_gpt5_analyzer import GitHubModelsAnalyzer
                gpt_analyzer = GitHubModelsAnalyzer()
                print("✓ GPT Analyzer initialized for real-time explanations")
            except ValueError as e:
                if "GITHUB_TOKEN not found" in str(e):
                    print("⚠ GITHUB_TOKEN not set. Please set your GitHub token to enable real-time GPT analysis.")
                    print("   Run 'python set_github_token.py' or set GITHUB_TOKEN environment variable.")
                else:
                    print(f"⚠ Could not initialize GPT Analyzer: {e}")
            except Exception as e:
                print(f"⚠ Could not initialize GPT Analyzer: {e}")
                print("   Make sure your GITHUB_TOKEN is valid and has access to GitHub Models.")

        frame_events = []  # Store frame events for GPT analysis
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frames based on analysis mode for speed optimization
            if self.analysis_mode == "ultra_fast":
                # In ultra_fast mode, process 1 frame every 2 seconds for faster analysis
                step = int(fps * 2) if fps > 0 else 2
            elif self.analysis_mode == "quick":
                # In quick mode, process 1 frame every 1 second
                step = int(fps) if fps > 0 else 1
            else:
                # In balanced mode, process 2 frames per second
                step = int(fps / 2) if fps > 0 else 1
                
            # Ensure we don't divide by zero
            step = max(1, step)
            
            if frame_count % step == 0:
                # Resize and preprocess the frame for the model
                # Convert BGR (OpenCV) to RGB as expected by EfficientNet
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frame_preprocessed = np.expand_dims(frame_resized, axis=0)
                frame_preprocessed = preprocess_input(frame_preprocessed)

                # Extract features using the feature extractor model
                features = feature_extractor_model.predict(frame_preprocessed)

                # Make prediction using the model
                y_pred = model.predict(features)
                y_pred_class = np.argmax(y_pred, axis=1)[0]
                confidence = float(np.max(y_pred))

                # Classify as Normal (0) or Not Normal (1)
                is_normal = y_pred_class == normal_class
                if is_normal:
                    predictions.append(0)  # Normal
                else:
                    predictions.append(1)  # Not Normal

                # Create frame event for potential GPT analysis with enhanced information
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                # Add basic frame analysis for better GPT context
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(frame_gray))
                motion_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
                
                frame_event = {
                    "frame_number": frame_count,
                    "timestamp": timestamp,
                    "timestamp_formatted": self.format_timestamp(timestamp),
                    "is_normal": is_normal,
                    "confidence": confidence,
                    "prediction": "Normal" if is_normal else "Anomaly",
                    "brightness": brightness,
                    "motion_level": motion_level,
                    "description": f"{motion_level} motion activity detected" if not is_normal else "normal activity"
                }
                frame_events.append(frame_event)

                print(f"Frame {frame_count}: Prediction = {'Normal' if is_normal else 'Not Normal'} (Confidence: {confidence:.2f})")

                # Enable real-time GPT explanations for all modes to provide live video analysis
                # Send real-time update with GPT explanation every 10 frames
                if len(frame_events) % 10 == 0 and gpt_analyzer and len(frame_events) >= 5:
                    try:
                        # Generate concise explanation for recent frames
                        recent_events = frame_events[-5:]  # Last 5 frames
                        explanation_prompt = self._create_explanation_prompt(recent_events, self.video_path)
                        
                        # Generate concise explanation
                        concise_explanation = gpt_analyzer.generate_concise_event_description(
                            recent_events, self.video_path
                        )
                        
                        if concise_explanation and len(concise_explanation.strip()) > 10:
                            realtime_msg = f"[REAL-TIME UPDATE] {concise_explanation}"
                            self.realtime_update.emit(realtime_msg)
                            print(f"✓ Real-time explanation generated for frames {recent_events[0]['frame_number']}-{recent_events[-1]['frame_number']}")
                    except Exception as e:
                        print(f"⚠ Real-time explanation failed: {e}")

            frame_count += 1

            # Update progress
            progress = (min(frame_count, total_frames) / total_frames) * 100
            self.progress_updated.emit(progress)  # Emit signal for progress update

        cap.release()

        # -------------------------------
        # YOLO-BASED ABUSE PRE-CHECK
        # -------------------------------
        yolo_abuse_votes = 0

        if ENHANCED_ANALYZER_AVAILABLE:
            try:
                quick_yolo = EnhancedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.3,
                    iou_threshold=0.45
                )

                yolo_result = quick_yolo.process_video_complete(
                    video_path=self.video_path,
                    frames_per_second=0.5
                )

                for event in yolo_result.get("events", []):
                    if event.get("person_count", 0) >= 2:
                        yolo_abuse_votes += 1

            except Exception as e:
                print(f"YOLO pre-check failed: {e}")

        # Adjust voting logic based on sensitivity mode
        if len(predictions) == 0:
            video_class = "Normal"
        else:
            weighted_predictions = np.array(predictions)
            if self.sensitivity_mode == "Low":
                not_normal_weight = 1
                video_class = "Not Normal" if np.sum(weighted_predictions * not_normal_weight) / (len(weighted_predictions) * not_normal_weight) > 0.5 else "Normal"
            else:
                not_normal_weight = 2
                video_class = "Not Normal" if np.sum(weighted_predictions * not_normal_weight) / (len(weighted_predictions) * not_normal_weight) > 0.01 else "Normal"

        # -------------------------------
        # FORCE "NOT NORMAL" IF YOLO SEES INTERACTION
        # -------------------------------
        if yolo_abuse_votes >= 3:
            video_class = "Not Normal"
            print("⚠ YOLO interaction evidence detected → Forcing Not Normal")

        self.classification_done.emit(video_class)  # Emit signal for classification result
        
        # Perform detailed analysis if enabled and anomaly detected
        if self.enable_analysis and video_class == "Not Normal":
            try:
                print("Starting detailed video analysis with enhanced YOLO detection...")
                
                # Use Enhanced Video Analyzer with YOLO for fast and accurate analysis
                if ENHANCED_ANALYZER_AVAILABLE:
                    print(f"Using Enhanced Video Analyzer with YOLO in {self.analysis_mode} mode for fast processing...")
                    
                    # Set parameters based on analysis mode for speed optimization
                    if self.analysis_mode == "ultra_fast":
                        yolo_model = "yolov8n.pt"  # Nano model for maximum speed
                        conf_threshold = 0.5  # Higher threshold to reduce false positives
                        frames_per_second = 0.1  # Process even fewer frames
                        max_frames = 5  # Limit to only 5 frames
                    elif self.analysis_mode == "quick":
                        yolo_model = "yolov8n.pt"  # Nano model for speed
                        conf_threshold = 0.4
                        frames_per_second = 0.3
                        max_frames = 10
                    else:  # balanced mode
                        yolo_model = "yolov8n.pt"  # Nano model for speed
                        conf_threshold = 0.3
                        frames_per_second = 0.5
                        max_frames = 15
                    
                    enhanced_analyzer = EnhancedVideoAnalyzer(
                        yolo_model=yolo_model,
                        use_pose=False,
                        caption_model=None,  # Can enable 'blip' for AI descriptions
                        conf_threshold=conf_threshold,
                        iou_threshold=0.45
                    )
                    
                    # Process video with enhanced analyzer
                    print(f"Processing video with enhanced analyzer: {self.video_path}")
                    result = enhanced_analyzer.process_video_complete(
                        video_path=self.video_path,
                        frames_per_second=frames_per_second,
                        anomaly_threshold=0.5
                    )
                    
                    # Generate report text from timeline report
                    report_text = result.get('report', '')
                    
                    # If report is empty, try to generate it manually
                    if not report_text or len(report_text.strip()) < 50:
                        print("Warning: Report text is empty or too short, generating from events...")
                        events = result.get('events', [])
                        if events and len(events) > 0:
                            print(f"Generating report from {len(events)} events...")
                            try:
                                report_text = enhanced_analyzer.generate_timeline_report(events, self.video_path)
                                print(f"Generated report length: {len(report_text)} characters")
                            except Exception as e:
                                print(f"Error generating report: {e}")
                                report_text = ""
                    
                    # If still empty, create a basic summary
                    if not report_text or len(report_text.strip()) < 50:
                        print("Creating basic summary as fallback...")
                        events = result.get('events', [])
                        report_text = self._create_basic_summary(events, result, self.video_path)
                        print(f"Basic summary length: {len(report_text)} characters")
                    
                    # Final fallback - ensure we always have something
                    if not report_text or len(report_text.strip()) < 10:
                        report_text = (
                            f"VIDEO ANALYSIS COMPLETED\n"
                            f"{'='*80}\n\n"
                            f"Source Video: {self.video_path}\n"
                            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Events Analyzed: {result.get('events_count', 0)}\n"
                            f"Anomalies Detected: {result.get('anomaly_count', 0)}\n"
                            f"\nFull detailed report saved to:\n{result.get('output_path', 'N/A')}\n"
                            f"\nPlease check the report file for complete analysis details."
                        )
                    
                    # ------------------------------------
                    # GPT FRAME-BASED VIDEO EXPLANATION
                    # ------------------------------------
                    gpt_explanation = ""

                    try:
                        gpt_explanation = enhanced_analyzer.generate_gpt_frame_explanation(
                            events=result.get("events", []),
                            video_path=self.video_path
                        )
                    except Exception as e:
                        print(f"GPT frame explanation failed: {e}")
                        gpt_explanation = "GPT explanation could not be generated."
                    
                    # Merge GPT explanation into final report
                    final_report = (
                        report_text
                        + "\n\n"
                        + "=" * 80
                        + "\nGPT FRAME-BASED VIDEO EXPLANATION\n"
                        + "=" * 80
                        + "\n"
                        + gpt_explanation
                    )
                    
                    # Create analysis result in expected format
                    analysis_result = {
                        'timestamp': datetime.now().isoformat(),
                        'video_path': self.video_path,
                        'description': final_report,
                        'report': final_report,  # Also include as 'report' key
                        'events_count': result.get('events_count', 0),
                        'anomaly_count': result.get('anomaly_count', 0),
                        'output_path': result.get('output_path', ''),
                        'events': result.get('events', [])  # Include events for anomaly type extraction
                    }
                    
                    print(f"Enhanced analysis completed: {result['events_count']} events, {result['anomaly_count']} anomalies")
                    print(f"Report text length: {len(report_text)} characters")
                    print(f"Report saved to: {result.get('output_path', 'N/A')}")
                # Fallback to basic video analyzer if enhanced one is not available
                elif VIDEO_ANALYZER_AVAILABLE:
                    print("Using basic Video Analyzer for processing...")
                    
                    # Create a fast, minimal analysis
                    video_analyzer = VideoAnalyzer()
                    
                    # Generate a simple, fast report
                    analysis_result = self._generate_fast_analysis_report(self.video_path)
                    
                    print("Basic analysis completed")
                else:
                    raise Exception("No video analyzer available")
                
                self.analysis_done.emit(analysis_result)
                print("Detailed analysis completed")
            except Exception as e:
                print(f"Detailed analysis failed: {e}")
                import traceback
                traceback.print_exc()
                error_result = {
                    "error": f"Detailed analysis failed: {str(e)}",
                    "analysis": None
                }
                self.analysis_done.emit(error_result)
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp in HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _generate_fast_analysis_report(self, video_path: str) -> Dict:
        """Generate a fast, minimal analysis report for quick results."""
        from datetime import datetime
        
        # Simple report with minimal processing
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_text = f"""
REAL-TIME VIDEO ANALYSIS REPORT
==============================

Video File: {video_path}
Analysis Time: {timestamp}

SUMMARY:
--------
This report includes real-time analysis generated during video processing.
The system analyzed key frames and provided live explanations of detected activities.

REAL-TIME INSIGHTS:
------------------
Real-time GPT analysis was enabled during processing to provide immediate insights.
Check the application interface for live updates during video analysis.

NOTE: This is a fast analysis report. For detailed analysis with comprehensive frame-by-frame
examination, please use the CLI tools or enable detailed analysis mode.
"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'video_path': video_path,
            'description': report_text,
            'report': report_text,
            'events_count': 0,
            'anomaly_count': 0,
            'output_path': '',
            'events': []
        }
    
    def _create_explanation_prompt(self, events: List[Dict], video_path: str) -> str:
        """Create a prompt for GPT explanation based on recent events."""
        video_name = os.path.basename(video_path)
        event_descriptions = []
        
        for event in events:
            status = "normal activity" if event["is_normal"] else "anomalous activity"
            description = event.get("description", status)
            brightness_info = f", brightness: {event.get('brightness', 0):.1f}" if "brightness" in event else ""
            motion_info = f", motion: {event.get('motion_level', 'unknown')}" if "motion_level" in event else ""
            event_descriptions.append(f"At {event['timestamp_formatted']}: {description} (confidence: {event['confidence']:.2f}{brightness_info}{motion_info})")
        
        prompt = f"""
        Video: {video_name}
        Recent events:
        {chr(10).join(event_descriptions)}
        
        Please provide a concise explanation of what's happening in the video based on these recent events.
        Include a brief summary of the overall activity and highlight any significant changes or anomalies.
        """
        return prompt


class MainWindow(QMainWindow):
    """
    Main application window with a semi-transparent background image.
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("🎥 Advanced Video Anomaly Detection System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: transparent; color: #ecf0f1;")
        self.sensitivity_mode = "High"  # Default mode is High Sensitivity
        self.analysis_mode = "ultra_fast"  # Default analysis mode: ultra_fast, quick, balanced, or detailed
        # Enable analysis if either analyzer is available
        self.analysis_enabled = VIDEO_ANALYZER_AVAILABLE or ENHANCED_ANALYZER_AVAILABLE
        
        # Print availability status
        if ENHANCED_ANALYZER_AVAILABLE:
            print("✓ Enhanced Video Analyzer (YOLO) is available and will be used for detailed analysis")
        elif VIDEO_ANALYZER_AVAILABLE:
            print("✓ Basic Video Analyzer is available")
        else:
            print("⚠ No video analyzer available - detailed analysis disabled")
        
        # Print GPT analyzer status
        if GEMINI_AVAILABLE:
            try:
                # Test if GITHUB_TOKEN is properly set
                import os
                token = os.getenv('GITHUB_TOKEN', '')
                if token:
                    print("✓ GPT Analyzer is ENABLED and ready for real-time analysis")
                else:
                    print("⚠ GPT Analyzer available but GITHUB_TOKEN not set - real-time analysis disabled")
                    print("   Set GITHUB_TOKEN environment variable to enable GPT analysis")
            except:
                print("⚠ GPT Analyzer available but not properly configured")
        else:
            print("ℹ GPT Analyzer not available - install required dependencies to enable")
        self.current_video_path = None

        # Set up background label
        self.background_label = QLabel(self)

        # Load and process the background image from data directory
        image_path = "data/image.jpg"  # Use the image from data directory
        if os.path.exists(image_path):
            background_pixmap = QPixmap(image_path)
            
            # Apply semi-transparency using QPainter
            painter = QPainter(background_pixmap)
            painter.fillRect(background_pixmap.rect(), QColor(0, 0, 0, 120))  # RGBA: 120 = semi-transparency
            painter.end()
        else:
            # Create a default background if image not found
            background_pixmap = QPixmap(1400, 900)
            background_pixmap.fill(QColor(44, 62, 80))  # Dark blue background

        self.background_label.setPixmap(background_pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.lower()  # Ensure it stays below all other widgets

        #video player
        self.video_frame = QFrame(self)
        self.video_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;  /* Dark grey background */
                border: 3px solid #3498db;  /* Blue border */
                border-radius: 12px;       /* Rounded corners */
            }
        """)
        self.video_frame.setFixedSize(480, 350)  # Slightly larger than the video widget for padding

        # Add shadow effect to the QFrame
        shadow_effect = QGraphicsDropShadowEffect(self)
        shadow_effect.setBlurRadius(20)               # Increase for softer edges
        shadow_effect.setOffset(0, 0)                 # Shadow offset (x, y)
        shadow_effect.setColor(QColor(0, 0, 0, 200))  # Black shadow with transparency
        self.video_frame.setGraphicsEffect(shadow_effect)

        self.video_widget = QVideoWidget(self.video_frame)
        self.video_widget.setGeometry(6, 6, 468, 338)  # Leave space for the border

        self.video_player = QMediaPlayer(self)
        self.video_player.setVideoOutput(self.video_widget)

        # Widgets
        self.label = QLabel("Select a video to classify", self)
        self.label.setFont(QFont("Helvetica", 24, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton("Select Video", self)
        self.select_button.setFont(QFont("Helvetica", 16))
        self.select_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 15px;
                padding: 15px;
                max-width: 300px;
            }
            QPushButton:hover {
                background-color: #1abc9c;
            }
        """)
        self.select_button.clicked.connect(self.open_file_dialog)

        self.sensitivity_label = QLabel(f"Mode: {self.sensitivity_mode} Sensitivity", self)
        self.sensitivity_label.setFont(QFont("Helvetica", 16))
        self.sensitivity_label.setAlignment(Qt.AlignCenter)

        self.toggle_button = QPushButton("Switch", self)
        self.toggle_button.setFont(QFont("Helvetica", 16))
        self.toggle_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 15px;
                padding: 10px 15px;
                max-width: 150px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_sensitivity)

        # Analysis toggle button
        self.analysis_label = QLabel(f"Detailed Analysis: {'Enabled' if self.analysis_enabled else 'Disabled'}", self)
        self.analysis_label.setFont(QFont("Helvetica", 16))
        self.analysis_label.setAlignment(Qt.AlignCenter)

        self.analysis_toggle_button = QPushButton("Toggle Analysis", self)
        self.analysis_toggle_button.setFont(QFont("Helvetica", 16))
        self.analysis_toggle_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.analysis_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border-radius: 15px;
                padding: 10px 15px;
                max-width: 200px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self.analysis_toggle_button.clicked.connect(self.toggle_analysis)
        
        # Analysis mode selection
        self.analysis_mode_label = QLabel(f"Analysis Mode: {self.analysis_mode.title()}", self)
        self.analysis_mode_label.setFont(QFont("Helvetica", 16))
        self.analysis_mode_label.setAlignment(Qt.AlignCenter)
        
        self.analysis_mode_button = QPushButton("Change Mode", self)
        self.analysis_mode_button.setFont(QFont("Helvetica", 16))
        self.analysis_mode_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.analysis_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border-radius: 15px;
                padding: 10px 15px;
                max-width: 200px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.analysis_mode_button.clicked.connect(self.toggle_analysis_mode)
        
        # GPT Analysis Status
        gpt_status = "ENABLED" if GEMINI_AVAILABLE else "DISABLED"
        gpt_color = "#2ecc71" if GEMINI_AVAILABLE else "#e74c3c"
        self.gpt_status_label = QLabel(f"GPT Analysis: {gpt_status}", self)
        self.gpt_status_label.setFont(QFont("Helvetica", 16))
        self.gpt_status_label.setAlignment(Qt.AlignCenter)
        self.gpt_status_label.setStyleSheet(f"color: {gpt_color}; font-weight: bold;")
        
        # Add tooltip with instructions if GPT is disabled
        if not GEMINI_AVAILABLE:
            self.gpt_status_label.setToolTip("Set GITHUB_TOKEN environment variable to enable real-time GPT analysis")

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #34495e;
                border-radius: 15px;
                text-align: center;
                font: bold 14px;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #1abc9c;
                border-radius: 15px;
            }
        """)
        self.progress_bar.setValue(0)

        # Create main layout with sections
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Title Section
        title_section = QGroupBox()
        title_section.setTitle("🎥 Video Anomaly Detection System")
        title_section.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #ecf0f1;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        title_layout = QVBoxLayout(title_section)
        title_layout.addWidget(self.label)
        main_layout.addWidget(title_section)
        
        # Video Section
        video_section = QGroupBox()
        video_section.setTitle("📹 Video Player")
        video_section.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #ecf0f1;
                border: 2px solid #e74c3c;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        video_layout = QVBoxLayout(video_section)
        video_layout.addWidget(self.video_frame, alignment=Qt.AlignCenter)
        video_layout.addSpacing(10)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button, alignment=Qt.AlignCenter)
        video_layout.addLayout(button_layout)
        main_layout.addWidget(video_section)
        
        # Control Section
        control_section = QGroupBox()
        control_section.setTitle("⚙️ Analysis Controls")
        control_section.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #ecf0f1;
                border: 2px solid #9b59b6;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        control_layout = QGridLayout(control_section)
        
        # Sensitivity controls
        control_layout.addWidget(self.sensitivity_label, 0, 0)
        sensitivity_button_layout = QHBoxLayout()
        sensitivity_button_layout.addWidget(self.toggle_button)
        control_layout.addLayout(sensitivity_button_layout, 0, 1)
        
        # Analysis controls
        control_layout.addWidget(self.analysis_label, 1, 0)
        analysis_button_layout = QHBoxLayout()
        analysis_button_layout.addWidget(self.analysis_toggle_button)
        control_layout.addLayout(analysis_button_layout, 1, 1)
        
        # Analysis mode controls
        control_layout.addWidget(self.analysis_mode_label, 2, 0)
        analysis_mode_button_layout = QHBoxLayout()
        analysis_mode_button_layout.addWidget(self.analysis_mode_button)
        control_layout.addLayout(analysis_mode_button_layout, 2, 1)
        
        # GPT Analysis Status
        control_layout.addWidget(self.gpt_status_label, 3, 0)
        
        main_layout.addWidget(control_section)
        
        # Progress Section
        progress_section = QGroupBox()
        progress_section.setTitle("📊 Processing Status")
        progress_section.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #ecf0f1;
                border: 2px solid #1abc9c;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        progress_layout = QVBoxLayout(progress_section)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addWidget(progress_section)
        
        # Analysis Report Section
        self.analysis_section = QGroupBox()
        self.analysis_section.setTitle("📋 Complete Video Analysis Report")
        self.analysis_section.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #ecf0f1;
                border: 3px solid #f39c12;
                border-radius: 15px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: rgba(44, 62, 80, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 15px 0 15px;
                background-color: #f39c12;
                color: white;
                border-radius: 8px;
            }
        """)
        analysis_layout = QVBoxLayout(self.analysis_section)
        
        # Create scrollable text area for analysis report
        self.analysis_scroll = QScrollArea()
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumSize(900, 500)
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 2px solid #f39c12;
                border-radius: 10px;
                padding: 20px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 13px;
                line-height: 1.5;
                selection-background-color: #3498db;
            }
        """)
        
        self.analysis_scroll.setWidget(self.analysis_text)
        self.analysis_scroll.setWidgetResizable(True)
        self.analysis_scroll.setMinimumSize(920, 520)
        self.analysis_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #34495e;
                width: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background-color: #f39c12;
                border-radius: 7px;
                min-height: 25px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #e67e22;
            }
        """)
        
        # Initially hide the analysis section
        self.analysis_section.setVisible(False)
        analysis_layout.addWidget(self.analysis_scroll)
        main_layout.addWidget(self.analysis_section)
        
        # Create scrollable main widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #34495e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #3498db;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #2980b9;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #34495e;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #3498db;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #2980b9;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        # Create container widget for the main layout
        container = QWidget()
        container.setLayout(main_layout)
        
        # Set the container as the scroll area's widget
        scroll_area.setWidget(container)
        
        # Set the scroll area as the central widget
        self.setCentralWidget(scroll_area)

        # Set window to fullscreen
        self.showFullScreen()

    def resizeEvent(self, event):
        """
        Update the background label size when the window is resized.
        """
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        super(MainWindow, self).resizeEvent(event)

    def open_file_dialog(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4);;All Files (*)")
        if video_path:
            self.current_video_path = video_path
            self.video_player.setSource(QUrl.fromLocalFile(video_path))
            self.video_player.play()
            # Clear previous analysis
            self.analysis_section.setVisible(False)
            self.analysis_text.clear()
            self.start_video_processing(video_path)

    def start_video_processing(self, video_path):
        # Print the sensitivity mode to ensure it's being passed correctly
        print(f"Starting video processing with sensitivity mode: {self.sensitivity_mode}")
        
        # Recreate the VideoProcessor instance with the updated sensitivity mode
        self.processor = VideoProcessor(video_path, self.sensitivity_mode, self.analysis_mode, self.analysis_enabled)
        self.processor.progress_updated.connect(self.update_progress_bar)
        self.processor.classification_done.connect(self.show_result)
        self.processor.analysis_done.connect(self.show_detailed_analysis)
        self.processor.realtime_update.connect(self.show_realtime_update)
        self.processor.start()
    
    def show_realtime_update(self, update_text):
        """Display real-time GPT explanations in the analysis section with enhanced formatting."""
        # Format the real-time update with a timestamp and header
        from datetime import datetime
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        formatted_update = f"=" * 60 + f"\n-REAL-TIME GPT VIDEO ANALYSIS UPDATE-\n" + f"=" * 60 + f"\n{timestamp} {update_text}\n" + f"=" * 60 + "\n"
        
        # Add the real-time update to the analysis text
        current_text = self.analysis_text.toPlainText()
        if current_text:
            new_text = current_text + "\n\n" + formatted_update
        else:
            new_text = formatted_update
        
        self.analysis_text.setPlainText(new_text)
        self.analysis_section.setVisible(True)
        
        # Scroll to bottom to show latest update
        cursor = self.analysis_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.analysis_text.setTextCursor(cursor)
        
        # Flash the analysis section to draw attention
        self.flash_analysis_section()
    
    def flash_analysis_section(self):
        """Flash the analysis section to draw attention to real-time updates."""
        # Store original stylesheet
        original_style = self.analysis_section.styleSheet()
        
        # Flash with highlight color
        self.analysis_section.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #ecf0f1;
                border: 3px solid #e74c3c;
                border-radius: 15px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: rgba(44, 62, 80, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 15px 0 15px;
                background-color: #e74c3c;
                color: white;
                border-radius: 8px;
            }
        """)
        
        # Reset after a short delay
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, lambda: self.analysis_section.setStyleSheet(original_style))
    
    def update_gpt_status(self):
        """Update the GPT status label based on current token availability."""
        import os
        token = os.getenv('GITHUB_TOKEN', '')
        if GEMINI_AVAILABLE and token:
            self.gpt_status_label.setText("GPT Analysis: ENABLED")
            self.gpt_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.gpt_status_label.setToolTip("")
        elif GEMINI_AVAILABLE:
            self.gpt_status_label.setText("GPT Analysis: DISABLED (No Token)")
            self.gpt_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.gpt_status_label.setToolTip("Set GITHUB_TOKEN environment variable to enable real-time GPT analysis")
        else:
            self.gpt_status_label.setText("GPT Analysis: NOT AVAILABLE")
            self.gpt_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            self.gpt_status_label.setToolTip("Install required dependencies to enable GPT analysis")

    def update_progress_bar(self, value):
        self.progress_bar.setValue(int(value))

    def show_result(self, result):
        # Create a custom QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Video Classification")
        
        if result == "Normal":
            msg.setText("The video is classified as: Normal")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2ecc71; /* Green */
                    color: white;
                    min-width: 500px;
                    min-height: 350px;
                    padding: 20px;
                }
                QMessageBox QLabel {
                    background: transparent;
                    color: white;
                    font-size: 22px;
                }
                QMessageBox QPushButton {
                    background-color: #1abc9c;
                    color: white;
                    border-radius: 15px;
                    padding: 20px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #16a085;
                }
            """)
        else:
            msg.setText("The video is classified as: Not Normal")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #e74c3c; /* Red */
                    color: white;
                    min-width: 500px;
                    min-height: 350px;
                    padding: 20px;
                }
                QMessageBox QLabel {
                    background: transparent;
                    color: white;
                    font-size: 22px;
                }
                QMessageBox QPushButton {
                    background-color: #c0392b;
                    color: white;
                    border-radius: 15px;
                    padding: 20px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #e74c3c;
                }
            """)
        
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        
        # Adjust the font for the button
        for button in msg.buttons():
            button.setFont(QFont("Helvetica", 16, QFont.Bold))

        msg.resize(500, 350)  # Make the message box much bigger
        msg.exec()

    def show_detailed_analysis(self, analysis_result):
        """
        Display detailed video analysis results in the dedicated section.
        """
        if analysis_result.get("error"):
            # Show error in analysis section
            error_msg = f"Analysis Error:\n{analysis_result['error']}"
            if analysis_result.get("description"):
                error_msg += f"\n\n{analysis_result['description']}"
            self.analysis_text.setPlainText(error_msg)
            self.analysis_section.setVisible(True)
            # Scroll to analysis section
            self.scroll_to_analysis_section()
            return
        
        # Get analysis text - check both 'description' and 'report' keys
        analysis_text = analysis_result.get("description") or analysis_result.get("report") or "No analysis available"
        
        # Extract and highlight anomaly types if available
        if analysis_result.get("anomaly_count", 0) > 0:
            # Try to extract anomaly types from the report
            anomaly_info = self._extract_anomaly_types(analysis_text, analysis_result)
            if anomaly_info:
                # Prepend anomaly type information to the report
                analysis_text = anomaly_info + "\n\n" + analysis_text
        
        # Add real-time frame descriptions if available
        frame_descriptions = self._extract_frame_descriptions(analysis_result)
        if frame_descriptions:
            analysis_text = analysis_text + "\n\n" + frame_descriptions
        
        # If analysis_text is empty or too short, try to build a summary
        if not analysis_text or len(analysis_text.strip()) < 50:
            summary_parts = []
            summary_parts.append("Analysis Summary:")
            if analysis_result.get("events_count"):
                summary_parts.append(f"Events Analyzed: {analysis_result['events_count']}")
            if analysis_result.get("anomaly_count") is not None:
                summary_parts.append(f"Anomalies Detected: {analysis_result['anomaly_count']}")
            if analysis_result.get("output_path"):
                summary_parts.append(f"\nFull report saved to:\n{analysis_result['output_path']}")
                summary_parts.append("\nPlease check the report file for complete details.")
            
            if summary_parts:
                analysis_text = "\n".join(summary_parts)
            else:
                analysis_text = "Analysis completed but no description available. Check the report file if one was generated."
        
        # Display in the analysis section
        self.analysis_text.setPlainText(analysis_text)
        self.analysis_section.setVisible(True)
        
        # Scroll to top of analysis text
        self.analysis_text.moveCursor(QTextCursor.MoveOperation.Start)
        
        # Scroll to analysis section
        self.scroll_to_analysis_section()
    
    def _extract_anomaly_types(self, report_text: str, analysis_result: dict) -> str:
        """
        Extract and format anomaly type information from the report.
        
        Args:
            report_text: The full report text
            analysis_result: The analysis result dictionary
            
        Returns:
            Formatted string with anomaly type information
        """
        try:
            # If we have direct access to events, extract anomaly types from there
            if 'events' in analysis_result:
                events_with_anomalies = [e for e in analysis_result['events'] if e.get('anomaly_flag')]
                if events_with_anomalies:
                    anomaly_types = {}
                    for event in events_with_anomalies:
                        anomaly_type = event.get('anomaly_type', 'Unknown')
                        if anomaly_type:
                            anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
                    
                    if anomaly_types:
                        anomaly_lines = ["ANOMALY TYPES DETECTED:"]
                        anomaly_lines.append("=" * 50)
                        for atype, count in anomaly_types.items():
                            anomaly_lines.append(f"  {atype.title()}: {count} occurrence(s)")
                        anomaly_lines.append("=" * 50)
                        return "\n".join(anomaly_lines)
            
            # If we don't have direct access to events, try to parse from report text
            # Look for anomaly type distribution in the report
            lines = report_text.split('\n')
            anomaly_section_found = False
            anomaly_lines = ["ANOMALY TYPES DETECTED:"]
            anomaly_lines.append("=" * 50)
            
            for line in lines:
                if "Anomaly Type Distribution:" in line:
                    anomaly_section_found = True
                    continue
                elif anomaly_section_found:
                    if line.strip() == "" or line.startswith("=") or line.startswith("-"):
                        break
                    if ":" in line and not line.startswith(" "):
                        anomaly_lines.append("  " + line.strip())
            
            if len(anomaly_lines) > 2:  # More than just header and separator
                anomaly_lines.append("--------------------------------------------------")
                return "\n".join(anomaly_lines)
                
        except Exception as e:
            print(f"Error extracting anomaly types: {e}")
        
        # Fallback: Just indicate that anomalies were detected
        if analysis_result.get("anomaly_count", 0) > 0:
            return "ANOMALY DETECTED: Please check the detailed report below for specific anomaly types."
        
        return ""
    
    def _extract_frame_descriptions(self, analysis_result: dict) -> str:
        """
        Extract and format frame descriptions from the analysis result.
        
        Args:
            analysis_result: The analysis result dictionary
            
        Returns:
            Formatted string with frame descriptions
        """
        try:
            # If we have direct access to events, extract frame descriptions from there
            if 'events' in analysis_result:
                events_with_descriptions = [e for e in analysis_result['events'] if e.get('frame_description')]
                if events_with_descriptions:
                    description_lines = ["FRAME-BY-FRAME DESCRIPTIONS:"]
                    description_lines.append("--------------------------------------------------")
                    for i, event in enumerate(events_with_descriptions, 1):
                        timestamp = event.get('timestamp_formatted', f"Frame {i}")
                        description = event.get('frame_description', 'No description available')
                        description_lines.append(f"  [{timestamp}] {description}")
                    description_lines.append("--------------------------------------------------")
                    return "\n".join(description_lines)
        except Exception as e:
            print(f"Error extracting frame descriptions: {e}")
        
        return ""
    
    def _create_basic_summary(self, events: List[Dict], result: Dict, video_path: str) -> str:
        """Create a basic summary if report generation fails."""
        summary = []
        summary.append("================================================================================")
        summary.append("VIDEO ANALYSIS SUMMARY")
        summary.append("================================================================================")
        summary.append("")
        summary.append(f"Source Video: {video_path}")
        summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Total Events: {len(events)}")
        summary.append("")
        
        if events:
            total_persons = sum([e.get('person_count', 0) for e in events])
            total_anomalies = sum([1 for e in events if e.get('anomaly_flag', False)])
            all_objects = set()
            for e in events:
                all_objects.update(e.get('objects', []))
            
            # Calculate average persons per frame
            avg_persons = total_persons / len(events) if events else 0
            
            summary.append("STATISTICS")
            summary.append("--------------------------------------------------------------------------------")
            summary.append(f"Total Person Detections: {total_persons}")
            summary.append(f"Average Persons Per Frame: {avg_persons:.1f}")
            summary.append(f"Unique Objects: {len(all_objects)}")
            if all_objects:
                summary.append(f"Objects: {', '.join(sorted(all_objects))}")
            summary.append(f"Anomalies: {total_anomalies}")
            
            # Add anomaly rate
            if len(events) > 0:
                anomaly_rate = (total_anomalies / len(events)) * 100
                summary.append(f"Anomaly Rate: {anomaly_rate:.1f}%")
            
            summary.append("")
            
            # Add temporal insights
            if len(events) >= 3:
                person_counts = [e.get('person_count', 0) for e in events]
                max_persons = max(person_counts)
                min_persons = min(person_counts)
                
                max_idx = person_counts.index(max_persons)
                min_idx = person_counts.index(min_persons)
                
                max_time = events[max_idx].get('timestamp_formatted', 'N/A')
                min_time = events[min_idx].get('timestamp_formatted', 'N/A')
                
                summary.append("TEMPORAL INSIGHTS")
                summary.append("--------------------------------------------------------------------------------")
                summary.append(f"Peak Occupancy: {max_persons} persons at {max_time}")
                summary.append(f"Low Occupancy: {min_persons} persons at {min_time}")
                summary.append("")
            
            summary.append("SAMPLE EVENTS")
            summary.append("--------------------------------------------------------------------------------")
            for i, event in enumerate(events[:3], 1):
                summary.append(f"{i}. {event.get('timestamp_formatted', 'N/A')}: {event.get('person_count', 0)} person(s), {len(event.get('objects', []))} object(s)")
                if event.get('anomaly_flag'):
                    summary.append(f"   ⚠ ANOMALY: {event.get('anomaly_type', 'Unknown')}")
        else:
            summary.append("No events detected.")
        
        if result.get('output_path'):
            summary.append("")
            summary.append(f"Full report: {result['output_path']}")
        
        return "\n".join(summary)
    
    def _generate_summary_from_events(self, events: List[Dict], result: Dict) -> str:
        """
        Generate a summary report from events if full report is not available.
        
        Args:
            events: List of event dictionaries
            result: Result dictionary from enhanced analyzer
            
        Returns:
            Summary text
        """
        if not events:
            return "No events detected in the video."
        
        summary_parts = []
        summary_parts.append("================================================================================")
        summary_parts.append("VIDEO ANALYSIS SUMMARY")
        summary_parts.append("================================================================================")
        summary_parts.append("")
        summary_parts.append(f"Source Video: {result.get('video_path', 'Unknown')}")
        summary_parts.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append(f"Total Events Analyzed: {len(events)}")
        summary_parts.append("")
        
        # Calculate statistics
        total_persons = sum([e.get('person_count', 0) for e in events])
        total_anomalies = sum([1 for e in events if e.get('anomaly_flag', False)])
        all_objects = set()
        for e in events:
            all_objects.update(e.get('objects', []))
        
        summary_parts.append("SUMMARY STATISTICS")
        summary_parts.append("--------------------------------------------------------------------------------")
        summary_parts.append(f"Total Person Detections: {total_persons}")
        summary_parts.append(f"Unique Object Types: {len(all_objects)}")
        if all_objects:
            summary_parts.append(f"Objects Detected: {', '.join(sorted(all_objects))}")
        summary_parts.append(f"Anomaly Events: {total_anomalies}")
        summary_parts.append(f"Anomaly Percentage: {(total_anomalies/len(events)*100):.1f}%")
        summary_parts.append("")
        
        # Sample events
        summary_parts.append("SAMPLE EVENTS (First 5)")
        summary_parts.append("--------------------------------------------------------------------------------")
        for i, event in enumerate(events[:5], 1):
            summary_parts.append(f"Event #{i}:")
            summary_parts.append(f"  Time: {event.get('timestamp_formatted', 'N/A')}")
            summary_parts.append(f"  People: {event.get('person_count', 0)}")
            summary_parts.append(f"  Objects: {', '.join(event.get('objects', [])) if event.get('objects') else 'None'}")
            summary_parts.append(f"  Activity: {event.get('activity', 'N/A')}")
            summary_parts.append("")
        
        if len(events) > 5:
            summary_parts.append(f"... and {len(events) - 5} more events")
            summary_parts.append("")
        
        summary_parts.append("================================================================================")
        summary_parts.append(f"Full detailed report saved to: {result.get('output_path', 'N/A')}")
        summary_parts.append("================================================================================")
        
        return "\n".join(summary_parts)
    
    def scroll_to_analysis_section(self):
        """
        Scroll the main window to show the analysis section.
        """
        # Get the scroll area from the central widget
        scroll_area = self.centralWidget()
        if isinstance(scroll_area, QScrollArea):
            # Find the analysis section widget
            analysis_widget = self.analysis_section
            if analysis_widget.isVisible():
                # Scroll to the analysis section
                scroll_area.ensureWidgetVisible(analysis_widget)

    def toggle_analysis(self):
        """
        Toggle detailed analysis on/off.
        """
        if not VIDEO_ANALYZER_AVAILABLE:
            msg = QMessageBox(self)
            msg.setWindowTitle("Analysis Not Available")
            msg.setText("Detailed analysis is not available. Please check the video_analyzer module.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
            return
        
        self.analysis_enabled = not self.analysis_enabled
        self.analysis_label.setText(f"Detailed Analysis: {'Enabled' if self.analysis_enabled else 'Disabled'}")
        print(f"Detailed analysis {'enabled' if self.analysis_enabled else 'disabled'}")
    
    def toggle_analysis_mode(self):
        """
        Toggle analysis mode between ultra_fast, quick, balanced, and detailed.
        """
        modes = ["ultra_fast", "quick", "balanced", "detailed"]
        current_index = modes.index(self.analysis_mode)
        next_index = (current_index + 1) % len(modes)
        self.analysis_mode = modes[next_index]
        self.analysis_mode_label.setText(f"Analysis Mode: {self.analysis_mode.title()}")
        print(f"Analysis mode changed to: {self.analysis_mode}")

    def toggle_sensitivity(self):
        """
        Toggle sensitivity mode between High and Low.
        """
        if self.sensitivity_mode == "High":
            self.sensitivity_mode = "Low"
            self.sensitivity_label.setText(f"Mode: {self.sensitivity_mode} Sensitivity")
            print("Low sensitivity mode enabled")
        else:
            self.sensitivity_mode = "High"
            self.sensitivity_label.setText(f"Mode: {self.sensitivity_mode} Sensitivity")
            print("High sensitivity mode enabled")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
