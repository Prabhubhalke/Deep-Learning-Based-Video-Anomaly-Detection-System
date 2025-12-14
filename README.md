# Deep-Learning-Based-Video-Anomaly-Detection-System
# Video Anomaly Detection System

This is a complete video anomaly detection system that can classify videos as "Normal" or "Anomaly" using deep learning techniques. The system uses EfficientNetB0 for feature extraction and a custom neural network for classification. It now includes advanced GitHub Models integration for faster, more natural language report generation and video explanation features.

## Project Structure

```
├── data/
│   ├── train_frames_2/          # Training frames extracted from videos
│   │   ├── anomaly/
│   │   └── normal/
│   ├── test_frames_2/           # Testing frames extracted from videos
│   │   ├── anomaly/
│   │   └── normal/
│   ├── prepared_data/           # Prepared datasets and trained models
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_test.npy
│   │   ├── y_test.npy
│   │   ├── label_encoder.npy
│   │   ├── anomaly_detection_model.h5
│   │   └── high_accuracy_model.h5
│   └── *.mp4                   # Original video files
├── train2.csv                  # Training video labels
├── test2.csv                   # Testing video labels
├── extract.py                  # Frame extraction script
├── main.py                     # Main training script
├── high_accuracy.py            # High accuracy model training
├── video_analyzer.py           # Detailed video analysis
├── enhanced_video_analyzer.py  # Enhanced YOLO-based analysis
├── github_gpt5_analyzer.py     # GitHub Models report generation
├── batch_process_videos.py     # Batch video processing
├── cli_analyzer.py             # Command-line analysis tools
├── application.py              # GUI application
├── test_fix.py                 # Test script for discrepancy handling
└── requirements.txt            # Python dependencies
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install ffmpeg and add it to your PATH:
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

## Usage

### 1. Data Preparation

The system requires labeled video data in CSV files:
- `train2.csv`: Training data with video names and labels
- `test2.csv`: Testing data with video names and labels

Example format:
```csv
video_name,label
video1.mp4,normal
video2.mp4,anomaly
```

### 2. Frame Extraction

Run the frame extraction script to convert videos to frames:
```bash
python extract.py
```

This will create frame directories in `data/train_frames_2` and `data/test_frames_2`.

### 3. Model Training

Train the model using the prepared dataset:
```bash
python main.py
```

For higher accuracy training:
```bash
python high_accuracy.py
```

### 4. Batch Analysis

Analyze multiple videos at once:
```bash
python batch_process_videos.py
```

### 5. GUI Application

Run the graphical user interface:
```bash
python application.py
```

The GUI application provides:
- Video file selection
- Real-time video playback
- Anomaly detection with confidence scores
- Sensitivity mode selection (High/Low)
- Detailed analysis reports
- Enhanced YOLO-based object detection
- Natural language reports powered by GitHub Models
- Multiple analysis modes (Ultra-Fast, Quick, Balanced, Detailed)
- Video explanation feature for concise summaries
- Comprehensive risk assessment and recommendations

### 6. Command-line Analysis

Analyze videos using the command-line interface:
```bash
python cli_analyzer.py /path/to/video.mp4
```

## Features

- **Deep Learning Model**: Uses EfficientNetB0 for feature extraction and custom neural networks for classification
- **Real-time Processing**: Processes videos in real-time for immediate feedback
- **Multiple Sensitivity Modes**: Adjustable sensitivity for different use cases
- **Detailed Analysis**: Provides comprehensive reports for detected anomalies
- **Batch Processing**: Analyze multiple videos simultaneously
- **GUI Interface**: User-friendly graphical interface for easy operation
- **GitHub Models Integration**: Generates natural language reports using GitHub Models for faster analysis
- **Video Explanation Feature**: Creates concise, professional summaries of video content
- **Enhanced YOLO Analysis**: Uses YOLOv8 models for detailed object detection and anomaly classification
- **Optimized Processing**: Multiple analysis modes for balancing speed and accuracy
- **Plain Text Reporting**: All reports are generated in plain text format without markdown syntax
- **Ultra-Fast Mode**: New ultra-fast analysis mode for near-instant results
- **Comprehensive Reporting**: Enhanced reports with risk assessment, temporal insights, and detailed recommendations
- **Multi-layered Analysis**: Combines statistical analysis, AI insights, and security assessments
- **Discrepancy Handling**: Intelligent handling of inconsistencies between different detection systems

## Model Performance

The trained model achieves good accuracy on the test dataset with:
- Precision, Recall and F1-Score metrics displayed during training
- Binary classification between "Normal" and "Anomaly" classes

The enhanced system also provides:
- Real-time object detection using YOLOv8 models
- Anomaly classification into specific types (crowd, violence, chaos, abuse, suspicious_items)
- Natural language analysis reports generated by GitHub Models
- Video explanations with professional summaries
- Multiple processing modes for different speed/accuracy requirements
- Comprehensive risk assessment with actionable recommendations
- Temporal pattern analysis for behavioral insights
- Intelligent discrepancy resolution between detection systems

### Analysis Modes

The system offers four analysis modes to balance speed and accuracy:

1. **Ultra-Fast Mode**: Processes only 5 frames with 0.1 FPS using YOLOv8 nano model for near-instant results (1-5 seconds)
2. **Quick Mode**: Uses YOLOv8 nano model for fastest processing (5-15 seconds)
3. **Balanced Mode**: Uses YOLOv8 small model for balanced speed/accuracy (15-30 seconds)
4. **Detailed Mode**: Uses YOLOv8 medium model for highest accuracy (30-60 seconds)

Users can select the appropriate mode based on their needs through the GUI interface.

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV-Python 4.5+
- NumPy 1.21+
- Pandas 1.3+
- Scikit-learn 1.0+
- PySide6 6.0+
- Ultralytics 8.0+ (for YOLO-based analysis)
- Azure AI Inference 1.0+ (for GitHub Models integration)
- FFmpeg (for video processing)
- Python-dotenv 0.19+ (for environment management)

See `requirements.txt` for the complete list of dependencies.

## Configuration

Before using the GitHub Models features, you need to configure your GitHub token:

1. Create a GitHub personal access token at https://github.com/settings/tokens
2. Give it the `models:read` permission
3. Add your token to the `.env` file:
   ```
   GITHUB_TOKEN=your_actual_github_token_here
   ```

## Video Explanation Feature

The system now includes an advanced video explanation feature that generates concise, professional descriptions of what's happening in analyzed videos using GitHub Models. This feature provides:

- **Concise Summaries**: 3-5 line professional explanations of video content
- **Human-like Reporting**: Explanations written in a style similar to human security analysts
- **Anomaly Focus**: Highlights key activities and anomalies detected in the video
- **Automatic Integration**: Video explanations are automatically prepended to analysis reports
- **Intelligent Discrepancy Handling**: Provides context-aware explanations when different detection systems produce conflicting results

## GitHub Models Integration

The system now includes integration with GitHub Models for faster, more natural language report generation. This integration provides:

- **Natural Language Reports**: Human-readable analysis reports generated by GitHub Models
- **Faster Processing**: Cloud-based AI processing reduces local computation time
- **Professional Formatting**: Well-structured reports with executive summaries, findings and recommendations
- **Fallback Mechanism**: Automatic fallback to traditional report generation if AI service is unavailable
- **Rate Limit Handling**: Built-in retry mechanisms for handling API rate limits
- **Error Recovery**: Graceful handling of API errors and timeouts

To use the GitHub Models integration:

1. Create a GitHub personal access token at https://github.com/settings/tokens
2. Give it the `models:read` permission
3. Add your token to the `.env` file:
   ```
   GITHUB_TOKEN=your_actual_github_token_here
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. The system will automatically use GitHub Models when generating detailed analysis reports

The GitHub Models integration enhances the user experience by providing more intuitive and actionable analysis reports while maintaining the accuracy of the underlying detection algorithms.

## Testing GitHub Models Integration

To test the GitHub Models integration without processing actual videos:

```bash
python test_fix.py
```

This will run tests to verify the discrepancy handling functionality and GitHub Models integration.

## Enhanced Reporting Features

The system now includes enhanced reporting capabilities with:

- **Risk Assessment**: Automated security risk evaluation based on detected anomalies
- **Temporal Insights**: Analysis of patterns and trends throughout the video
- **Statistical Summaries**: Detailed metrics on object counts, activities and anomalies
- **Actionable Recommendations**: Security-focused suggestions based on analysis findings
- **Multi-source Integration**: Combines results from multiple analysis engines
- **Comprehensive Reports**: Single-file reports containing all analysis information
- **Discrepancy Resolution**: Clear explanations when different detection systems produce conflicting results

## Recent Improvements

### Discrepancy Handling Enhancement
The system now intelligently handles cases where different detection systems produce conflicting results:
- When the deep learning model classifies a video as "Not Normal" but YOLO analysis shows no clear anomalies, the system provides contextual explanations
- GPT-based analysis now considers metadata flags, off-camera activity, and environmental factors
- Users receive comprehensive information to make informed decisions even when detection systems disagree

### Performance Optimizations
- Enhanced frame processing algorithms for faster analysis
- Improved memory management for large video files
- Optimized YOLO model selection based on analysis mode

### Robustness Improvements
- Better error handling for API connectivity issues
- Enhanced retry mechanisms for rate-limited requests
- Graceful degradation when external services are unavailable

## Core Components Status

The following core components are actively maintained and used in the system:

- `application.py` - Main GUI application with real-time analysis
- `main.py` - Model training and management
- `video_analyzer.py` - Core video analysis functionality
- `enhanced_video_analyzer.py` - Enhanced YOLO-based analysis with discrepancy handling
- `github_gpt5_analyzer.py` - GitHub Models integration for natural language reports
- `batch_process_videos.py` - Batch processing capabilities
- `cli_analyzer.py` - Command-line interface tools
- `extract.py` - Frame extraction utilities
- `high_accuracy.py` - High accuracy model training options

These files form the backbone of the video anomaly detection system and are actively developed and maintained.
