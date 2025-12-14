# Project Structure Documentation

## Overview
This document provides an accurate representation of the current project structure and component status for the Video Anomaly Detection System.

## Current Project Structure

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

## Component Descriptions

### Core Components (Actively Maintained)

1. **`application.py`** - Main GUI application with real-time analysis capabilities
   - Provides user-friendly interface for video analysis
   - Integrates all system components
   - Supports real-time playback and anomaly detection

2. **`main.py`** - Model training and management
   - Core deep learning model training functionality
   - Model saving and loading operations
   - Dataset preparation and management

3. **`video_analyzer.py`** - Core video analysis functionality
   - Traditional frame-by-frame analysis
   - Statistical anomaly detection
   - Base analysis capabilities

4. **`enhanced_video_analyzer.py`** - Enhanced YOLO-based analysis
   - Real-time object detection using YOLOv8
   - Advanced anomaly detection algorithms
   - Discrepancy handling between detection systems
   - Frame-to-text conversion for GPT analysis

5. **`github_gpt5_analyzer.py`** - GitHub Models integration
   - Natural language report generation
   - GPT-based video explanation capabilities
   - API communication with GitHub Models
   - Rate limit handling and error recovery

6. **`batch_process_videos.py`** - Batch processing capabilities
   - Multi-video analysis in sequence
   - Automated report generation
   - Bulk processing optimizations

7. **`cli_analyzer.py`** - Command-line interface tools
   - Terminal-based video analysis
   - Scriptable analysis workflows
   - Automation support

8. **`extract.py`** - Frame extraction utilities
   - Video to frame conversion
   - Dataset preparation
   - Frame organization and management

9. **`high_accuracy.py`** - High accuracy model training options
   - Extended training procedures
   - Enhanced model architectures
   - Accuracy-focused training parameters

10. **`test_fix.py`** - Test script for discrepancy handling
    - Verification of discrepancy resolution logic
    - GPT integration testing
    - System component validation

### Data Components

1. **`data/`** - Main data directory
   - Contains all video frames and trained models
   - Organized into training and testing subdirectories
   - Stores processed analysis results

2. **`train2.csv`** - Training video labels
   - Maps video filenames to classification labels
   - Used during model training phase

3. **`test2.csv`** - Testing video labels
   - Validation dataset for model evaluation
   - Used for accuracy assessment

4. **`requirements.txt`** - Python dependencies
   - Complete list of required packages
   - Version specifications for reproducibility

## System Architecture

### Data Flow
1. Video files are processed by `extract.py` to create frame sequences
2. Frames are used to train models via `main.py` or `high_accuracy.py`
3. Trained models are saved in `data/prepared_data/`
4. Videos are analyzed using either:
   - GUI application (`application.py`)
   - CLI tools (`cli_analyzer.py`)
   - Batch processor (`batch_process_videos.py`)
5. Analysis results are enhanced with GPT explanations via `github_gpt5_analyzer.py`
6. Reports are generated and saved to the data directory

### Component Interactions
- **GUI Application** ↔ **Enhanced Analyzer** ↔ **GPT Analyzer**
- **CLI Tools** → **Core Analyzer** → **GPT Analyzer**
- **Batch Processor** → **Multiple Analyzers** → **Report Generator**
- **Training Scripts** ↔ **Data Directory** ↔ **Model Files**

## Recent Updates

### Enhanced Discrepancy Handling
The system now intelligently manages cases where different detection systems produce conflicting results:
- Deep learning model classifies video as "Not Normal"
- YOLO analysis shows no clear anomalies
- GPT-based analysis provides contextual explanations

### Improved GitHub Models Integration
- Enhanced rate limit handling with exponential backoff
- Better error recovery for API connectivity issues
- Improved timeout management
- More reliable model identifiers

## Best Practices

### Development
- All core components are actively maintained
- New features should integrate with existing analysis pipeline
- GPT explanations should be consistent with detection results
- Error handling should be graceful and informative

### Deployment
- Ensure all dependencies in `requirements.txt` are installed
- Configure GitHub token in `.env` file for GPT features
- Verify ffmpeg installation for video processing
- Maintain proper data directory structure

### Maintenance
- Regular model retraining with new data
- Monitor GitHub API usage and rate limits
- Update YOLO models for improved detection accuracy
- Review discrepancy handling logic periodically

## Future Enhancements

1. **Ensemble Methods** - Combine results from multiple detection systems
2. **Confidence Scoring** - Provide reliability metrics for each detection
3. **Temporal Analysis** - Track activity patterns over time
4. **Visual Indicators** - GUI enhancements to show detection system contributions
5. **Advanced Discrepancy Resolution** - More sophisticated algorithms for handling conflicts