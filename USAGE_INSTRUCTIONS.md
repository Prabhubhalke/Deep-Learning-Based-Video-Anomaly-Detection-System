# Video Anomaly Detection System - Usage Instructions

## Getting Started

### Prerequisites
Make sure you have all required dependencies installed:
```bash
pip install -r requirements.txt
```

Required files:
- `data/prepared_data/anomaly_detection_model.h5` (main model)
- `data/prepared_data/label_encoder.npy` (labels)
- `data/image.jpg` (background image for GUI)

### Running the GUI Application

#### Option 1: Using the Startup Script (Windows)
Double-click on `start_app.bat` or run from command line:
```cmd
start_app.bat
```

#### Option 2: Direct Python Execution
```bash
python improved_application.py
```

### Running from Command Line (CLI)

For quick analysis without GUI:
```bash
python cli_analyzer.py path/to/video.mp4
```

Options:
- `-m ultra_fast`: Fastest processing (~5 seconds)
- `-m quick`: Quick processing (~15 seconds)
- `-m balanced`: Balanced speed/accuracy (~30 seconds)
- `-m detailed`: Highest accuracy (~60 seconds)
- `-v`: Verbose output with sample report

Example:
```bash
python cli_analyzer.py data/sample_video.mp4 -m quick -v
```

### Batch Processing Multiple Videos

Process multiple videos at once:
```bash
python batch_process_videos.py path/to/videos/
```

Options:
- `-o output_dir`: Specify output directory for reports
- `-m mode`: Processing mode (same as CLI analyzer)
- `--recursive`: Process subdirectories recursively

Example:
```bash
python batch_process_videos.py data/videos/ -m balanced -o data/reports/ --recursive
```

## Using the GUI Application

### Main Features

1. **Video Selection**
   - Click "Select Video" to choose a video file
   - Supported formats: MP4, AVI, MOV, MKV

2. **Sensitivity Modes**
   - **High Sensitivity** (default): Detects even minor anomalies
   - **Low Sensitivity**: Requires stronger evidence for anomaly detection
   - Click "Switch" to toggle between modes

3. **Analysis Controls**
   - **Detailed Analysis**: Enable/disable detailed YOLO-based analysis
   - **Analysis Mode**: Choose processing speed vs. accuracy
     - Ultra-Fast: ~5 seconds
     - Quick: ~15 seconds
     - Balanced: ~30 seconds
     - Detailed: ~60 seconds

4. **Results Display**
   - Classification result popup (Normal/Not Normal)
   - Detailed analysis report in scrollable section
   - Real-time updates during processing

### Menu Options

- **File > Open Video**: Select a video file
- **File > Exit**: Close the application
- **View > Toggle Fullscreen**: Switch between windowed/fullscreen
- **Settings > Preferences**: Application settings

## Troubleshooting

### Common Issues

1. **Application won't start**
   - Check that all required model files exist in `data/prepared_data/`
   - Ensure all dependencies are installed (`pip install -r requirements.txt`)

2. **"No video analyzer available"**
   - Install YOLO requirements: `pip install ultralytics`

3. **Slow processing**
   - Use "Ultra-Fast" or "Quick" analysis modes
   - Close other applications to free up system resources

4. **Memory errors**
   - Reduce analysis mode to "Ultra-Fast" or "Quick"
   - Process shorter videos

### Performance Tips

1. **For fastest results**: Use Ultra-Fast mode with YOLOv8n model
2. **For best accuracy**: Use Detailed mode with YOLOv8s/m model
3. **For batch processing**: Use Quick or Balanced mode
4. **Large videos**: Consider trimming before analysis

## Understanding Results

### Classification Results
- **Normal**: No significant anomalies detected
- **Not Normal**: Potential anomalies detected

### Detailed Analysis Report
The report includes:
1. **Executive Summary**: Brief overview of findings
2. **Timeline Analysis**: Events detected throughout the video
3. **Anomaly Details**: Specific anomalies with timestamps
4. **Object Analysis**: Objects detected in the video
5. **Temporal Insights**: Patterns over time
6. **Security Assessment**: Risk level evaluation
7. **Recommendations**: Security suggestions

### Anomaly Types
- **Crowd**: Unusually high number of people
- **Violence**: Weapons or fighting-related objects
- **Chaos**: Overly crowded or chaotic scenes
- **Abuse**: Potential abuse or harmful behavior
- **Suspicious Items**: Suspicious or prohibited items

## File Locations

- **Reports**: Saved in `data/` directory with `_timeline_report.txt` suffix
- **Models**: Located in `data/prepared_data/`
- **Background Image**: `data/image.jpg`

## Technical Notes

### System Requirements
- Python 3.7+
- TensorFlow 2.10+
- 8GB+ RAM recommended
- CUDA-compatible GPU recommended (but not required)

### Models Used
- **EfficientNetB0**: Feature extraction for classification
- **YOLOv8**: Object detection and detailed analysis
- **Custom Neural Network**: Anomaly classification

### Processing Pipeline
1. Video frames are extracted at regular intervals
2. Each frame is analyzed with EfficientNetB0 for classification
3. If anomalies are detected, detailed YOLO analysis is performed
4. Results are compiled into a comprehensive report
5. GitHub Models integration provides natural language summaries (if configured)