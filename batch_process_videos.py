#!/usr/bin/env python3
"""
Batch Video Processing Script for Anomaly Detection
Processes multiple videos and generates reports for each one.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_video_analyzer import EnhancedVideoAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
    print("✓ Enhanced Video Analyzer (YOLO) is available")
except ImportError as e:
    ENHANCED_ANALYZER_AVAILABLE = False
    print(f"⚠ Enhanced Video Analyzer not available: {e}")

try:
    from optimized_video_analyzer import OptimizedVideoAnalyzer
    OPTIMIZED_ANALYZER_AVAILABLE = True
    print("✓ Optimized Video Analyzer is available")
except ImportError as e:
    OPTIMIZED_ANALYZER_AVAILABLE = False
    print(f"⚠ Optimized Video Analyzer not available: {e}")

def get_video_files(directory: str) -> List[str]:
    """
    Get all video files from a directory.
    
    Args:
        directory: Path to directory containing videos
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))
    
    return sorted(video_files)

def process_single_video(video_path: str, output_dir: str, mode: str = "balanced"):
    """
    Process a single video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save reports
        mode: Processing mode (ultra_fast, quick, balanced, detailed)
    """
    print(f"\n{'='*80}")
    print(f"Processing video: {video_path}")
    print(f"Mode: {mode}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Select analyzer based on availability and mode
        if ENHANCED_ANALYZER_AVAILABLE:
            # Set parameters based on mode
            if mode == "ultra_fast":
                analyzer = EnhancedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.5,
                    iou_threshold=0.45
                )
                frames_per_second = 0.1
            elif mode == "quick":
                analyzer = EnhancedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.4,
                    iou_threshold=0.45
                )
                frames_per_second = 0.3
            elif mode == "balanced":
                analyzer = EnhancedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.3,
                    iou_threshold=0.45
                )
                frames_per_second = 0.5
            else:  # detailed
                analyzer = EnhancedVideoAnalyzer(
                    yolo_model="yolov8s.pt",  # Use small model for better accuracy
                    conf_threshold=0.25,
                    iou_threshold=0.45
                )
                frames_per_second = 1.0
            
            # Process video
            result = analyzer.process_video_complete(
                video_path=video_path,
                frames_per_second=frames_per_second,
                anomaly_threshold=0.5
            )
            
        elif OPTIMIZED_ANALYZER_AVAILABLE:
            # Fallback to optimized analyzer
            if mode == "ultra_fast":
                analyzer = OptimizedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.5
                )
                frames_per_second = 0.1
                max_frames = 5
            elif mode == "quick":
                analyzer = OptimizedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.4
                )
                frames_per_second = 0.3
                max_frames = 10
            elif mode == "balanced":
                analyzer = OptimizedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.3
                )
                frames_per_second = 0.5
                max_frames = 15
            else:  # detailed
                analyzer = OptimizedVideoAnalyzer(
                    yolo_model="yolov8s.pt",
                    conf_threshold=0.25
                )
                frames_per_second = 1.0
                max_frames = 20
            
            # Process video
            result = analyzer.process_video_fast(
                video_path=video_path,
                frames_per_second=frames_per_second,
                max_frames=max_frames
            )
        else:
            print("❌ No video analyzer available!")
            return False
        
        # Report results
        processing_time = time.time() - start_time
        print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Events analyzed: {result.get('events_count', 0)}")
        print(f"⚠️ Anomalies detected: {result.get('anomaly_count', 0)}")
        print(f"📄 Report saved to: {result.get('output_path', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing video {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to process videos."""
    parser = argparse.ArgumentParser(description="Batch process videos for anomaly detection")
    parser.add_argument("input_dir", help="Directory containing video files")
    parser.add_argument("-o", "--output_dir", help="Output directory for reports (default: input_dir)")
    parser.add_argument("-m", "--mode", choices=["ultra_fast", "quick", "balanced", "detailed"], 
                       default="balanced", help="Processing mode")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video files
    print(f"🔍 Searching for video files in: {args.input_dir}")
    if args.recursive:
        print("🔄 Recursive search enabled")
    
    video_files = get_video_files(args.input_dir)
    
    if not video_files:
        print("❌ No video files found!")
        return
    
    print(f"🎬 Found {len(video_files)} video files")
    for i, video_file in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(video_file)}")
    
    # Confirm processing
    print(f"\n⚙️ Processing mode: {args.mode}")
    print(f"📂 Output directory: {output_dir}")
    
    confirm = input("\nProceed with processing? (y/N): ")
    if confirm.lower() != 'y':
        print("👋 Processing cancelled")
        return
    
    # Process videos
    successful = 0
    failed = 0
    
    print(f"\n🚀 Starting batch processing...")
    start_time = time.time()
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing {os.path.basename(video_file)}...")
        
        if process_single_video(video_file, output_dir, args.mode):
            successful += 1
        else:
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("🏁 BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏰ Total time: {total_time:.2f} seconds")
    print(f"⚡ Average time per video: {total_time/len(video_files):.2f} seconds" if video_files else "")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()