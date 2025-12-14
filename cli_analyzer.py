#!/usr/bin/env python3
"""
Command-Line Video Anomaly Analyzer
Simple command-line interface for video anomaly detection.
"""

import os
import sys
import argparse
import time
from pathlib import Path

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

def analyze_video(video_path: str, mode: str = "balanced", verbose: bool = False):
    """
    Analyze a video file for anomalies.
    
    Args:
        video_path: Path to video file
        mode: Processing mode (ultra_fast, quick, balanced, detailed)
        verbose: Whether to show detailed output
    """
    print(f"🎬 Analyzing video: {video_path}")
    print(f"⚙️  Mode: {mode}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
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
                print("⚡ Using Ultra-Fast mode (processes ~5 frames)")
            elif mode == "quick":
                analyzer = EnhancedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.4,
                    iou_threshold=0.45
                )
                frames_per_second = 0.3
                print("🏃 Using Quick mode")
            elif mode == "balanced":
                analyzer = EnhancedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.3,
                    iou_threshold=0.45
                )
                frames_per_second = 0.5
                print("⚖️  Using Balanced mode")
            else:  # detailed
                analyzer = EnhancedVideoAnalyzer(
                    yolo_model="yolov8s.pt",  # Use small model for better accuracy
                    conf_threshold=0.25,
                    iou_threshold=0.45
                )
                frames_per_second = 1.0
                print("🔬 Using Detailed mode")
            
            if verbose:
                print("🔍 Processing with Enhanced Video Analyzer...")
            
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
                print("⚡ Using Ultra-Fast mode (processes ~5 frames)")
            elif mode == "quick":
                analyzer = OptimizedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.4
                )
                frames_per_second = 0.3
                max_frames = 10
                print("🏃 Using Quick mode")
            elif mode == "balanced":
                analyzer = OptimizedVideoAnalyzer(
                    yolo_model="yolov8n.pt",
                    conf_threshold=0.3
                )
                frames_per_second = 0.5
                max_frames = 15
                print("⚖️  Using Balanced mode")
            else:  # detailed
                analyzer = OptimizedVideoAnalyzer(
                    yolo_model="yolov8s.pt",
                    conf_threshold=0.25
                )
                frames_per_second = 1.0
                max_frames = 20
                print("🔬 Using Detailed mode")
            
            if verbose:
                print("🔍 Processing with Optimized Video Analyzer...")
            
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
        print(f"\n✅ Analysis completed in {processing_time:.2f} seconds")
        print(f"📊 Events analyzed: {result.get('events_count', 0)}")
        print(f"⚠️ Anomalies detected: {result.get('anomaly_count', 0)}")
        print(f"📄 Report saved to: {result.get('output_path', 'N/A')}")
        
        # Show summary of anomalies if any were found
        if result.get('anomaly_count', 0) > 0:
            print(f"\n🚨 ANOMALY ALERT!")
            print("⚠️  Potential security threats detected in the video.")
            
            # Show anomaly types if available
            if 'events' in result:
                anomaly_types = {}
                for event in result['events']:
                    if event.get('anomaly_flag'):
                        anomaly_type = event.get('anomaly_type', 'Unknown')
                        anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
                
                if anomaly_types:
                    print("\n🔍 Anomaly breakdown:")
                    for atype, count in anomaly_types.items():
                        print(f"   • {atype.title()}: {count} occurrence(s)")
        
        # Show a sample of the report
        if verbose and result.get('report'):
            print(f"\n📋 Sample report content:")
            report_lines = result['report'].split('\n')
            for i, line in enumerate(report_lines[:15]):  # Show first 15 lines
                print(f"   {line}")
            if len(report_lines) > 15:
                print("   ... (report truncated)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing video: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Video Anomaly Detector - CLI Version")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("-m", "--mode", choices=["ultra_fast", "quick", "balanced", "detailed"], 
                       default="balanced", help="Processing mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--version", action="version", version="Video Anomaly Detector v1.0")
    
    args = parser.parse_args()
    
    # Analyze the video
    success = analyze_video(args.video_path, args.mode, args.verbose)
    
    if success:
        print(f"\n🎉 Analysis complete! Check the report file for full details.")
        sys.exit(0)
    else:
        print(f"\n💥 Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()