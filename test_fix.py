#!/usr/bin/env python3
"""
Test script to verify the fix for discrepancy between deep learning model 
and YOLO analysis in video anomaly detection.
"""

from enhanced_video_analyzer import EnhancedVideoAnalyzer

def test_no_anomalies_case():
    """Test case where deep learning model detects abuse but YOLO doesn't detect anomalies"""
    print("=" * 80)
    print("TEST CASE 1: No YOLO anomalies but classified as Not Normal by deep learning")
    print("=" * 80)
    
    analyzer = EnhancedVideoAnalyzer()
    
    # Simulate events where no persons are detected (only cars)
    sample_events = [
        {
            'timestamp_formatted': '00:00:01',
            'person_count': 0,
            'objects': ['car'],
            'anomaly_flag': False
        },
        {
            'timestamp_formatted': '00:00:05', 
            'person_count': 0,
            'objects': ['car', 'car'],
            'anomaly_flag': False
        },
        {
            'timestamp_formatted': '00:00:10',
            'person_count': 0,
            'objects': ['car', 'car', 'car'],
            'anomaly_flag': False
        }
    ]
    
    result = analyzer.generate_gpt_frame_explanation(sample_events, 'abuse_video.mp4')
    print("GPT Explanation:")
    print(result)
    print("\n")

def test_with_anomalies_case():
    """Test case where YOLO detects anomalies"""
    print("=" * 80)
    print("TEST CASE 2: YOLO detects anomalies")
    print("=" * 80)
    
    analyzer = EnhancedVideoAnalyzer()
    
    # Simulate events where persons are detected and anomalies are flagged
    sample_events = [
        {
            'timestamp_formatted': '00:00:01',
            'person_count': 1,
            'objects': ['person'],
            'anomaly_flag': False
        },
        {
            'timestamp_formatted': '00:00:05', 
            'person_count': 3,
            'objects': ['person', 'person', 'person'],
            'anomaly_flag': True  # Multiple persons = potential abuse
        },
        {
            'timestamp_formatted': '00:00:10',
            'person_count': 2,
            'objects': ['person', 'person', 'knife'],
            'anomaly_flag': True  # Person with weapon
        }
    ]
    
    result = analyzer.generate_gpt_frame_explanation(sample_events, 'abuse_video.mp4')
    print("GPT Explanation:")
    print(result)
    print("\n")

if __name__ == "__main__":
    print("Testing EnhancedVideoAnalyzer fix for discrepancy handling...")
    test_no_anomalies_case()
    test_with_anomalies_case()
    print("Test completed!")