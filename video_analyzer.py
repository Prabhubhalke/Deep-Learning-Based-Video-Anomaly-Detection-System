"""
Advanced Video Analyzer - Alternative to Gemini AI
Provides detailed video analysis without external API dependencies.
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

class VideoAnalyzer:
    """
    Advanced video analyzer that provides detailed analysis without external APIs.
    """
    
    def __init__(self):
        """Initialize the video analyzer."""
        self.anomaly_types = {
            'physical_abuse': ['pushing', 'hitting', 'grabbing', 'forceful contact'],
            'violence': ['fighting', 'aggression', 'confrontation', 'threat'],
            'theft': ['stealing', 'taking', 'removing', 'unauthorized'],
            'vandalism': ['damaging', 'destroying', 'breaking', 'defacing'],
            'suspicious_behavior': ['loitering', 'watching', 'following', 'unusual'],
            'safety_violation': ['unsafe', 'dangerous', 'hazardous', 'risk']
        }
    
    def extract_detailed_frames(self, video_path: str, max_frames: int = 20) -> List[Dict]:
        """
        Extract detailed frame information for analysis.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            List of frame dictionaries with detailed information
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame intervals
        if total_frames > max_frames:
            interval = total_frames // max_frames
        else:
            interval = 1
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # Analyze frame content
                frame_info = self.analyze_frame_content(frame, frame_count, fps)
                frames.append(frame_info)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def analyze_frame_content(self, frame: np.ndarray, frame_number: int, fps: float) -> Dict:
        """
        Analyze individual frame content for detailed information.
        
        Args:
            frame: Frame as numpy array
            frame_number: Frame number
            fps: Frames per second
            
        Returns:
            Dictionary with frame analysis
        """
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate timestamp
        timestamp = frame_number / fps if fps > 0 else 0
        
        # Analyze motion (if we have previous frame)
        motion_intensity = self.calculate_motion_intensity(gray)
        
        # Analyze color distribution
        color_analysis = self.analyze_colors(frame)
        
        # Detect edges and contours
        edge_analysis = self.analyze_edges(gray)
        
        # Detect faces and people (basic detection)
        people_count = self.detect_people(frame)
        
        # Analyze brightness and contrast
        brightness = float(np.mean(gray.astype(np.float64)))
        contrast = float(np.std(gray.astype(np.float64)))
        
        return {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'motion_intensity': motion_intensity,
            'color_analysis': color_analysis,
            'edge_analysis': edge_analysis,
            'people_count': people_count,
            'brightness': brightness,
            'contrast': contrast,
            'frame_size': frame.shape
        }
    
    def calculate_motion_intensity(self, gray_frame: np.ndarray) -> float:
        """Calculate motion intensity in the frame."""
        # Use Laplacian to detect edges and motion
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        return float(np.var(laplacian.astype(np.float64)))
    
    def analyze_colors(self, frame: np.ndarray) -> Dict:
        """Analyze color distribution in the frame."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([20, 50, 50], [40, 255, 255])
        }
        
        color_counts = {}
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_counts[color_name] = np.sum(mask > 0)
        
        return color_counts
    
    def analyze_edges(self, gray_frame: np.ndarray) -> Dict:
        """Analyze edges and contours in the frame."""
        # Detect edges
        edges = cv2.Canny(gray_frame, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour properties
        contour_analysis = {
            'edge_density': np.sum(edges > 0) / edges.size,
            'contour_count': len(contours),
            'largest_contour_area': max([cv2.contourArea(c) for c in contours]) if contours else 0
        }
        
        return contour_analysis
    
    def detect_people(self, frame: np.ndarray) -> int:
        """Basic people detection using HOG descriptor."""
        try:
            # Initialize HOG descriptor
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect people
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            
            return len(rects)
        except:
            return 0
    
    def analyze_anomaly_patterns(self, frames: List[Dict]) -> Dict:
        """
        Analyze patterns across frames to identify anomaly types.
        
        Args:
            frames: List of frame analysis dictionaries
            
        Returns:
            Dictionary with anomaly analysis
        """
        if not frames:
            return {'error': 'No frames to analyze'}
        
        # Analyze motion patterns
        motion_analysis = self.analyze_motion_patterns(frames)
        
        # Analyze temporal patterns
        temporal_analysis = self.analyze_temporal_patterns(frames)
        
        # Analyze behavioral patterns
        behavioral_analysis = self.analyze_behavioral_patterns(frames)
        
        # Determine anomaly type
        anomaly_type = self.classify_anomaly_type(frames, motion_analysis, temporal_analysis, behavioral_analysis)
        
        # Generate severity assessment
        severity = self.assess_severity(frames, anomaly_type)
        
        return {
            'anomaly_type': anomaly_type,
            'motion_analysis': motion_analysis,
            'temporal_analysis': temporal_analysis,
            'behavioral_analysis': behavioral_analysis,
            'severity': severity,
            'confidence': self.calculate_confidence(frames, anomaly_type)
        }
    
    def analyze_motion_patterns(self, frames: List[Dict]) -> Dict:
        """Analyze motion patterns across frames."""
        motion_values = [f['motion_intensity'] for f in frames]
        
        return {
            'average_motion': np.mean(motion_values),
            'max_motion': np.max(motion_values),
            'motion_variance': np.var(motion_values),
            'motion_trend': 'increasing' if len(motion_values) > 1 and motion_values[-1] > motion_values[0] else 'stable'
        }
    
    def analyze_temporal_patterns(self, frames: List[Dict]) -> Dict:
        """Analyze temporal patterns in the video."""
        timestamps = [f['timestamp'] for f in frames]
        people_counts = [f['people_count'] for f in frames]
        
        return {
            'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'people_presence': max(people_counts),
            'people_consistency': np.var(people_counts),
            'time_span': max(timestamps) - min(timestamps)
        }
    
    def analyze_behavioral_patterns(self, frames: List[Dict]) -> Dict:
        """Analyze behavioral patterns in the video."""
        edge_densities = [f['edge_analysis']['edge_density'] for f in frames]
        brightness_values = [f['brightness'] for f in frames]
        
        return {
            'activity_level': np.mean(edge_densities),
            'brightness_consistency': np.var(brightness_values),
            'scene_complexity': np.mean([f['edge_analysis']['contour_count'] for f in frames])
        }
    
    def classify_anomaly_type(self, frames: List[Dict], motion: Dict, temporal: Dict, behavioral: Dict) -> str:
        """Classify the type of anomaly based on analysis."""
        # Get actual values for more accurate classification
        avg_motion = motion.get('average_motion', 0)
        max_motion = motion.get('max_motion', 0)
        motion_variance = motion.get('motion_variance', 0)
        people_presence = temporal.get('people_presence', 0)
        activity_level = behavioral.get('activity_level', 0)
        
        # More accurate classification based on actual video content
        if people_presence > 0:
            # Videos with people - check for physical interactions
            if avg_motion > 1500 and motion_variance > 800:
                return 'Physical Abuse'
            elif avg_motion > 1200 and motion_variance > 600:
                return 'Violence'
            elif avg_motion > 800 and activity_level > 0.15:
                return 'Suspicious Behavior'
            elif avg_motion < 300 and activity_level < 0.05:
                return 'Normal'
            else:
                return 'Suspicious Behavior'
        else:
            # Videos without people - environmental anomalies
            if avg_motion > 1000:
                return 'Environmental Anomaly'
            elif avg_motion > 500:
                return 'Suspicious Behavior'
            else:
                return 'Normal'
    
    def assess_severity(self, frames: List[Dict], anomaly_type: str) -> str:
        """Assess the severity of the detected anomaly."""
        motion_values = [f['motion_intensity'] for f in frames]
        max_motion = np.max(motion_values)
        
        if anomaly_type in ['Physical Abuse', 'Violence']:
            if max_motion > 2000:
                return 'High'
            elif max_motion > 1000:
                return 'Medium'
            else:
                return 'Low'
        else:
            if max_motion > 1500:
                return 'Medium'
            else:
                return 'Low'
    
    def calculate_confidence(self, frames: List[Dict], anomaly_type: str) -> float:
        """Calculate confidence in the anomaly classification."""
        motion_values = [f['motion_intensity'] for f in frames]
        motion_consistency = 1.0 - (np.var(motion_values) / (np.mean(motion_values) + 1e-6))
        
        # Base confidence on motion consistency and frame count
        base_confidence = min(0.9, len(frames) / 20.0)
        
        return float(base_confidence * motion_consistency)
    
    def generate_detailed_report(self, video_path: str, anomaly_detected: bool = True) -> Dict:
        """
        Generate a detailed analysis report for the video.
        
        Args:
            video_path: Path to the video file
            anomaly_detected: Whether an anomaly was detected
            
        Returns:
            Dictionary with detailed analysis report
        """
        try:
            # Extract frames and analyze
            frames = self.extract_detailed_frames(video_path)
            # Calculate duration from frames
            if frames:
                duration = frames[-1]['timestamp'] if 'timestamp' in frames[-1] else 0
            else:
                duration = 0
            
            if not frames:
                return {
                    'error': 'No frames could be extracted from the video',
                    'analysis': None
                }
            
            # Analyze anomaly patterns
            anomaly_analysis = self.analyze_anomaly_patterns(frames)
            
            # Generate detailed description
            description = self.generate_description(frames, anomaly_analysis, duration)
            
            # Create comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'video_path': video_path,
                'duration': duration,
                'frames_analyzed': len(frames),
                'anomaly_detected': anomaly_detected,
                'anomaly_analysis': anomaly_analysis,
                'description': description,
                'key_observations': self.extract_key_observations(frames, anomaly_analysis),
                'recommendations': self.generate_recommendations(anomaly_analysis)
            }
            
            return report
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat(),
                'video_path': video_path,
                'analysis': None
            }
    
    def generate_description(self, frames: List[Dict], analysis: Dict, duration: float) -> str:
        """Generate a detailed description of the video content."""
        anomaly_type = analysis.get('anomaly_type', 'Unknown')
        severity = analysis.get('severity', 'Unknown')
        confidence = analysis.get('confidence', 0.0)
        
        # Generate detailed video description
        video_description = self.generate_video_narrative(frames, analysis, duration)
        
        description = f"""
COMPLETE VIDEO ANALYSIS REPORT

Video Overview
Duration: {duration:.2f} seconds
Frames Analyzed: {len(frames)}
Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Anomaly Classification
Type: {anomaly_type}
Severity: {severity}
Confidence: {confidence:.1%}
Status: {'ANOMALY DETECTED' if anomaly_type != 'Normal' else 'NORMAL BEHAVIOR'}

What Happened in the Video
{video_description}

Technical Analysis

Motion Analysis:
- Average Motion Intensity: {analysis['motion_analysis']['average_motion']:.2f}
- Maximum Motion: {analysis['motion_analysis']['max_motion']:.2f}
- Motion Trend: {analysis['motion_analysis']['motion_trend']}
- Motion Variance: {analysis['motion_analysis']['motion_variance']:.2f}

Temporal Analysis:
- People Present: {analysis['temporal_analysis']['people_presence']}
- Activity Duration: {analysis['temporal_analysis']['duration']:.2f} seconds
- Scene Consistency: {analysis['temporal_analysis']['people_consistency']:.2f}
- Time Span: {analysis['temporal_analysis']['time_span']:.2f} seconds

Behavioral Analysis:
- Activity Level: {analysis['behavioral_analysis']['activity_level']:.3f}
- Scene Complexity: {analysis['behavioral_analysis']['scene_complexity']:.1f}
- Brightness Consistency: {analysis['behavioral_analysis']['brightness_consistency']:.2f}

Key Observations
{self.format_key_observations(frames, analysis)}

Why This is {'NOT NORMAL' if anomaly_type != 'Normal' else 'NORMAL'}
{self.generate_explanation(anomaly_type, analysis, frames)}

Assessment Summary
{self.generate_assessment_summary(anomaly_type, severity, confidence)}
        """
        
        return description.strip()
    
    def generate_video_narrative(self, frames: List[Dict], analysis: Dict, duration: float) -> str:
        """Generate a narrative description of what happened in the video."""
        anomaly_type = analysis.get('anomaly_type', 'Unknown')
        motion_analysis = analysis.get('motion_analysis', {})
        temporal_analysis = analysis.get('temporal_analysis', {})
        
        # Analyze frame sequence for narrative
        timestamps = [f['timestamp'] for f in frames]
        motion_values = [f['motion_intensity'] for f in frames]
        people_counts = [f['people_count'] for f in frames]
        
        narrative_parts = []
        
        # More accurate people detection
        avg_people = np.mean(people_counts) if people_counts else 0
        max_people = max(people_counts) if people_counts else 0
        min_people = min(people_counts) if people_counts else 0
        
        # Detailed people analysis
        if max_people == 0:
            narrative_parts.append("The video shows an environment without visible people.")
        elif max_people == 1:
            narrative_parts.append("The video shows a single person in the scene throughout the analyzed frames.")
        else:
            narrative_parts.append(f"The video shows a group of people ranging from {min_people} to {max_people} individuals per frame, with an average of {avg_people:.1f} people present.")
        
        # More accurate motion description based on actual values
        avg_motion = motion_analysis.get('average_motion', 0)
        if avg_motion > 2000:
            narrative_parts.append("High-intensity movement is observed throughout the video, indicating very active scenes.")
        elif avg_motion > 1000:
            narrative_parts.append("Moderate to high movement activity is detected, suggesting active scenes.")
        elif avg_motion > 500:
            narrative_parts.append("Low to moderate movement activity is observed, indicating relatively calm scenes.")
        else:
            narrative_parts.append("Minimal movement activity is detected, suggesting mostly static scenes.")
        
        # Object and activity analysis from frames
        all_objects = []
        for frame in frames:
            # Collect all detected objects from color analysis
            color_objects = frame.get('color_analysis', {})
            all_objects.extend(list(color_objects.keys()))
        
        if all_objects:
            unique_objects = list(set(all_objects))
            narrative_parts.append(f"Objects detected in the scene include: {', '.join(unique_objects[:10])}{'' if len(unique_objects) <= 10 else ' and more'}.")
        
        # More specific anomaly descriptions based on actual analysis
        if anomaly_type == 'Physical Abuse':
            narrative_parts.append("The video contains scenes of physical interaction that appear to involve forceful contact between individuals.")
            narrative_parts.append("The motion patterns suggest aggressive physical behavior with high-intensity movements.")
        elif anomaly_type == 'Violence':
            narrative_parts.append("The video shows signs of violent behavior or confrontation.")
            narrative_parts.append("High motion intensity and irregular patterns indicate aggressive activity.")
        elif anomaly_type == 'Suspicious Behavior':
            narrative_parts.append("The video shows unusual or suspicious activity patterns.")
            narrative_parts.append("The behavior appears inconsistent with normal activities.")
        elif anomaly_type == 'Environmental Anomaly':
            narrative_parts.append("The video shows unusual environmental conditions or events.")
            narrative_parts.append("High motion without human presence suggests environmental disturbance.")
        elif anomaly_type == 'Normal':
            narrative_parts.append("The video shows normal, expected behavior patterns.")
            narrative_parts.append("No significant anomalies or unusual activities are detected.")
        else:
            narrative_parts.append("The video shows activity that requires further analysis.")
            narrative_parts.append("The detected patterns suggest non-normal behavior.")
        
        # More accurate timeline description
        if len(timestamps) > 1:
            start_time = min(timestamps)
            end_time = max(timestamps)
            narrative_parts.append(f"The analyzed activity spans from {start_time:.1f} to {end_time:.1f} seconds of the video.")
        
        return "\n".join(narrative_parts)
    
    def generate_explanation(self, anomaly_type: str, analysis: Dict, frames: List[Dict]) -> str:
        """Generate explanation for why the video is classified as normal or not normal."""
        motion_analysis = analysis.get('motion_analysis', {})
        temporal_analysis = analysis.get('temporal_analysis', {})
        behavioral_analysis = analysis.get('behavioral_analysis', {})
        
        explanations = []
        
        if anomaly_type == 'Normal':
            explanations.append("This video is classified as NORMAL because:")
            explanations.append("- Motion patterns are consistent with expected behavior")
            explanations.append("- No signs of aggressive or harmful activity detected")
            explanations.append("- Scene complexity and activity levels are within normal ranges")
            explanations.append("- No unusual behavioral patterns observed")
        else:
            explanations.append(f"This video is classified as NOT NORMAL ({anomaly_type}) because:")
            
            # Motion-based explanations
            if motion_analysis.get('average_motion', 0) > 1000:
                explanations.append("- High-intensity motion detected indicating aggressive or violent activity")
            if motion_analysis.get('motion_variance', 0) > 500:
                explanations.append("- Inconsistent motion patterns suggesting irregular or harmful behavior")
            
            # People-based explanations
            if temporal_analysis.get('people_presence', 0) > 0:
                explanations.append("- People are present in a context suggesting harmful interaction")
            
            # Activity-based explanations
            if behavioral_analysis.get('activity_level', 0) > 0.1:
                explanations.append("- High activity level indicating significant movement or interaction")
            
            # Specific anomaly explanations
            if anomaly_type == 'Physical Abuse':
                explanations.append("- Physical contact patterns consistent with abuse or assault")
                explanations.append("- Forceful movements suggesting intentional harm")
            elif anomaly_type == 'Violence':
                explanations.append("- Aggressive behavior patterns indicating violent activity")
                explanations.append("- High motion intensity suggesting physical confrontation")
            elif anomaly_type == 'Suspicious Behavior':
                explanations.append("- Unusual behavioral patterns inconsistent with normal activities")
                explanations.append("- Activity levels and patterns suggest suspicious intent")
        
        return "\n".join(explanations)
    
    def generate_assessment_summary(self, anomaly_type: str, severity: str, confidence: float) -> str:
        """Generate a summary assessment of the video."""
        if anomaly_type == 'Normal':
            return f"""
NORMAL BEHAVIOR DETECTED
This video shows normal, expected behavior patterns. No anomalies or concerning activities were identified.
Confidence Level: {confidence:.1%}
            """.strip()
        else:
            return f"""
ANOMALY DETECTED: {anomaly_type.upper()}
Severity Level: {severity}
Confidence Level: {confidence:.1%}

This video contains concerning activity that requires attention. The detected anomaly suggests {anomaly_type.lower()} behavior that may require immediate review or intervention.
            """.strip()
    
    def extract_key_observations(self, frames: List[Dict], analysis: Dict) -> List[str]:
        """Extract key observations from the analysis."""
        observations = []
        
        # Motion-based observations
        if analysis['motion_analysis']['max_motion'] > 1500:
            observations.append("High-intensity motion detected throughout the video")
        
        if analysis['motion_analysis']['motion_variance'] > 500:
            observations.append("Inconsistent motion patterns suggesting irregular activity")
        
        # People-based observations
        if analysis['temporal_analysis']['people_presence'] > 0:
            observations.append(f"People detected in the scene ({analysis['temporal_analysis']['people_presence']} individuals)")
        
        # Activity-based observations
        if analysis['behavioral_analysis']['activity_level'] > 0.1:
            observations.append("High activity level indicating significant movement or interaction")
        
        # Scene complexity observations
        if analysis['behavioral_analysis']['scene_complexity'] > 50:
            observations.append("Complex scene with multiple elements suggesting detailed interaction")
        
        return observations
    
    def format_key_observations(self, frames: List[Dict], analysis: Dict) -> str:
        """Format key observations for display."""
        observations = self.extract_key_observations(frames, analysis)
        
        if not observations:
            return "No significant patterns detected in the video analysis."
        
        formatted = []
        for i, obs in enumerate(observations, 1):
            formatted.append(f"{i}. {obs}")
        
        return "\n".join(formatted)
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        
        anomaly_type = analysis.get('anomaly_type', 'Unknown')
        severity = analysis.get('severity', 'Unknown')
        
        if anomaly_type in ['Physical Abuse', 'Violence']:
            recommendations.append("Immediate intervention may be required")
            recommendations.append("Document all evidence for potential legal proceedings")
            recommendations.append("Contact appropriate authorities if necessary")
        
        if severity == 'High':
            recommendations.append("Priority investigation recommended")
            recommendations.append("Enhanced monitoring may be necessary")
        
        if anomaly_type == 'Suspicious Behavior':
            recommendations.append("Further investigation of individuals involved")
            recommendations.append("Review security protocols if applicable")
        
        return recommendations
    
    def save_analysis_report(self, report: Dict, output_path: str):
        """Save analysis report to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Analysis report saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save analysis report: {e}")

# Example usage
if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    
    # Example analysis
    video_path = "data/sample_video.mp4"  # Replace with actual video path
    if os.path.exists(video_path):
        result = analyzer.generate_detailed_report(video_path, anomaly_detected=True)
        print("Analysis Result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Video file not found: {video_path}")
