import os
import requests
import time
from typing import List, Dict
from datetime import datetime
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

class GitHubModelsAnalyzer:
    """
    GPT-5 based analyzer using GitHub Models REST API
    """

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found")

        self.endpoint = "https://models.github.ai/inference/chat/completions"
        self.model = "gpt-4o"  # Trying simpler model identifier

    def _call_gpt(self, prompt: str, max_tokens=800) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional video surveillance analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }

        # Add retry mechanism for rate limiting
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:  # Not the last attempt
                        print(f"[WARN] Rate limited. Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"[ERROR] Rate limited after {max_retries} attempts. Response: {response.text}")
                        return f"API rate limit exceeded. Please try again later."
                elif response.status_code == 404:  # Model not found
                    print(f"[ERROR] Model not found: {self.model}. Response: {response.text}")
                    return f"Model {self.model} is not available. Please check the model identifier."
                else:
                    # Print detailed error information for debugging
                    print(f"[DEBUG] HTTP {response.status_code} Error")
                    print(f"[DEBUG] Response text: {response.text}")
                    print(f"[DEBUG] Request payload: {payload}")
                    response.raise_for_status()
            except requests.exceptions.Timeout:
                print(f"[ERROR] Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return "Request timed out. Please try again later."
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return f"Failed to connect to API: {str(e)}"
        
        # This shouldn't be reached, but just in case
        return "Unexpected error occurred while calling the API."

    # 🔹 REAL-TIME explanation (used every 10 frames)
    def generate_concise_event_description(self, events: List[Dict], video_path: str) -> str:
        prompt = f"""
Provide a concise 3–5 line professional explanation of what is happening in this video.

Video: {os.path.basename(video_path)}
Recent events:
{events[-5:]}

Focus on anomalies and activity changes.
"""
        return self._call_gpt(prompt, max_tokens=300)

    # 🔹 FINAL REPORT
    def generate_professional_report(self, events: List[Dict], video_path: str) -> str:
        anomalies = [e for e in events if e.get("anomaly_flag")]
        prompt = f"""
Generate a professional VIDEO ANALYSIS REPORT.

Video: {os.path.basename(video_path)}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Events: {len(events)}
Anomalies: {len(anomalies)}

Event timeline:
{events[:15]}

Structure:
1. Executive Summary (4–5 lines)
2. Key Findings with timestamps
3. Risk Assessment
4. Recommendations
"""
        return self._call_gpt(prompt, max_tokens=1200)