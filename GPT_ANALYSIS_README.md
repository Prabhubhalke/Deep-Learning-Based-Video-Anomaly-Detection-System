# GPT Video Analysis Setup Guide

This guide explains how to set up and use GPT-powered real-time video analysis in the Video Anomaly Detection System.

## Prerequisites

1. A GitHub account with access to GitHub Models
2. A GitHub Personal Access Token (PAT) with access to GitHub Models

## Getting Your GitHub Token

1. Go to GitHub.com and sign in to your account
2. Click on your profile picture in the top-right corner
3. Select "Settings"
4. In the left sidebar, scroll down and click "Developer settings"
5. Click "Personal access tokens" → "Tokens (classic)"
6. Click "Generate new token" → "Generate new token (classic)"
7. Give your token a name (e.g., "Video Analysis Token")
8. Set an expiration date
9. Under "Select scopes", check the following:
   - `read:org` (if needed for organization access)
   - `repo` (for repository access)
10. Click "Generate token"
11. Copy the generated token immediately (you won't see it again)

## Setting Up GPT Analysis

### Method 1: Using the Setup Script (Recommended)

Run the setup script to automatically configure your token:

```bash
# On Windows (PowerShell)
python set_github_token.py

# Or double-click the batch file
set_github_token.bat
```

The script will prompt you to enter your GitHub token and will save it to the `.env` file.

### Method 2: Manual Environment Variable Setup

#### On Windows (PowerShell):

```powershell
$Env:GITHUB_TOKEN="your_actual_token_here"
```

#### On Windows (Command Prompt):

```cmd
set GITHUB_TOKEN=your_actual_token_here
```

#### On Linux/Mac:

```bash
export GITHUB_TOKEN="your_actual_token_here"
```

### Method 3: Edit .env File Directly

Open the `.env` file in a text editor and replace the empty value:

```
# GitHub Models Integration
GITHUB_TOKEN=your_actual_token_here
```

## Verifying Setup

1. After setting your token, restart the application
2. Check the console output for confirmation:
   - ✓ GPT Analyzer initialized for real-time explanations
3. In the GUI, look for the "GPT Analysis: ENABLED" indicator in green

## Using Real-Time GPT Analysis

1. Launch the application:
   ```bash
   python application.py
   ```

2. Select a video file to analyze

3. The system will:
   - Process video frames in real-time
   - Send frame data to GPT for analysis every 10 frames
   - Display live updates in the analysis section
   - Generate a comprehensive final report

## Troubleshooting

### Common Issues:

1. **"GITHUB_TOKEN not found" error**
   - Solution: Make sure you've set the environment variable correctly
   
2. **"Invalid token" error**
   - Solution: Verify your token has the correct permissions and hasn't expired
   
3. **GPT analysis not showing in GUI**
   - Solution: Check that the "GPT Analysis: ENABLED" indicator is green

### Checking Token Validity:

You can test your token with this simple script:

```python
import os
from gemini_analyzer import GitHubModelsAnalyzer

try:
    analyzer = GitHubModelsAnalyzer()
    print("✓ Token is valid and GPT analyzer is working")
except Exception as e:
    print(f"✗ Token error: {e}")
```

## Best Practices

1. **Token Security**: Never share your GitHub token or commit it to version control
2. **Rate Limits**: Be aware that GitHub Models may have rate limits
3. **Costs**: Some GitHub Models may incur costs based on usage
4. **Privacy**: Understand that video analysis data is sent to GitHub Models

## Support

If you encounter issues:
1. Check that your GitHub token is valid and has proper permissions
2. Ensure you have internet connectivity
3. Verify the `.env` file contains your token correctly
4. Contact support if problems persist