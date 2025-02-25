# Deepfake Detection System

![Deepfake Detection System](https://img.shields.io/badge/Version-1.0-blue.svg) ![Python](https://img.shields.io/badge/Python-3.12-green.svg) ![Django](https://img.shields.io/badge/Django-5.1-orange.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A sophisticated, real-time deepfake detection system built with Django, leveraging computer vision and audio analysis to identify manipulated video content. This system processes uploaded videos, extracts metadata, and analyzes visual-audio consistency to determine authenticity with a confidence score, presented in an intuitive web interface.

---

## Table of Contents
1. [Overview](#overview)
2. [Technical Architecture](#technical-architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Advanced Capabilities](#advanced-capabilities)
7. [Use Cases](#use-cases)
8. [Industries and Sectors](#industries-and-sectors)
9. [Competitors](#competitors)
10. [Future Enhancements](#future-enhancements)
11. [License](#license)

---

## Overview

The Deepfake Detection System is designed to combat the rising threat of synthetic media by analyzing video content for signs of manipulation. It employs OpenCV for video frame analysis, FFmpeg for audio extraction, and Librosa for audio feature extraction, integrated within a Django framework. The system outputs a verdict ("Fake" or "Real") with a confidence score (e.g., "82.75% Fake"), based on the consistency between visual and audio signals.

This implementation prioritizes modularity, extensibility, and real-time feedback, making it suitable for both standalone use and integration into larger security or media verification platforms.

---

## Technical Architecture

### Core Components
- **Backend**: Django 5.1 with Python 3.12
  - Handles file uploads, processing orchestration, and HTTP responses.
  - Uses a synchronous workflow for simplicity, with potential for async upgrades (e.g., Celery).
- **Video Processing**: OpenCV (`cv2`)
  - Extracts metadata (size, duration, FPS, resolution).
  - Performs frame-by-frame face detection using Haar Cascade Classifier as a proxy for visual consistency.
- **Audio Processing**: FFmpeg + Librosa
  - FFmpeg extracts audio tracks efficiently into WAV format (16kHz, mono).
  - Librosa analyzes audio features (pitch, tempo, energy, voiced segments) for consistency scoring.
- **Frontend**: Django Templates with Tailwind CSS
  - Responsive UI with animations for upload and results display.
  - Real-time feedback via a loading spinner during processing.
- **Logging**: Python `logging` module
  - Detailed debug logs with timestamps for performance monitoring and troubleshooting.

### Workflow
1. **Upload**: Users upload videos via a drag-and-drop interface.
2. **Metadata Extraction**: OpenCV extracts video metadata instantly.
3. **Deepfake Analysis**:
   - Visual: Samples frames, detects faces, calculates consistency percentage.
   - Audio: Extracts audio, counts voiced segments, weights by energy for consistency.
   - Decision: Compares visual and audio consistency; if difference > 30% or visual < 20% with audio > 80%, marks as "Fake."
   - Confidence: `100 - (consistency_diff / 100) * 50`, labeled as "X% Fake" or "X% Real."
4. **Results**: Displays metadata, percentages, decision, and confidence in a tabular format.

### Dependencies
- `django==5.1`
- `opencv-python`
- `ffmpeg-python`
- `librosa`
- `numpy`
- FFmpeg binary (system-installed)

---

## Features

- **Real-Time Analysis**: Processes videos on upload with immediate results.
- **Metadata Extraction**: Provides file size, duration, FPS, and resolution.
- **Consistency Scoring**: Calculates visual and audio consistency percentages.
- **Confidence Verdict**: Outputs "X% Fake" or "X% Real" with color-coded confidence.
- **Detailed Logging**: Tracks every step for debugging and performance analysis.
- **Responsive UI**: Modern Tailwind CSS design with animations and loading states.

---

## Installation

### Prerequisites
- Python 3.12+
- FFmpeg installed (`sudo apt-get install ffmpeg` on Ubuntu, `brew install ffmpeg` on macOS)
- Virtual environment recommended

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Example `requirements.txt`:
   ```
   django==5.1
   opencv-python
   ffmpeg-python
   librosa
   numpy
   ```
4. **Apply Migrations**:
   ```bash
   python manage.py migrate
   ```
5. **Run the Server**:
   ```bash
   python manage.py runserver 9000
   ```

---

## Usage

1. Navigate to `http://127.0.0.1:9000/` in your browser.
2. Drag and drop or select a video file (MP4, AVI, MOV; max 100MB).
3. View the results, including metadata, visual/audio consistency, decision, and confidence score.

---

## Advanced Capabilities

### Current Implementation
- **Simplified Detection**: Uses face detection (Haar Cascade) and audio segment counting as proxies for emotion consistency.
- **Synchronous Processing**: Handles one video at a time with immediate results.

### Potential Enhancements
- **Machine Learning Integration**:
  - Replace Haar Cascade with a Convolutional Neural Network (CNN) like MTCNN or DeepFace for precise facial feature extraction.
  - Use pre-trained audio emotion classifiers (e.g., VGGish) for specific emotion detection.
- **Asynchronous Processing**:
  - Integrate Celery with Redis for background task processing, enabling multi-video analysis and progress updates via WebSockets.
- **Real-Time Progress**:
  - Poll `/status/` endpoint with AJAX to show percentage completion during analysis.
- **Scalability**:
  - Deploy with Docker, Gunicorn, and Nginx for production-grade performance.
  - Add database storage (e.g., PostgreSQL) for result persistence and historical analysis.

### Technical Metrics
- **Processing Time**: ~1-5 seconds for short videos (<30s) on a standard CPU.
- **Accuracy**: Heuristic-based (not ML-trained), suitable for basic detection; ML could improve precision to 90%+.

---

## Use Cases

1. **Content Moderation**:
   - Verify user-uploaded videos on social media platforms to prevent misinformation.
2. **Digital Forensics**:
   - Assist law enforcement in identifying manipulated evidence videos.
3. **Media Authentication**:
   - Validate news footage authenticity for journalists and broadcasters.
4. **Entertainment Industry**:
   - Ensure promotional content isn’t manipulated before release.
5. **Legal Proceedings**:
   - Provide expert analysis for court cases involving video evidence.

---

## Industries and Sectors

### 1. Technology Companies
- **Social Media Giants**: Meta, Twitter, TikTok need deepfake detection to maintain platform trust and comply with regulations (e.g., EU DSA).
- **Cloud Providers**: AWS, Google Cloud could integrate this into AI/ML services for enterprise clients.

### 2. Media and Journalism
- **News Outlets**: BBC, CNN require tools to verify sources and combat fake news.
- **Content Creators**: YouTubers, streamers benefit from ensuring content authenticity.

### 3. Government and Law Enforcement
- **Cybersecurity Agencies**: Detect state-sponsored disinformation campaigns.
- **Police Departments**: Validate CCTV or body cam footage.

### 4. Finance and Insurance
- **Banks**: Prevent video-based identity fraud in KYC processes.
- **Insurance Firms**: Verify claim videos for authenticity.

### 5. Healthcare
- **Telemedicine**: Ensure patient videos aren’t manipulated for fraudulent diagnoses.
- **Research**: Validate video data in behavioral studies.

---

## Competitors

### 1. Deepware Scanner
- **Overview**: Open-source deepfake detection tool using ML models.
- **Strengths**: High accuracy with pre-trained models, community-driven.
- **Weaknesses**: Requires technical expertise, no built-in UI.
- **Comparison**: Our system offers a user-friendly web interface but lacks ML sophistication.

### 2. Sentinel
- **Overview**: Commercial deepfake detection service for enterprises.
- **Strengths**: Advanced ML, cloud-based, API integration.
- **Weaknesses**: Costly, proprietary, less customizable.
- **Comparison**: Ours is open-source and locally deployable, ideal for custom needs.

### 3. Microsoft Video Authenticator
- **Overview**: AI-based tool for detecting deepfake signatures.
- **Strengths**: Backed by Microsoft’s AI research, high precision.
- **Weaknesses**: Limited to Windows ecosystems, not open-source.
- **Comparison**: Our system is platform-agnostic and extensible.

### 4. Sensity (formerly Deeptrace)
- **Overview**: AI-driven deepfake detection for businesses.
- **Strengths**: Real-time analysis, enterprise-grade.
- **Weaknesses**: Subscription-based, less transparency.
- **Comparison**: We offer a free, self-hosted alternative with potential for real-time upgrades.

---

## Future Enhancements

1. **ML Model Integration**:
   - Train a CNN on datasets like DFDC (Deepfake Detection Challenge) for higher accuracy.
2. **Multi-Modal Analysis**:
   - Add lip-sync detection (e.g., SyncNet) and temporal consistency checks.
3. **Scalability**:
   - Support batch processing and cloud deployment (e.g., AWS S3 for video storage).
4. **User Features**:
   - Add video preview, detailed analysis breakdowns, and exportable reports.
5. **Security**:
   - Implement rate limiting, authentication, and HTTPS for production use.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Notes for Contributors
- **Codebase**: Modular design allows easy swapping of detection algorithms or UI components.
- **Extensibility**: Add new features by extending `views.py` or integrating ML pipelines.
- **Community**: Contributions welcome for ML enhancements, UI improvements, or deployment scripts.
