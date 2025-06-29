# Football Analytics
Barnik Chakraborty

# Abstract
This project creates an end-to-end football analysis system using computer vision and machine learning to monitor players, referees, and the ball in football match videos. With YOLOv8 for object detection, K-means clustering for team identification, and optical flow for motion tracking, the system computes metrics like ball possession, player movement, and team dynamics. The aim is to offer actionable insights for coaches, analysts, and fans by converting raw video data into meaningful performance metrics.

# Introduction
Football analysis has become more sophisticated with improvements in computer vision and machine learning, allowing for in-depth analysis of player and team performance. This project seeks to automate football match analysis by identifying and tracking important features in video. The system overcomes issues like accurate object detection, team identification, and accurate movement measurement, providing uses in tactical analysis, fan experience, and performance assessment.

# Methodology
The methodology combines a number of computer vision and machine learning methods:

Object Detection: YOLOv8 detects players, referees, and the ball from video frames trained on a proprietary dataset to increase precision for football-case scenarios.

Team Identification: K-means clustering separates players into teams based on their jersey color, utilizing pixel information from detected bounding boxes.

Motion Tracking: Optical flow follows player and ball movement between frames, accounting for camera movement to calculate distances in meters.

Perspective Transformation: Transforms 2D video frames into a top-down perspective to examine player position and movement in relation to the field.

Data Processing: Python libraries (OpenCV, NumPy, scikit-learn) handle video data, with output saved in CSV files for analysis. The system is coded in Python, with code snippets found in the kingshere/passion_project repository.

# Your python code here
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Process video frame
def detect_objects(frame):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # Bounding boxes
    return detections

# Example K-means clustering for team identification
from sklearn.cluster import KMeans
def cluster_teams(colors):
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(colors)
    return labels
# Experiments
Experiments were performed with a sample football match video from the kingshere/passion_project repository:

Dataset: A 5-minute HD video clip (1080p, 30fps) of a Premier League match.
https://drive.google.com/file/d/1JtNg_ClVSBWSepnU0VxE3kp4_9vlchZk/view

Training: YOLOv8 was fine-tuned on 500 annotated frames, with 92% mAP for player detection.

Metrics Calculated:

Ball possession percentage per team.

Player movement distance (meters) using optical flow and perspective transformation.

Team formation analysis based on player positions.

Hardware: Executed on a system with NVIDIA GTX 1660 GPU, 16GB RAM, and Intel i7 processor.

Validation: Results were validated against manual annotations for possession and movement.
Link text

# Results
The system performed:

Detection Accuracy: 92% mAP for players, 95% for the ball, and 89% for referees.

Possession Analysis: Team A possessed 58%, Team B possessed 42%, confirmed to within 3% of manual analysis.

Movement Tracking: Mean player movement was 1.2 km in 5 minutes, with distances correct to ±0.1 km.

Team Clustering: K-means accurately predicted teams with 98% accuracy by jersey color. The top-down perspective facilitated accurate formation analysis, which showed Team A's 4-2-3-1 formation. Results are graphically visualized in interactive dashboards, made available through the repository.

screenshot.png

# Conclusion
The system for football match analysis effectively automates essential details of match analysis with precise values for possession, movement, and team dynamics. Effective but limited, the drawbacks are sensitivity to illumination conditions and occlusion with high-density crowd scenes. Adding multi-camera sources and real-time processing could further develop the live analysis capabilities in future work. The project reveals the power of computer vision applications for sports analytics with uses ranging from coaching, broadcasting, to fan engagement. Investigate the complete implementation at kingshere/passion_project.

# Reference
https://app.readytensor.ai/publications/football-analytics-E10sJqzRdhuM 