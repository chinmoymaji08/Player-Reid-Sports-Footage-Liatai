#  Project Report – Player Re-Identification in Sports Footage

**Candidate**: Chinmoy Maji  
**Role**: AI Intern  
**Company**: Liat.ai

**Assignment for AI Intern Role at Liat.ai**  
This project implements a robust **Player Re-Identification System** using computer vision and deep learning. It supports both single-camera tracking and cross-camera player mapping in sports video footage.

---

##  Project Overview

Player Re-ID ensures that the same player retains a consistent ID:
-  Even after **temporarily going out of view**
-  Across **different camera angles**

###  Task Options Implemented
1. **Single-Camera Re-Identification**:  
   Detect and track players across time in a single video, re-assigning consistent IDs after players reappear.

2. **Cross-Camera Player Mapping**:  
   Match players from one camera (e.g., `broadcast.mp4`) to another (e.g., `tacticam.mp4`) with consistent ID mapping.

##  Objective

Develop a system that can:
1. Track players within a single video feed (single-camera ReID).
2. Map consistent player IDs across different camera feeds (cross-camera ReID).


##  Methodology

###  Object Detection
- Used YOLOv8 (custom fine-tuned model) to detect players (class 0).

###  Feature Extraction
- Built a ResNet18-based feature extractor to capture appearance embeddings.
- Added projection layers with L2 normalization for cosine similarity.

###  Re-Identification Strategy
- Combined **appearance similarity** (cosine) with **spatial features** (position, size).
- Used **Hungarian Algorithm** to match players between detections and tracks.
- Maintained ID history and handled disappearances using a simple track memory.

###  Cross-Camera Mapping
- Averaged embeddings per player from both feeds.
- Used cosine similarity + Hungarian matching to build one-to-one mappings.

---
