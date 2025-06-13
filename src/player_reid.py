#!/usr/bin/env python3
"""
Player Re-Identification System for Sports Footage
Liat.ai Assignment - AI Intern Role

This system implements player re-identification using deep learning techniques
for both single-camera tracking and cross-camera mapping scenarios.

Author: AI Intern Candidate
Company: Liat.ai
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
import json
import os
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import argparse
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerFeatureExtractor(nn.Module):
    """
    Deep learning model for extracting player appearance features
    Uses ResNet-based architecture for robust feature representation
    """
    
    def __init__(self, feature_dim=512):
        super(PlayerFeatureExtractor, self).__init__()

        
        # Use ResNet18 as backbone
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add custom feature projection layer
        self.feature_projector = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim),
            L2Norm(dim=1)  # L2 normalize for cosine similarity
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.feature_projector(features)

class L2Norm(nn.Module):
    """L2 normalization layer"""
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim)

class PlayerTracker:
    """
    Main class for player tracking and re-identification
    Handles both single-camera and cross-camera scenarios
    """
    
    def __init__(self, model_path: str, feature_dim: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(model_path)
        self.feature_extractor = PlayerFeatureExtractor(feature_dim).to(self.device)
        self.feature_extractor.eval()
        
        # Tracking parameters
        self.max_disappeared = 30  # Max frames a player can be missing
        self.max_distance = 0.5    # Max feature distance for matching
        self.similarity_threshold = 0.7
        
        # Data structures for tracking
        self.player_features = {}  # player_id -> feature history
        self.player_positions = {}  # player_id -> position history
        self.player_disappeared = defaultdict(int)
        self.next_player_id = 1
        
        # Transform for feature extraction
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_player_features(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract appearance features from player crop"""
        x1, y1, x2, y2 = bbox
        
        # Crop player region with some padding
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        player_crop = image[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return np.zeros(512)
        
        # Preprocess and extract features
        tensor_input = self.transform(player_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(tensor_input)
            return features.squeeze().cpu().numpy()
    
    def calculate_spatial_features(self, bbox: List[int], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Calculate spatial features (position, size, aspect ratio)"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape
        
        # Normalize coordinates
        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        aspect_ratio = width / (height + 1e-6)
        
        return np.array([center_x, center_y, width, height, aspect_ratio])
    
    def match_players(self, current_detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Match current detections with existing player tracks"""
        if not current_detections:
            return []
        
        # Extract features for current detections
        current_features = []
        current_spatial = []
        
        for detection in current_detections:
            bbox = detection['bbox']
            
            # Appearance features
            app_features = self.extract_player_features(frame, bbox)
            current_features.append(app_features)
            
            # Spatial features
            spatial_features = self.calculate_spatial_features(bbox, frame.shape[:2])
            current_spatial.append(spatial_features)
        
        current_features = np.array(current_features)
        current_spatial = np.array(current_spatial)
        
        # If no existing tracks, assign new IDs
        if not self.player_features:
            for i, detection in enumerate(current_detections):
                player_id = self.next_player_id
                self.next_player_id += 1
                
                detection['player_id'] = player_id
                self.player_features[player_id] = deque([current_features[i]], maxlen=10)
                self.player_positions[player_id] = deque([current_spatial[i]], maxlen=10)
                
            return current_detections
        
        # Calculate similarity matrix
        existing_ids = list(self.player_features.keys())
        similarity_matrix = np.zeros((len(current_detections), len(existing_ids)))
        
        for i, curr_feat in enumerate(current_features):
            for j, player_id in enumerate(existing_ids):
                # Appearance similarity
                hist_features = np.array(list(self.player_features[player_id]))
                app_sim = np.max(cosine_similarity([curr_feat], hist_features)[0])
                
                # Spatial similarity (distance between positions)
                hist_positions = np.array(list(self.player_positions[player_id]))
                pos_dist = np.min(np.linalg.norm(
                    current_spatial[i][:2] - hist_positions[:, :2], axis=1
                ))
                spatial_sim = np.exp(-pos_dist * 5)  # Convert distance to similarity
                
                # Combined similarity
                similarity_matrix[i, j] = 0.7 * app_sim + 0.3 * spatial_sim
        
        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
        
        # Assign IDs based on matching
        matched_detections = []
        used_ids = set()
        
        for row_idx, col_idx in zip(row_indices, col_indices):
            if similarity_matrix[row_idx, col_idx] > self.similarity_threshold:
                player_id = existing_ids[col_idx]
                current_detections[row_idx]['player_id'] = player_id
                
                # Update feature and position history
                self.player_features[player_id].append(current_features[row_idx])
                self.player_positions[player_id].append(current_spatial[row_idx])
                self.player_disappeared[player_id] = 0
                
                matched_detections.append(current_detections[row_idx])
                used_ids.add(player_id)
        
        # Assign new IDs to unmatched detections
        for i, detection in enumerate(current_detections):
            if 'player_id' not in detection:
                player_id = self.next_player_id
                self.next_player_id += 1
                
                detection['player_id'] = player_id
                self.player_features[player_id] = deque([current_features[i]], maxlen=10)
                self.player_positions[player_id] = deque([current_spatial[i]], maxlen=10)
                
                matched_detections.append(detection)
        
        # Update disappeared counter for unmatched tracks
        for player_id in existing_ids:
            if player_id not in used_ids:
                self.player_disappeared[player_id] += 1
                
                # Remove tracks that have been missing too long
                if self.player_disappeared[player_id] > self.max_disappeared:
                    del self.player_features[player_id]
                    del self.player_positions[player_id]
                    del self.player_disappeared[player_id]
        
        return matched_detections
    
    def process_single_video(self, video_path: str, output_path: str) -> Dict:
        """Process single video for player re-identification"""
        logger.info(f"Processing single video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            # Extract player detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Assuming class 0 is player (adjust based on your model)
                        if class_id == 0 and confidence > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'frame': int(frame_count)
                            })
            
            # Match players and assign IDs
            matched_detections = self.match_players(detections, frame)
            
            # Draw results on frame
            for detection in matched_detections:
                x1, y1, x2, y2 = detection['bbox']
                player_id = detection['player_id']
                confidence = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw player ID
                label = f"Player {player_id}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out.write(frame)
            
            # Store frame results
            frame_results.append({
                'frame': frame_count,
                'detections': matched_detections,
                'timestamp': frame_count / fps
            })
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # Save results
        results_dict = {
            'video_info': {
                'path': video_path,
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            },
            'tracking_results': frame_results,
            'player_statistics': self.get_player_statistics()
        }
        
        return results_dict
    
    def process_cross_camera(self, video1_path: str, video2_path: str, 
                           output_dir: str) -> Dict:
        """Process two videos for cross-camera player mapping"""
        logger.info(f"Processing cross-camera mapping: {video1_path} and {video2_path}")
        
        # Process both videos independently first
        results1 = self.process_single_video(video1_path, 
                                           os.path.join(output_dir, 'video1_tracked.mp4'))
        
        # Reset tracker for second video
        self.reset_tracker()
        
        results2 = self.process_single_video(video2_path, 
                                           os.path.join(output_dir, 'video2_tracked.mp4'))
        
        # Extract player features from both videos for mapping
        mapping = self.create_cross_camera_mapping(results1, results2, 
                                                  video1_path, video2_path)
        
        return {
            'video1_results': results1,
            'video2_results': results2,
            'cross_camera_mapping': mapping
        }
    
    def create_cross_camera_mapping(self, results1: Dict, results2: Dict,
                                   video1_path: str, video2_path: str) -> Dict:
        """Create mapping between players across cameras"""
        logger.info("Creating cross-camera player mapping")
        
        # Extract representative features for each player from both videos
        player_features_v1 = self.extract_representative_features(results1, video1_path)
        player_features_v2 = self.extract_representative_features(results2, video2_path)
        
        # Calculate similarity matrix between all player pairs
        v1_ids = list(player_features_v1.keys())
        v2_ids = list(player_features_v2.keys())
        
        similarity_matrix = np.zeros((len(v1_ids), len(v2_ids)))
        
        for i, id1 in enumerate(v1_ids):
            for j, id2 in enumerate(v2_ids):
                sim = cosine_similarity([player_features_v1[id1]], 
                                      [player_features_v2[id2]])[0][0]
                similarity_matrix[i, j] = sim
        
        # Hungarian algorithm for optimal mapping
        row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
        
        # Create mapping with confidence scores
        mapping = {}
        for row_idx, col_idx in zip(row_indices, col_indices):
            confidence = similarity_matrix[row_idx, col_idx]
            if confidence > 0.6:  # Minimum confidence threshold
                mapping[v1_ids[row_idx]] = {
                    'video2_id': v2_ids[col_idx],
                    'confidence': confidence
                }
        
        return mapping
    
    def extract_representative_features(self, results: Dict, video_path: str) -> Dict:
        """Extract representative features for each player"""
        cap = cv2.VideoCapture(video_path)
        player_features = defaultdict(list)
        
        for frame_result in results['tracking_results']:
            frame_num = frame_result['frame']
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            for detection in frame_result['detections']:
                player_id = detection['player_id']
                bbox = detection['bbox']
                
                # Extract features
                features = self.extract_player_features(frame, bbox)
                player_features[player_id].append(features)
        
        cap.release()
        
        # Calculate representative feature for each player (mean)
        representative_features = {}
        for player_id, features_list in player_features.items():
            if features_list:
                representative_features[player_id] = np.mean(features_list, axis=0)
        
        return representative_features
    
    def get_player_statistics(self) -> Dict:
        """Get statistics about tracked players"""
        stats = {
            'total_players': len(self.player_features),
            'active_players': len([pid for pid, disappeared in self.player_disappeared.items() 
                                 if disappeared < self.max_disappeared]),
            'player_ids': list(self.player_features.keys())
        }
        return stats
    
    def reset_tracker(self):
        """Reset tracker state for processing new video"""
        self.player_features.clear()
        self.player_positions.clear()
        self.player_disappeared.clear()
        self.next_player_id = 1

def main():
    parser = argparse.ArgumentParser(description='Player Re-Identification System')
    parser.add_argument('--mode', choices=['single', 'cross'], required=True,
                       help='Processing mode: single video or cross-camera')
    parser.add_argument('--model_path', required=True,
                       help='Path to YOLO model file')
    parser.add_argument('--input1', required=True,
                       help='Path to first input video')
    parser.add_argument('--input2', 
                       help='Path to second input video (for cross-camera mode)')
    parser.add_argument('--output', required=True,
                       help='Output directory or file path')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = PlayerTracker(args.model_path)
    
    if args.mode == 'single':
        # Single video processing
        results = tracker.process_single_video(args.input1, args.output)
        
        # Save results
        results_path = args.output.replace('.mp4', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Single video processing complete. Results saved to {results_path}")
        
    elif args.mode == 'cross':
        if not args.input2:
            raise ValueError("Cross-camera mode requires two input videos")
        
        # Cross-camera processing
        os.makedirs(args.output, exist_ok=True)
        results = tracker.process_cross_camera(args.input1, args.input2, args.output)
        
        # Save results
        results_path = os.path.join(args.output, 'cross_camera_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Cross-camera processing complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
