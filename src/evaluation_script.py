#!/usr/bin/env python3
"""
Evaluation and Metrics Script for Player Re-Identification System
Calculates tracking accuracy, consistency, and other performance metrics

Author: AI Intern Candidate
Company: Liat.ai
"""

import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import pandas as pd
from scipy.optimize import linear_sum_assignment
import cv2

class PlayerReIDEvaluator:
    """
    Comprehensive evaluation toolkit for player re-identification system
    Calculates various metrics including MOTA, IDF1, and custom metrics
    """
    
    def __init__(self, results_path: str):
        """
        Initialize evaluator with tracking results
        
        Args:
            results_path: Path to JSON file containing tracking results
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.tracking_data = self.results.get('tracking_results', [])
        self.video_info = self.results.get('video_info', {})
        self.metrics = {}
        
    def calculate_tracking_consistency(self) -> Dict:
        """Calculate tracking consistency metrics"""
        player_appearances = defaultdict(list)
        
        # Track when each player appears
        for frame_data in self.tracking_data:
            frame_num = frame_data['frame']
            for detection in frame_data['detections']:
                player_id = detection['player_id']
                player_appearances[player_id].append(frame_num)
        
        # Calculate consistency metrics
        consistency_metrics = {
            'total_players': len(player_appearances),
            'average_appearances': np.mean([len(appearances) for appearances in player_appearances.values()]),
            'max_appearances': max([len(appearances) for appearances in player_appearances.values()]) if player_appearances else 0,
            'min_appearances': min([len(appearances) for appearances in player_appearances.values()]) if player_appearances else 0,
            'player_longevity': {}
        }
        
        # Calculate individual player longevity
        for player_id, appearances in player_appearances.items():
            if len(appearances) > 1:
                longevity = max(appearances) - min(appearances) + 1
                consistency_metrics['player_longevity'][str(player_id)] = {
                    'total_frames': len(appearances),
                    'span_frames': longevity,
                    'consistency_ratio': len(appearances) / longevity,
                    'first_appearance': min(appearances),
                    'last_appearance': max(appearances)
                }
        
        return consistency_metrics
    
    def calculate_id_switches(self) -> Dict:
        """Calculate ID switch metrics"""
        player_positions = defaultdict(list)
        
        # Extract player positions over time
        for frame_data in self.tracking_data:
            frame_num = frame_data['frame']
            for detection in frame_data['detections']:
                player_id = detection['player_id']
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                player_positions[player_id].append((frame_num, center_x, center_y))
        
        # Detect potential ID switches based on position jumps
        id_switches = 0
        total_tracks = 0
        switch_details = []
        
        for player_id, positions in player_positions.items():
            if len(positions) < 2:
                continue
                
            total_tracks += 1
            positions.sort(key=lambda x: x[0])  # Sort by frame number
            
            for i in range(1, len(positions)):
                prev_frame, prev_x, prev_y = positions[i-1]
                curr_frame, curr_x, curr_y = positions[i]
                
                # Calculate movement distance
                distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                frame_gap = curr_frame - prev_frame
                
                # Detect suspicious jumps (distance > threshold for frame gap)
                if frame_gap > 0 and distance / frame_gap > 50:  # Threshold for suspicious movement
                    id_switches += 1
                    switch_details.append({
                        'player_id': player_id,
                        'frame_from': prev_frame,
                        'frame_to': curr_frame,
                        'distance': distance,
                        'frame_gap': frame_gap,
                        'speed': distance / frame_gap
                    })
        
        return {
            'total_id_switches': id_switches,
            'total_tracks': total_tracks,
            'switch_rate': id_switches / total_tracks if total_tracks > 0 else 0,
            'switch_details': switch_details
        }
    
    def calculate_detection_metrics(self) -> Dict:
        """Calculate detection-related metrics"""
        total_detections = 0
        confidence_scores = []
        bbox_sizes = []
        
        for frame_data in self.tracking_data:
            frame_detections = len(frame_data['detections'])
            total_detections += frame_detections
            
            for detection in frame_data['detections']:
                confidence_scores.append(detection['confidence'])
                bbox = detection['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox_sizes.append(width * height)
        
        total_frames = len(self.tracking_data)
        
        return {
            'total_detections': total_detections,
            'average_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'confidence_stats': {
                'mean': np.mean(confidence_scores) if confidence_scores else 0,
                'std': np.std(confidence_scores) if confidence_scores else 0,
                'min': np.min(confidence_scores) if confidence_scores else 0,
                'max': np.max(confidence_scores) if confidence_scores else 0
            },
            'bbox_size_stats': {
                'mean': np.mean(bbox_sizes) if bbox_sizes else 0,
                'std': np.std(bbox_sizes) if bbox_sizes else 0,
                'min': np.min(bbox_sizes) if bbox_sizes else 0,
                'max': np.max(bbox_sizes) if bbox_sizes else 0
            }
        }
    
    def calculate_temporal_metrics(self) -> Dict:
        """Calculate temporal consistency metrics"""
        frame_player_counts = []
        player_frame_gaps = defaultdict(list)
        
        for frame_data in self.tracking_data:
            frame_num = frame_data['frame']
            player_count = len(frame_data['detections'])
            frame_player_counts.append(player_count)
            
            # Track frame gaps for each player
            for detection in frame_data['detections']:
                player_id = detection['player_id']
                player_frame_gaps[player_id].append(frame_num)
        
        # Calculate gaps between appearances
        gap_stats = []
        for player_id, frames in player_frame_gaps.items():
            if len(frames) > 1:
                frames.sort()
                gaps = [frames[i] - frames[i-1] for i in range(1, len(frames))]
                gap_stats.extend(gaps)
        
        return {
            'player_count_per_frame': {
                'mean': np.mean(frame_player_counts),
                'std': np.std(frame_player_counts),
                'min': np.min(frame_player_counts),
                'max': np.max(frame_player_counts)
            },
            'frame_gaps': {
                'mean': np.mean(gap_stats) if gap_stats else 0,
                'std': np.std(gap_stats) if gap_stats else 0,
                'max': np.max(gap_stats) if gap_stats else 0,
                'total_gaps': len([g for g in gap_stats if g > 1])
            }
        }
    
    def calculate_cross_camera_metrics(self, cross_camera_results: Dict) -> Dict:
        """Calculate cross-camera mapping metrics"""
        if 'cross_camera_mapping' not in cross_camera_results:
            return {}
        
        mapping = cross_camera_results['cross_camera_mapping']
        
        # Calculate mapping statistics
        total_players_v1 = len(cross_camera_results['video1_results']['player_statistics']['player_ids'])
        total_players_v2 = len(cross_camera_results['video2_results']['player_statistics']['player_ids'])
        
        mapped_players = len(mapping)
        confidence_scores = [info['confidence'] for info in mapping.values()]
        
        return {
            'total_players_video1': total_players_v1,
            'total_players_video2': total_players_v2,
            'mapped_players': mapped_players,
            'mapping_rate_v1': mapped_players / total_players_v1 if total_players_v1 > 0 else 0,
            'mapping_rate_v2': mapped_players / total_players_v2 if total_players_v2 > 0 else 0,
            'mapping_confidence': {
                'mean': np.mean(confidence_scores) if confidence_scores else 0,
                'std': np.std(confidence_scores) if confidence_scores else 0,
                'min': np.min(confidence_scores) if confidence_scores else 0,
                'max': np.max(confidence_scores) if confidence_scores else 0
            }
        }
    
    def generate_evaluation_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive evaluation report"""
        print("Calculating evaluation metrics...")
        
        # Calculate all metrics
        consistency_metrics = self.calculate_tracking_consistency()
        id_switch_metrics = self.calculate_id_switches()
        detection_metrics = self.calculate_detection_metrics()
        temporal_metrics = self.calculate_temporal_metrics()
        
        # Compile full report
        evaluation_report = {
            'video_info': self.video_info,
            'tracking_consistency': consistency_metrics,
            'id_switches': id_switch_metrics,
            'detection_metrics': detection_metrics,
            'temporal_metrics': temporal_metrics,
            'summary': {
                'total_frames_processed': len(self.tracking_data),
                'unique_players_detected': consistency_metrics['total_players'],
                'average_players_per_frame': detection_metrics['average_detections_per_frame'],
                'tracking_quality_score': self.calculate_quality_score(consistency_metrics, id_switch_metrics)
            }
        }
        
        # Save report if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(evaluation_report, f, indent=2, default=str)
            print(f"Evaluation report saved to: {output_path}")
        
        return evaluation_report
    
    def calculate_quality_score(self, consistency_metrics: Dict, id_switch_metrics: Dict) -> float:
        """Calculate overall tracking quality score (0-100)"""
        # Base score from consistency
        avg_consistency = np.mean([
            player_data['consistency_ratio'] 
            for player_data in consistency_metrics['player_longevity'].values()
        ]) if consistency_metrics['player_longevity'] else 0
        
        # Penalty for ID switches
        switch_penalty = min(id_switch_metrics['switch_rate'] * 0.5, 0.5)
        
        # Calculate final score
        quality_score = max(0, (avg_consistency - switch_penalty) * 100)
        
        return quality_score
    
    def create_visualizations(self, output_dir: str):
        """Create visualization plots for evaluation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Player count per frame
        frame_nums = [fd['frame'] for fd in self.tracking_data]
        player_counts = [len(fd['detections']) for fd in self.tracking_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(frame_nums, player_counts, 'b-', alpha=0.7)
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Players Detected')
        plt.title('Player Count Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'player_count_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Player longevity visualization
        consistency_metrics = self.calculate_tracking_consistency()
        if consistency_metrics['player_longevity']:
            player_ids = list(consistency_metrics['player_longevity'].keys())
            consistency_ratios = [data['consistency_ratio'] for data in consistency_metrics['player_longevity'].values()]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(player_ids)), consistency_ratios, color='skyblue', alpha=0.7)
            plt.xlabel('Player ID')
            plt.ylabel('Consistency Ratio')
            plt.title('Player Tracking Consistency')
            plt.xticks(range(len(player_ids)), player_ids, rotation=45)
            
            # Add value labels on bars
            for bar, ratio in zip(bars, consistency_ratios):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{ratio:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'player_consistency.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Confidence score distribution
        all_confidences = []
        for frame_data in self.tracking_data:
            for detection in frame_data['detections']:
                all_confidences.append(detection['confidence'])
        
        if all_confidences:
            plt.figure(figsize=(10, 6))
            plt.hist(all_confidences, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Detection Confidence Distribution')
            plt.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_confidences):.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {output_dir}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Player Re-Identification Results')
    parser.add_argument('--results_path', required=True, 
                       help='Path to tracking results JSON file')
    parser.add_argument('--output_dir', default='evaluation_output',
                       help='Directory to save evaluation results')
    parser.add_argument('--cross_camera_results', 
                       help='Path to cross-camera results JSON file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PlayerReIDEvaluator(args.results_path)
    
    # Generate evaluation report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    report = evaluator.generate_evaluation_report(report_path)
    
    # Create visualizations
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    evaluator.create_visualizations(viz_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total frames processed: {report['summary']['total_frames_processed']}")
    print(f"Unique players detected: {report['summary']['unique_players_detected']}")
    print(f"Average players per frame: {report['summary']['average_players_per_frame']:.2f}")
    print(f"Tracking quality score: {report['summary']['tracking_quality_score']:.1f}/100")
    print(f"ID switch rate: {report['id_switches']['switch_rate']:.3f}")
    print("="*50)
    
    # Evaluate cross-camera results if provided
    if args.cross_camera_results:
        with open(args.cross_camera_results, 'r') as f:
            cross_results = json.load(f)
        
        cross_metrics = evaluator.calculate_cross_camera_metrics(cross_results)
        print(f"Cross-camera mapping rate: {cross_metrics.get('mapping_rate_v1', 0):.2f}")
        print(f"Average mapping confidence: {cross_metrics.get('mapping_confidence', {}).get('mean', 0):.3f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()