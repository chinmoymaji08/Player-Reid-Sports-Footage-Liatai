#!/usr/bin/env python3
"""
Demo Script for Player Re-Identification System
Demonstrates both single-camera and cross-camera functionality

Author: AI Intern Candidate  
Company: Liat.ai
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerReIDDemo:
    """
    Demonstration class for Player Re-ID system
    Handles setup, execution, and result presentation
    """
    
    def __init__(self, model_path: str, videos_dir: str = "videos"):
        self.model_path = model_path
        self.videos_dir = Path(videos_dir)
        self.output_dir = Path("demo_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Expected video files based on assignment
        self.video_files = {
            'single': '15sec_input_720p.mp4',
            'broadcast': 'broadcast.mp4', 
            'tacticam': 'tacticam.mp4'
        }
        
    def check_requirements(self) -> bool:
        """Check if all required files exist"""
        logger.info("Checking requirements...")
        
        # Check model file
        if not os.path.exists(self.model_path):
            logger.error(f"YOLO model not found at: {self.model_path}")
            return False
        
        # Check video files
        missing_videos = []
        for name, filename in self.video_files.items():
            video_path = self.videos_dir / filename
            if not video_path.exists():
                missing_videos.append(filename)
        
        if missing_videos:
            logger.error(f"Missing video files: {missing_videos}")
            logger.error(f"Please place videos in: {self.videos_dir}")
            return False
        
        logger.info("All requirements satisfied!")
        return True
    
    def run_single_camera_demo(self) -> str:
        """Run single-camera re-identification demo"""
        logger.info("Running single-camera demo (Option 2)...")
        
        input_video = self.videos_dir / self.video_files['single']
        output_video = self.output_dir / "single_camera_result.mp4"
        
        cmd = [
            sys.executable, "src/player_reid.py",
            "--mode", "single",
            "--model_path", self.model_path,
            "--input1", str(input_video),
            "--output", str(output_video)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Single-camera demo completed successfully!")
            return str(output_video).replace('.mp4', '_results.json')
        except subprocess.CalledProcessError as e:
            logger.error(f"Single-camera demo failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return None
    
    def run_cross_camera_demo(self) -> str:
        """Run cross-camera mapping demo"""
        logger.info("Running cross-camera demo (Option 1)...")
        
        broadcast_video = self.videos_dir / self.video_files['broadcast']
        tacticam_video = self.videos_dir / self.video_files['tacticam']
        cross_output_dir = self.output_dir / "cross_camera_results"
        
        cmd = [
            sys.executable, "src/player_reid.py",
            "--mode", "cross",
            "--model_path", self.model_path,
            "--input1", str(broadcast_video),
            "--input2", str(tacticam_video),
            "--output", str(cross_output_dir)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Cross-camera demo completed successfully!")
            return str(cross_output_dir / "cross_camera_results.json")
        except subprocess.CalledProcessError as e:
            logger.error(f"Cross-camera demo failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return None
    
    def run_evaluation(self, results_path: str, cross_results_path: str = None):
        """Run evaluation on results"""
        logger.info("Running evaluation...")
        
        eval_output_dir = self.output_dir / "evaluation"
        
        cmd = [
            sys.executable, "src/evaluation_script.py",
            "--results_path", results_path,
            "--output_dir", str(eval_output_dir)
        ]
        
        if cross_results_path:
            cmd.extend(["--cross_camera_results", cross_results_path])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Evaluation completed successfully!")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(f"Error output: {e.stderr}")

    def create_demo_report(self, single_results: str = None, cross_results: str = None):
        """Create comprehensive demo report"""
        logger.info("Creating demo report...")
        
        report = {
            "demo_info": {
                "timestamp": str(datetime.now()),
                "model_used": self.model_path,
                "videos_processed": []
            },
            "results_summary": {},
            "file_locations": {
                "output_directory": str(self.output_dir),
                "processed_videos": [],
                "evaluation_results": str(self.output_dir / "evaluation")
            }
        }
        
        if single_results and os.path.exists(single_results):
            with open(single_results, 'r') as f:
                single_data = json.load(f)
            
            report["results_summary"]["single_camera"] = {
                "video_processed": self.video_files['single'],
                "total_players": single_data.get('player_statistics', {}).get('total_players', 0),
                "total_frames": single_data.get('video_info', {}).get('total_frames', 0),
                "output_video": str(self.output_dir / "single_camera_result.mp4")
            }
            
            report["demo_info"]["videos_processed"].append(self.video_files['single'])
            report["file_locations"]["processed_videos"].append(str(self.output_dir / "single_camera_result.mp4"))
        
        if cross_results and os.path.exists(cross_results):
            with open(cross_results, 'r') as f:
                cross_data = json.load(f)
            
            mapping = cross_data.get('cross_camera_mapping', {})
            
            report["results_summary"]["cross_camera"] = {
                "videos_processed": [self.video_files['broadcast'], self.video_files['tacticam']],
                "players_mapped": len(mapping),
                "mapping_confidence": {
                    "average": sum([info['confidence'] for info in mapping.values()]) / len(mapping) if mapping else 0,
                    "details": mapping
                },
                "output_videos": [
                    str(self.output_dir / "cross_camera_results" / "video1_tracked.mp4"),
                    str(self.output_dir / "cross_camera_results" / "video2_tracked.mp4")
                ]
            }
            
            report["demo_info"]["videos_processed"].extend([self.video_files['broadcast'], self.video_files['tacticam']])
            report["file_locations"]["processed_videos"].extend([
                str(self.output_dir / "cross_camera_results" / "video1_tracked.mp4"),
                str(self.output_dir / "cross_camera_results" / "video2_tracked.mp4")
            ])
        
        report_path = self.output_dir / "demo_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Demo report saved to: {report_path}")
        return report_path

    def print_summary(self, report_path: str):
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print("\n" + "="*60)
        print("PLAYER RE-IDENTIFICATION DEMO SUMMARY")
        print("="*60)
        
        print(f"Model used: {report['demo_info']['model_used']}")
        print(f"Videos processed: {len(report['demo_info']['videos_processed'])}")
        
        if 'single_camera' in report['results_summary']:
            single = report['results_summary']['single_camera']
            print(f"\nSingle-Camera Results:")
            print(f"  - Video: {single['video_processed']}")
            print(f"  - Players detected: {single['total_players']}")
            print(f"  - Frames processed: {single['total_frames']}")
            print(f"  - Output: {single['output_video']}")
        
        if 'cross_camera' in report['results_summary']:
            cross = report['results_summary']['cross_camera']
            print(f"\nCross-Camera Results:")
            print(f"  - Videos: {', '.join(cross['videos_processed'])}")
            print(f"  - Players mapped: {cross['players_mapped']}")
            print(f"  - Average mapping confidence: {cross['mapping_confidence']['average']:.3f}")
            print(f"  - Output videos: {len(cross['output_videos'])} files")
        
        print(f"\nAll results saved to: {report['file_locations']['output_directory']}")
        print(f"Evaluation results: {report['file_locations']['evaluation_results']}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Player Re-Identification Demo')
    parser.add_argument('--model_path', required=True,
                       help='Path to YOLO model file')
    parser.add_argument('--videos_dir', default='videos',
                       help='Directory containing video files')
    parser.add_argument('--mode', choices=['single', 'cross', 'both'], default='both',
                       help='Demo mode to run')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation step')
    
    args = parser.parse_args()
    demo = PlayerReIDDemo(args.model_path, args.videos_dir)
    
    if not demo.check_requirements():
        logger.error("Requirements check failed. Please fix issues and try again.")
        return 1
    
    single_results = None
    cross_results = None

    if args.mode in ['single', 'both']:
        single_results = demo.run_single_camera_demo()

    if args.mode in ['cross', 'both']:
        cross_results = demo.run_cross_camera_demo()

    if not args.skip_eval:
        if single_results:
            demo.run_evaluation(single_results, cross_results)
        elif cross_results:
            demo.run_evaluation(cross_results)

    report_path = demo.create_demo_report(single_results, cross_results)
    demo.print_summary(report_path)

    logger.info("Demo completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())