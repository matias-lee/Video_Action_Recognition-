"""
Module: preprocess.py

This script processes raw video files from the UCF50 dataset, applying uniform 
random sampling to extract a fixed number of frames per video, and saves them 
as JPEG images for training the LRCN model.
"""

import os
import glob
from tqdm import tqdm
from utils import get_frames, store_frames

class VideoPreprocessor:
    """
    Handles the extraction and storage of frames from a dataset of raw videos.
    """
    def __init__(self, raw_data_dir, output_dir, n_frames=16):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.n_frames = n_frames

    def process_dataset(self):
        """
        Iterates through the UCF50 action classes, processes each video, 
        and saves the sampled frames to the output directory.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get all action class directories (e.g., BaseballPitch, Biking, etc.)
        action_classes = [d for d in os.listdir(self.raw_data_dir) 
                          if os.path.isdir(os.path.join(self.raw_data_dir, d))]

        print(f"Found {len(action_classes)} action classes. Beginning extraction...")

        for action in tqdm(action_classes, desc="Processing Classes"):
            raw_action_path = os.path.join(self.raw_data_dir, action)
            out_action_path = os.path.join(self.output_dir, action)
            os.makedirs(out_action_path, exist_ok=True)

            # Process each .avi video in the action class
            video_files = glob.glob(os.path.join(raw_action_path, '*.avi'))
            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                out_video_dir = os.path.join(out_action_path, video_name)
                
                # Skip if already processed
                if os.path.exists(out_video_dir) and len(os.listdir(out_video_dir)) >= self.n_frames:
                    continue
                
                os.makedirs(out_video_dir, exist_ok=True)
                
                # Extract frames using the refactored uniform random sampling
                frames, _ = get_frames(video_file, n_frames=self.n_frames)
                
                if frames:
                    store_frames(frames, out_video_dir)

if __name__ == "__main__":
    # Point these to the correct directories on your cluster
    RAW_DIR = "./data/UCF50"
    OUT_DIR = "./data/UCF50_frames"
    FRAMES_PER_VIDEO = 16  # You can adjust this hyperparameter
    
    preprocessor = VideoPreprocessor(RAW_DIR, OUT_DIR, n_frames=FRAMES_PER_VIDEO)
    preprocessor.process_dataset()