"""
Video Processor Module for VULCAN
Handles video frame extraction and preprocessing
"""

import cv2
import torch
from torchvision import transforms
from PIL import Image
from moviepy.editor import VideoFileClip
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Processes video files for AI model consumption
    Handles frame extraction, sampling, and preprocessing
    """
    
    def __init__(self, max_images_length: int = 45):
        """
        Initialize video processor
        
        Args:
            max_images_length: Maximum number of frames to extract
        """
        self.max_images_length = max_images_length
        self.transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        
    def extract_frames(self, video_path: str, sampling_strategy: str = "uniform") -> Tuple[List[torch.Tensor], List[Image.Image]]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            sampling_strategy: Strategy for frame sampling ("uniform", "keyframe", "adaptive")
            
        Returns:
            Tuple of (processed_frames, raw_frames)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.error(f"No frames found in video: {video_path}")
                return [], []
                
            sampling_interval = max(1, total_frames // self.max_images_length)
            
            processed_frames = []
            raw_frames = []
            frame_count = 0
            
            while cap.isOpened() and len(processed_frames) < self.max_images_length:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sampling_interval == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_frames.append(Image.fromarray(rgb_frame))
                    
                    # Process frame for model
                    pil_frame = self.transform(frame[:,:,::-1])
                    processed_frames.append(pil_frame)
                    
                frame_count += 1
                
            cap.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Extracted {len(processed_frames)} frames from {video_path}")
            return processed_frames, raw_frames
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return [], []
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            clip = VideoFileClip(video_path)
            info = {
                "duration": clip.duration,
                "fps": clip.fps,
                "resolution": (clip.w, clip.h),
                "total_frames": int(clip.duration * clip.fps)
            }
            clip.close()
            return info
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {str(e)}")
            return {}
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate if video file can be processed
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video is valid, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            return ret
        except:
            return False

class VideoPreprocessor:
    """
    Advanced video preprocessing utilities
    """
    
    @staticmethod
    def enhance_frame_quality(frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality using image processing techniques
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Enhanced frame
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    @staticmethod
    def detect_scene_changes(frames: List[np.ndarray], threshold: float = 0.3) -> List[int]:
        """
        Detect scene changes in video frames
        
        Args:
            frames: List of video frames
            threshold: Threshold for scene change detection
            
        Returns:
            List of frame indices where scene changes occur
        """
        scene_changes = [0]  # First frame is always a scene change
        
        for i in range(1, len(frames)):
            # Calculate histogram difference
            hist1 = cv2.calcHist([frames[i-1]], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frames[i]], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
            # Compare histograms
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            if correlation < (1 - threshold):
                scene_changes.append(i)
        
        return scene_changes
