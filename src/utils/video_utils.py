"""
Video Utilities for VULCAN
Common utility functions for video processing
"""

import os
import hashlib
import json
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoUtils:
    """Utility functions for video operations"""
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    
    @staticmethod
    def validate_video_file(file_path: str) -> bool:
        """
        Validate if file is a supported video format
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if valid video file, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in VideoUtils.SUPPORTED_FORMATS
    
    @staticmethod
    def get_video_hash(file_path: str) -> str:
        """
        Generate hash for video file (for caching purposes)
        
        Args:
            file_path: Path to video file
            
        Returns:
            SHA256 hash of the file
        """
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def get_safe_filename(original_name: str) -> str:
        """
        Create a safe filename for file operations
        
        Args:
            original_name: Original filename
            
        Returns:
            Safe filename with invalid characters removed
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        safe_name = original_name
        
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '_')
        
        # Remove extra whitespace and dots
        safe_name = safe_name.strip('. ')
        
        return safe_name
    
    @staticmethod
    def ensure_directory(dir_path: str) -> bool:
        """
        Ensure directory exists, create if it doesn't
        
        Args:
            dir_path: Path to directory
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            os.makedirs(dir_path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {dir_path}: {str(e)}")
            return False
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """
        Get file size in megabytes
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in MB
        """
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24):
        """
        Clean up temporary files older than specified age
        
        Args:
            temp_dir: Directory containing temporary files
            max_age_hours: Maximum age of files to keep (in hours)
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

class CacheManager:
    """Manages caching for processed videos and results"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = cache_dir
        VideoUtils.ensure_directory(cache_dir)
        self.cache_index_file = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict:
        """Load cache index from file"""
        try:
            if os.path.exists(self.cache_index_file):
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache index: {str(e)}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to file"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")
    
    def get_cache_key(self, video_path: str, question: str = None) -> str:
        """
        Generate cache key for video and question combination
        
        Args:
            video_path: Path to video file
            question: Optional question for QA caching
            
        Returns:
            Cache key string
        """
        video_hash = VideoUtils.get_video_hash(video_path)
        if question:
            question_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
            return f"{video_hash}_{question_hash}"
        return video_hash
    
    def cache_result(self, cache_key: str, result: Dict):
        """
        Cache processing result
        
        Args:
            cache_key: Unique cache key
            result: Result to cache
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Update cache index
            self.cache_index[cache_key] = {
                'file': cache_file,
                'timestamp': str(os.path.getmtime(cache_file))
            }
            self._save_cache_index()
            
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """
        Retrieve cached result
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached result or None if not found
        """
        try:
            if cache_key in self.cache_index:
                cache_file = self.cache_index[cache_key]['file']
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving cached result: {str(e)}")
        return None
    
    def clear_cache(self):
        """Clear all cached results"""
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.json'):
                        os.remove(os.path.join(root, file))
            
            self.cache_index = {}
            self._save_cache_index()
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

class PerformanceMonitor:
    """Monitor performance metrics for video processing"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics = {
            'total_videos_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def start_timer(self) -> float:
        """
        Start timing operation
        
        Returns:
            Start time
        """
        import time
        return time.time()
    
    def end_timer(self, start_time: float) -> float:
        """
        End timing operation and return duration
        
        Args:
            start_time: Start time from start_timer()
            
        Returns:
            Duration in seconds
        """
        import time
        return time.time() - start_time
    
    def record_processing(self, duration: float, success: bool = True):
        """
        Record processing metrics
        
        Args:
            duration: Processing duration in seconds
            success: Whether processing was successful
        """
        if success:
            self.metrics['total_videos_processed'] += 1
            self.metrics['total_processing_time'] += duration
            self.metrics['average_processing_time'] = (
                self.metrics['total_processing_time'] / 
                self.metrics['total_videos_processed']
            )
        else:
            self.metrics['errors'] += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics['cache_misses'] += 1
    
    def get_metrics(self) -> Dict:
        """
        Get current performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = (
            self.metrics['cache_hits'] / cache_requests 
            if cache_requests > 0 else 0
        )
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'total_cache_requests': cache_requests
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'total_videos_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
