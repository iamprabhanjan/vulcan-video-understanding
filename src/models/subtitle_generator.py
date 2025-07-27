"""
Subtitle Generator Module for VULCAN
Handles subtitle generation and processing using Whisper
"""

import os
import webvtt
import soundfile as sf
import moviepy.editor as mp
from typing import Optional, Dict, List
import logging
import whisper
import tempfile

logger = logging.getLogger(__name__)

class SubtitleGenerator:
    """
    Generates and processes subtitles for video content
    """
    
    def __init__(self, model_size: str = "base", language: str = "english"):
        """
        Initialize subtitle generator
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            language: Language for subtitle generation
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self.cache_dir = "workspace/inference_subtitles"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(f"{self.cache_dir}/mp3", exist_ok=True)
        
    def load_model(self):
        """Load Whisper model for subtitle generation"""
        if self.model is None:
            try:
                self.model = whisper.load_model(self.model_size)
                logger.info(f"Loaded Whisper model: {self.model_size}")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {str(e)}")
                raise
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video
            audio_path: Path for output audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            video_clip = mp.VideoFileClip(video_path)
            if video_clip.audio is None:
                logger.warning(f"No audio track found in {video_path}")
                video_clip.close()
                return False
                
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(
                audio_path, 
                codec="libmp3lame", 
                bitrate="320k",
                verbose=False,
                logger=None
            )
            audio_clip.close()
            video_clip.close()
            logger.info(f"Audio extracted successfully to {audio_path}")
            return True
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {str(e)}")
            return False
    
    def generate_subtitles(self, video_path: str, force_regenerate: bool = False) -> Optional[str]:
        """
        Generate subtitles for video using Whisper
        
        Args:
            video_path: Path to video file
            force_regenerate: Force regeneration even if subtitles exist
            
        Returns:
            Path to generated subtitle file or None if failed
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        subtitle_path = os.path.join(self.cache_dir, f"{video_id}.vtt")
        audio_path = os.path.join(self.cache_dir, "mp3", f"{video_id}.mp3")
        
        # Check if subtitles already exist
        if os.path.exists(subtitle_path) and not force_regenerate:
            logger.info(f"Using existing subtitles: {subtitle_path}")
            return subtitle_path
        
        try:
            # Load model if not already loaded
            self.load_model()
            
            # Extract audio
            if not self.extract_audio(video_path, audio_path):
                return None
            
            # Generate subtitles using Whisper
            logger.info("Generating subtitles using Whisper...")
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                verbose=False
            )
            
            # Save as VTT format
            self._save_as_vtt(result, subtitle_path)
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info(f"Subtitles generated successfully: {subtitle_path}")
            return subtitle_path
            
        except Exception as e:
            logger.error(f"Error generating subtitles for {video_path}: {str(e)}")
            # Clean up on error
            for temp_file in [audio_path, subtitle_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return None
    
    def _save_as_vtt(self, transcription_result: dict, output_path: str):
        """
        Save Whisper transcription result as VTT file
        
        Args:
            transcription_result: Whisper transcription result
            output_path: Path to save VTT file
        """
        with open(output_path, 'w', encoding='utf-8') as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            
            for segment in transcription_result['segments']:
                start_time = self._format_timestamp(segment['start'])
                end_time = self._format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                vtt_file.write(f"{start_time} --> {end_time}\n")
                vtt_file.write(f"{text}\n\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for VTT format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

class SubtitleProcessor:
    """
    Processes existing subtitle files
    """
    
    @staticmethod
    def load_subtitles(subtitle_path: str) -> Optional[List[Dict]]:
        """
        Load subtitles from VTT file
        
        Args:
            subtitle_path: Path to subtitle file
            
        Returns:
            List of subtitle dictionaries or None if failed
        """
        try:
            vtt_file = webvtt.read(subtitle_path)
            subtitles = []
            
            for caption in vtt_file:
                subtitles.append({
                    'start': caption.start_in_seconds,
                    'end': caption.end_in_seconds,
                    'text': caption.text.replace('\n', ' ').strip()
                })
            
            logger.info(f"Loaded {len(subtitles)} subtitle segments")
            return subtitles
            
        except Exception as e:
            logger.error(f"Error loading subtitles from {subtitle_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_subtitle_for_timeframe(subtitles: List[Dict], start_time: float, end_time: float) -> str:
        """
        Get subtitle text for specific timeframe
        
        Args:
            subtitles: List of subtitle dictionaries
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Combined subtitle text for timeframe
        """
        relevant_text = []
        
        for subtitle in subtitles:
            # Check if subtitle overlaps with timeframe
            if (subtitle['start'] <= end_time and subtitle['end'] >= start_time):
                if subtitle['text'] not in relevant_text:
                    relevant_text.append(subtitle['text'])
        
        return ' '.join(relevant_text)
    
    @staticmethod
    def filter_subtitles(subtitles: List[Dict], max_length: int = 400) -> List[Dict]:
        """
        Filter subtitles to meet length constraints
        
        Args:
            subtitles: List of subtitle dictionaries
            max_length: Maximum total character length
            
        Returns:
            Filtered subtitle list
        """
        filtered = []
        total_length = 0
        
        for subtitle in subtitles:
            text_length = len(subtitle['text'])
            if total_length + text_length <= max_length:
                filtered.append(subtitle)
                total_length += text_length
            else:
                break
        
        return filtered
