"""
Gradio Web Interface for VULCAN
User-friendly web interface for video question answering
"""

import gradio as gr
import os
import logging
from typing import Optional, Tuple
import tempfile
import shutil

# Import VULCAN modules
from ..models.video_processor import VideoProcessor
from ..models.subtitle_generator import SubtitleGenerator
from ..models.question_answerer import QuestionAnswerer
from ..utils.config import Config
from ..utils.video_utils import VideoUtils, CacheManager, PerformanceMonitor

logger = logging.getLogger(__name__)

class VulcanInterface:
    """Main interface class for VULCAN web application"""
    
    def __init__(self, config: Config):
        """
        Initialize VULCAN interface
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.video_processor = VideoProcessor(
            max_images_length=config.get('model.max_images_length', 45)
        )
        self.subtitle_generator = SubtitleGenerator(
            model_size=config.get('whisper.model_size', 'base'),
            language=config.get('whisper.language', 'english')
        )
        self.question_answerer = QuestionAnswerer(config.get_model_config())
        self.cache_manager = CacheManager(config.get('processing.cache_dir', 'cache'))
        self.performance_monitor = PerformanceMonitor()
        
        # Load models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            logger.info("Initializing VULCAN models...")
            
            # Load question answering model
            model_path = self.config.get('model.checkpoint_path')
            config_path = self.config.get('model.config_path')
            
            if model_path and config_path:
                self.question_answerer.load_model(model_path, config_path)
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def process_video_question(self, video_file, question: str) -> str:
        """
        Process video and question to generate answer
        
        Args:
            video_file: Uploaded video file
            question: User question
            
        Returns:
            Generated answer
        """
        try:
            if video_file is None:
                return "Please upload a video file."
            
            if not question or question.strip() == "":
                return "Please enter a question about the video."
            
            # Get video path
            video_path = video_file if isinstance(video_file, str) else video_file.name
            
            # Validate video file
            if not VideoUtils.validate_video_file(video_path):
                return "Invalid video file format. Please upload a supported video file."
            
            # Check file size
            file_size_mb = VideoUtils.get_file_size_mb(video_path)
            max_size_mb = self.config.get('security.max_file_size_mb', 500)
            
            if file_size_mb > max_size_mb:
                return f"File too large. Maximum size is {max_size_mb}MB."
            
            # Check cache first
            cache_key = self.cache_manager.get_cache_key(video_path, question)
            cached_result = self.cache_manager.get_cached_result(cache_key)
            
            if cached_result and self.config.get('processing.enable_caching', True):
                self.performance_monitor.record_cache_hit()
                logger.info(f"Using cached result for question: {question[:50]}...")
                return cached_result.get('answer', 'Cached result corrupted')
            
            self.performance_monitor.record_cache_miss()
            
            # Start processing timer
            start_time = self.performance_monitor.start_timer()
            
            # Process video
            logger.info(f"Processing video: {os.path.basename(video_path)}")
            
            # Extract frames
            processed_frames, raw_frames = self.video_processor.extract_frames(video_path)
            
            if not processed_frames:
                return "Error: Could not extract frames from video. Please check the video file."
            
            # Generate or load subtitles
            subtitle_text = ""
            if self.config.get('processing.add_subtitles', True):
                subtitle_path = self.subtitle_generator.generate_subtitles(video_path)
                if subtitle_path:
                    from ..models.subtitle_generator import SubtitleProcessor
                    subtitles = SubtitleProcessor.load_subtitles(subtitle_path)
                    if subtitles:
                        # Get subtitle text for the video duration
                        video_info = self.video_processor.get_video_info(video_path)
                        duration = video_info.get('duration', 0)
                        subtitle_text = SubtitleProcessor.get_subtitle_for_timeframe(
                            subtitles, 0, duration
                        )
            
            # Process question
            import torch
            frames_tensor = torch.stack(processed_frames) if processed_frames else torch.empty(0)
            answer = self.question_answerer.process_question(
                frames_tensor, subtitle_text, question
            )
            
            # Record processing time
            duration = self.performance_monitor.end_timer(start_time)
            self.performance_monitor.record_processing(duration, True)
            
            # Cache result
            if self.config.get('processing.enable_caching', True):
                result = {
                    'answer': answer,
                    'video_path': video_path,
                    'question': question,
                    'processing_time': duration
                }
                self.cache_manager.cache_result(cache_key, result)
            
            logger.info(f"Question processed successfully in {duration:.2f}s")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing video question: {str(e)}")
            self.performance_monitor.record_processing(0, False)
            return f"Error processing your request: {str(e)}"
    
    def get_video_info(self, video_file) -> str:
        """
        Get information about uploaded video
        
        Args:
            video_file: Uploaded video file
            
        Returns:
            Video information string
        """
        try:
            if video_file is None:
                return "No video uploaded"
            
            video_path = video_file if isinstance(video_file, str) else video_file.name
            info = self.video_processor.get_video_info(video_path)
            
            if info:
                return f"""
**Video Information:**
- Duration: {info.get('duration', 0):.1f} seconds
- FPS: {info.get('fps', 0):.1f}
- Resolution: {info.get('resolution', (0, 0))[0]}x{info.get('resolution', (0, 0))[1]}
- Total Frames: {info.get('total_frames', 0)}
- File Size: {VideoUtils.get_file_size_mb(video_path):.1f} MB
"""
            else:
                return "Could not retrieve video information"
                
        except Exception as e:
            return f"Error getting video info: {str(e)}"
    
    def clear_cache(self) -> str:
        """Clear processing cache"""
        try:
            self.cache_manager.clear_cache()
            return "Cache cleared successfully!"
        except Exception as e:
            return f"Error clearing cache: {str(e)}"
    
    def get_performance_stats(self) -> str:
        """Get performance statistics"""
        try:
            metrics = self.performance_monitor.get_metrics()
            return f"""
**Performance Statistics:**
- Videos Processed: {metrics['total_videos_processed']}
- Average Processing Time: {metrics['average_processing_time']:.2f}s
- Cache Hit Rate: {metrics['cache_hit_rate']:.1%}
- Total Errors: {metrics['errors']}
"""
        except Exception as e:
            return f"Error getting stats: {str(e)}"

def create_interface(config: Config) -> gr.Interface:
    """
    Create and configure Gradio interface
    
    Args:
        config: Configuration object
        
    Returns:
        Gradio interface object
    """
    # Initialize VULCAN interface
    vulcan = VulcanInterface(config)
    
    # Get interface configuration
    interface_config = config.get_interface_config()
    
    # Create Gradio interface
    with gr.Blocks(
        title=interface_config['title'],
        theme=interface_config.get('theme', 'default')
    ) as demo:
        
        gr.Markdown(f"# {interface_config['title']}")
        gr.Markdown(interface_config['description'])
        
        with gr.Row():
            with gr.Column(scale=2):
                # Video upload
                video_input = gr.Video(
                    label="Upload Video",
                    format="mp4"
                )
                
                # Video information display
                video_info_output = gr.Markdown(
                    label="Video Information",
                    value="Upload a video to see its information"
                )
                
                # Update video info when video is uploaded
                video_input.change(
                    fn=vulcan.get_video_info,
                    inputs=video_input,
                    outputs=video_info_output
                )
            
            with gr.Column(scale=3):
                # Question input
                question_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What is this video about?",
                    lines=3
                )
                
                # Submit button
                submit_btn = gr.Button("Ask Question", variant="primary")
                
                # Answer output
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=5,
                    interactive=False
                )
                
                # Process question when submitted
                submit_btn.click(
                    fn=vulcan.process_video_question,
                    inputs=[video_input, question_input],
                    outputs=answer_output
                )
                
                # Also process on Enter key
                question_input.submit(
                    fn=vulcan.process_video_question,
                    inputs=[video_input, question_input],
                    outputs=answer_output
                )
        
        # Performance and management section
        with gr.Accordion("Performance & Management", open=False):
            with gr.Row():
                with gr.Column():
                    stats_btn = gr.Button("Show Performance Stats")
                    stats_output = gr.Textbox(
                        label="Statistics",
                        lines=5,
                        interactive=False
                    )
                    stats_btn.click(
                        fn=vulcan.get_performance_stats,
                        outputs=stats_output
                    )
                
                with gr.Column():
                    clear_cache_btn = gr.Button("Clear Cache", variant="secondary")
                    cache_output = gr.Textbox(
                        label="Cache Status",
                        lines=2,
                        interactive=False
                    )
                    clear_cache_btn.click(
                        fn=vulcan.clear_cache,
                        outputs=cache_output
                    )
        
        # Example questions
        with gr.Accordion("Example Questions", open=False):
            gr.Markdown("""
            **Try these example questions:**
            - What is the main topic of this video?
            - Who are the people speaking in the video?
            - What key points are discussed?
            - Can you summarize the video content?
            - What visual elements are shown?
            - What happens at the beginning/middle/end?
            - Are there any specific details mentioned?
            """)
    
    return demo

def launch_interface(config_path: Optional[str] = None):
    """
    Launch the VULCAN web interface
    
    Args:
        config_path: Optional path to configuration file
    """
    try:
        # Load configuration
        config = Config(config_path)
        
        # Setup environment
        from ..utils.config import EnvironmentManager
        EnvironmentManager.setup_environment()
        EnvironmentManager.setup_logging(config)
        
        # Create and launch interface
        demo = create_interface(config)
        
        interface_config = config.get_interface_config()
        demo.launch(
            share=interface_config.get('share', False),
            debug=interface_config.get('debug', False),
            server_name="0.0.0.0",
            server_port=7860
        )
        
    except Exception as e:
        logger.error(f"Error launching interface: {str(e)}")
        raise

if __name__ == "__main__":
    launch_interface()
