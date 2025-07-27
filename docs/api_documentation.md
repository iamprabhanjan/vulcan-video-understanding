# API Documentation - VULCAN

## Overview
VULCAN (Video Understanding and Language-based Contextual Answer Network) provides both a web interface and programmatic API for video analysis and question answering.

## Core API Classes

### VideoProcessor
Handles video frame extraction and preprocessing.

```python
from src.models.video_processor import VideoProcessor

processor = VideoProcessor(max_images_length=45)

# Extract frames from video
frames, raw_frames = processor.extract_frames("video.mp4")

# Get video information
info = processor.get_video_info("video.mp4")

# Validate video file
is_valid = processor.validate_video("video.mp4")
```

**Methods:**
- `extract_frames(video_path, sampling_strategy="uniform")`: Extract frames from video
- `get_video_info(video_path)`: Get video metadata
- `validate_video(video_path)`: Validate video file

### SubtitleGenerator
Generates subtitles using Whisper and processes existing subtitle files.

```python
from src.models.subtitle_generator import SubtitleGenerator

generator = SubtitleGenerator(model_size="base", language="english")

# Generate subtitles
subtitle_path = generator.generate_subtitles("video.mp4")

# Load existing subtitles
from src.models.subtitle_generator import SubtitleProcessor
subtitles = SubtitleProcessor.load_subtitles("subtitles.vtt")
```

**Methods:**
- `generate_subtitles(video_path, force_regenerate=False)`: Generate VTT subtitles
- `load_model()`: Load Whisper model
- `extract_audio(video_path, audio_path)`: Extract audio from video

### QuestionAnswerer
Handles AI-powered question answering for video content.

```python
from src.models.question_answerer import QuestionAnswerer

answerer = QuestionAnswerer(model_config)
answerer.load_model(model_path, config_path)

# Process question
answer = answerer.process_question(video_frames, subtitle_text, question)
```

**Methods:**
- `load_model(model_path, config_path)`: Load vision-language model
- `process_question(video_frames, subtitle_text, question)`: Generate answer
- `prepare_input(frames, subtitle_text, question)`: Prepare model input

## Utility Classes

### Config
Configuration management for all VULCAN components.

```python
from src.utils.config import Config

# Load configuration
config = Config("configs/config.yaml")

# Get specific configuration values
model_config = config.get_model_config()
max_tokens = config.get("model.max_new_tokens", 512)

# Set configuration values
config.set("processing.enable_caching", True)
```

### VideoUtils
Common utility functions for video operations.

```python
from src.utils.video_utils import VideoUtils, CacheManager

# Validate video file
is_valid = VideoUtils.validate_video_file("video.mp4")

# Get file size
size_mb = VideoUtils.get_file_size_mb("video.mp4")

# Cache management
cache = CacheManager("cache_dir")
cache_key = cache.get_cache_key("video.mp4", "question")
cached_result = cache.get_cached_result(cache_key)
```

## Web Interface API

### VulcanInterface
Main interface class that orchestrates all components.

```python
from src.interface.gradio_app import VulcanInterface

interface = VulcanInterface(config)

# Process video and question
answer = interface.process_video_question(video_file, question)

# Get video information
info = interface.get_video_info(video_file)

# Performance statistics
stats = interface.get_performance_stats()
```

## Configuration Parameters

### Model Configuration
- `checkpoint_path`: Path to model checkpoint file
- `max_images_length`: Maximum number of frames to process
- `max_new_tokens`: Maximum tokens for answer generation
- `lora_r`, `lora_alpha`: LoRA fine-tuning parameters

### Processing Configuration
- `add_subtitles`: Enable/disable subtitle generation
- `sampling_strategy`: Frame sampling method ("uniform", "keyframe", "adaptive")
- `enable_caching`: Enable result caching
- `cache_dir`: Directory for cache storage

### Whisper Configuration
- `model_size`: Whisper model size ("tiny", "base", "small", "medium", "large")
- `language`: Language for subtitle generation
- `temperature`: Sampling temperature for generation

## Error Handling

All API methods include comprehensive error handling and logging:

```python
try:
    answer = interface.process_video_question(video_file, question)
except Exception as e:
    logger.error(f"Processing error: {str(e)}")
    return "Error processing your request"
```

## Performance Monitoring

The system includes built-in performance monitoring:

```python
from src.utils.video_utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Record processing metrics
start_time = monitor.start_timer()
# ... processing ...
duration = monitor.end_timer(start_time)
monitor.record_processing(duration, success=True)

# Get metrics
metrics = monitor.get_metrics()
```

## Caching System

Intelligent caching system for improved performance:

```python
# Cache results
cache_key = cache_manager.get_cache_key(video_path, question)
cache_manager.cache_result(cache_key, result)

# Retrieve cached results
cached = cache_manager.get_cached_result(cache_key)
```

## Security Features

- File size validation
- Format validation
- Filename sanitization
- Rate limiting (configurable)
- Suspicious file detection

## Dependencies

See `requirements.txt` for complete dependency list. Core dependencies:
- PyTorch >= 2.0.1
- Transformers >= 4.37.2
- OpenCV >= 4.7.0.72
- Gradio >= 4.0.0
- Whisper >= 1.1.10

## Example Usage

### Complete Processing Pipeline

```python
from src.utils.config import Config
from src.interface.gradio_app import VulcanInterface

# Initialize
config = Config("configs/config.yaml")
interface = VulcanInterface(config)

# Process video
video_path = "sample_video.mp4"
question = "What is the main topic of this video?"

answer = interface.process_video_question(video_path, question)
print(f"Answer: {answer}")
```

### Custom Processing

```python
from src.models.video_processor import VideoProcessor
from src.models.subtitle_generator import SubtitleGenerator
from src.models.question_answerer import QuestionAnswerer

# Initialize components
video_processor = VideoProcessor()
subtitle_generator = SubtitleGenerator()
question_answerer = QuestionAnswerer(config)

# Process video
frames, _ = video_processor.extract_frames("video.mp4")
subtitle_path = subtitle_generator.generate_subtitles("video.mp4")

# Generate answer
answer = question_answerer.process_question(frames, subtitle_text, question)
```
