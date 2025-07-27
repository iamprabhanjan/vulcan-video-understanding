# Usage Guide - VULCAN

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/iamprabhanjan/vulcan-video-understanding.git
cd vulcan-video-understanding

# Install dependencies
pip install -r requirements.txt

# Check if all dependencies are installed
python main.py --check-deps
```

### 2. Basic Usage

```bash
# Launch with default settings
python main.py

# Launch with custom configuration
python main.py --config configs/config.yaml

# Launch with debug mode
python main.py --debug

# Launch with public sharing enabled
python main.py --share
```

## Web Interface Guide

### Uploading Videos

1. **Supported Formats**: MP4, AVI, MOV, MKV, WebM, FLV, WMV
2. **Maximum Size**: 500MB (configurable)
3. **Recommended**: Clear audio for best subtitle generation

### Asking Questions

**Effective Question Types:**
- **Content Summary**: "What is this video about?"
- **Specific Details**: "Who are the speakers in this video?"
- **Visual Elements**: "What objects are shown in the video?"
- **Temporal Queries**: "What happens at the beginning/end?"
- **Contextual Questions**: "Why is this topic important?"

**Example Questions:**
```
- What is the main topic discussed in this video?
- Can you summarize the key points mentioned?
- Who are the people appearing in the video?
- What visual elements or objects are shown?
- What is the setting or location of this video?
- Are there any specific dates or numbers mentioned?
- What is the tone or mood of the video?
- What conclusions can be drawn from this content?
```

## Configuration Guide

### Basic Configuration

Edit `configs/config.yaml` to customize VULCAN:

```yaml
# Model settings
model:
  max_images_length: 45        # Number of frames to process
  max_new_tokens: 512         # Maximum response length

# Processing options
processing:
  add_subtitles: true         # Enable automatic subtitles
  enable_caching: true        # Cache results for faster repeated queries
  sampling_strategy: "uniform" # Frame sampling method

# Interface settings
interface:
  server_port: 7860          # Web interface port
  share: false               # Create public link
```

### Advanced Configuration

#### Whisper Settings
```yaml
whisper:
  model_size: "base"         # tiny, base, small, medium, large
  language: "english"        # Subtitle language
  temperature: 0.0           # Generation randomness
```

#### Performance Tuning
```yaml
performance:
  enable_gpu: true           # Use GPU acceleration
  mixed_precision: true      # Memory optimization
  max_memory_usage_gb: 16    # Memory limit
```

#### Security Settings
```yaml
security:
  max_file_size_mb: 500      # Maximum upload size
  allowed_extensions: [".mp4", ".avi", ".mov"]
  max_requests_per_minute: 10 # Rate limiting
```

## Command Line Options

```bash
python main.py [options]

Options:
  --config PATH       Path to configuration file (default: configs/config.yaml)
  --port PORT         Web interface port (default: 7860)
  --share             Create public Gradio link
  --debug             Enable debug mode with detailed logging
  --check-deps        Check if all dependencies are installed
  -h, --help          Show help message
```

## Performance Optimization

### Hardware Requirements

**Minimum:**
- 8GB RAM
- 4GB GPU memory (or CPU with 16GB RAM)
- 10GB free disk space

**Recommended:**
- 16GB+ RAM
- 8GB+ GPU memory (NVIDIA RTX 3060 or better)
- SSD storage for faster processing
- 20GB+ free disk space

### Processing Speed Tips

1. **Use GPU**: Ensure CUDA is properly installed
2. **Enable Caching**: Reuse results for repeated questions
3. **Optimize Video Size**: Smaller videos process faster
4. **Configure Frame Sampling**: Reduce max_images_length for faster processing

### Memory Management

```yaml
# Reduce memory usage
model:
  max_images_length: 30      # Fewer frames
performance:
  mixed_precision: true      # Use FP16
  batch_size: 1             # Smaller batches
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```yaml
# Solution: Reduce processing load
model:
  max_images_length: 20
performance:
  max_memory_usage_gb: 8
```

#### 2. Slow Processing
- Enable GPU acceleration
- Reduce video resolution
- Use smaller Whisper model
- Enable caching

#### 3. Poor Subtitle Quality
```yaml
# Solution: Use larger Whisper model
whisper:
  model_size: "medium"       # or "large"
  language: "english"        # Specify correct language
```

#### 4. Model Loading Errors
- Check model checkpoint path
- Ensure sufficient disk space
- Verify internet connection for downloads

### Error Messages

**"Model not found"**
- Check `model.checkpoint_path` in config
- Ensure model files are downloaded
- Run with `--debug` for detailed error info

**"CUDA out of memory"**
- Reduce `max_images_length`
- Enable `mixed_precision`
- Use smaller videos

**"Unsupported video format"**
- Convert video to supported format (MP4 recommended)
- Check file corruption

## Advanced Features

### Custom Model Configuration

```python
# Custom processing pipeline
from src.models.video_processor import VideoProcessor
from src.models.question_answerer import QuestionAnswerer

processor = VideoProcessor(max_images_length=30)
answerer = QuestionAnswerer(custom_config)
```

### Batch Processing

```python
# Process multiple videos
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
questions = ["What is this about?"] * len(videos)

for video, question in zip(videos, questions):
    answer = interface.process_video_question(video, question)
    print(f"{video}: {answer}")
```

### API Integration

```python
# Use as API
from src.interface.gradio_app import VulcanInterface
from src.utils.config import Config

config = Config("configs/config.yaml")
vulcan = VulcanInterface(config)

# Process programmatically
result = vulcan.process_video_question("video.mp4", "Question?")
```

## Best Practices

### Video Preparation
1. **Quality**: Use clear, well-lit videos
2. **Audio**: Ensure clear audio for subtitle generation
3. **Length**: Shorter videos (< 10 minutes) process faster
4. **Format**: MP4 with H.264 encoding recommended

### Question Formulation
1. **Be Specific**: Ask targeted questions
2. **Context**: Provide context if needed
3. **Multiple Questions**: Ask follow-up questions for details
4. **Avoid Ambiguity**: Use clear, unambiguous language

### Performance
1. **Cache Results**: Enable caching for repeated analysis
2. **Monitor Memory**: Watch system resources
3. **Use GPU**: Enable GPU acceleration when available
4. **Regular Cleanup**: Clear cache periodically

## Integration Examples

### Jupyter Notebook Usage
```python
# In Jupyter notebook
from main import VulcanInterface
from src.utils.config import Config

config = Config()
vulcan = VulcanInterface(config)

# Interactive analysis
video_path = "path/to/video.mp4"
questions = [
    "What is the main topic?",
    "Who are the speakers?",
    "What are the key points?"
]

for question in questions:
    answer = vulcan.process_video_question(video_path, question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### REST API Wrapper
```python
# Create simple REST API
from flask import Flask, request, jsonify

app = Flask(__name__)
vulcan = VulcanInterface(Config())

@app.route('/analyze', methods=['POST'])
def analyze_video():
    video_file = request.files['video']
    question = request.form['question']
    answer = vulcan.process_video_question(video_file, question)
    return jsonify({'answer': answer})
```

## Support and Resources

- **Documentation**: Check `docs/` folder for detailed guides
- **Issues**: Report bugs on GitHub Issues
- **Configuration**: See `configs/config.yaml` for all options
- **Logs**: Check `logs/vulcan.log` for troubleshooting
- **Performance**: Use `--debug` flag for detailed metrics
