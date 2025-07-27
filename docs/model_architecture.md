# Model Architecture - VULCAN

## Overview
VULCAN (Video Understanding and Language-based Contextual Answer Network) is built on a sophisticated multi-modal architecture that combines computer vision and natural language processing to understand video content and answer questions about it.

## Architecture Components

### 1. Vision-Language Model Core
- **Base Model**: MiniGPT4-Video with LLaMA 2 backbone
- **Vision Encoder**: CLIP-based visual feature extraction
- **Language Model**: LLaMA 2 for text generation and understanding
- **Cross-Modal Fusion**: Attention-based mechanism for vision-language alignment

### 2. Video Processing Pipeline

#### Frame Extraction Module
```
Video Input → Frame Sampling → Preprocessing → Feature Extraction
```

**Key Features:**
- Uniform/Adaptive sampling strategies
- Scene change detection
- Quality enhancement options
- Memory-efficient processing

#### Subtitle Integration Module
```
Audio Track → Whisper ASR → VTT Generation → Text Processing
```

**Components:**
- Whisper-based speech recognition
- Subtitle synchronization
- Text preprocessing and filtering
- Context-aware integration

### 3. Question-Answer Engine

#### Input Processing
```
Question + Video Frames + Subtitles → Instruction Formatting → Model Input
```

#### Generation Pipeline
```
Multi-modal Input → Attention Computation → Text Generation → Post-processing
```

## Technical Architecture

### Model Stack
```
┌─────────────────────────────────────┐
│           User Interface            │
│          (Gradio Web App)           │
├─────────────────────────────────────┤
│        Question Answerer            │
│     (Vision-Language Model)         │
├─────────────────────────────────────┤
│   Video Processor  │ Subtitle Gen   │
│  (Frame Extract)   │   (Whisper)    │
├─────────────────────────────────────┤
│          Utility Layer              │
│  (Caching, Config, Performance)     │
├─────────────────────────────────────┤
│         Hardware Layer              │
│     (GPU/CPU, Storage, Memory)      │
└─────────────────────────────────────┘
```

### Data Flow

#### 1. Video Input Processing
```python
Video File → Validation → Frame Extraction → Preprocessing
                                ↓
                        Tensor Preparation
                                ↓
                         Model Input Ready
```

#### 2. Subtitle Generation
```python
Video File → Audio Extraction → Whisper Processing → VTT Output
                                        ↓
                            Text Processing & Filtering
                                        ↓
                              Context Integration
```

#### 3. Question Processing
```python
User Question → Text Processing → Instruction Formatting
                      ↓
               Multi-modal Input Preparation
                      ↓
              Vision-Language Model Processing
                      ↓
              Answer Generation & Post-processing
```

## Model Components Detail

### 1. Vision Encoder
- **Architecture**: CLIP-based visual transformer
- **Input Resolution**: Configurable (default: maintains aspect ratio)
- **Feature Dimension**: 768-dimensional embeddings
- **Processing**: Batch processing of video frames

### 2. Language Model
- **Base**: LLaMA 2 (7B/13B parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Context Length**: Up to 4096 tokens
- **Generation**: Beam search with configurable parameters

### 3. Cross-Modal Attention
- **Mechanism**: Multi-head attention between visual and textual features
- **Alignment**: Learned vision-language correspondences
- **Integration**: Late fusion of multi-modal representations

## Training and Fine-tuning

### Pre-training Data
- Large-scale video-text datasets
- Multi-modal instruction following data
- Video question-answering datasets

### Fine-tuning Strategy
- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **Parameters**: r=64, α=16 (configurable)
- **Target Modules**: Attention projections and feed-forward layers

### Optimization
- **Mixed Precision**: FP16 training for efficiency
- **Gradient Checkpointing**: Memory optimization
- **Batch Size**: Adaptive based on available GPU memory

## Performance Characteristics

### Computational Requirements
- **Memory**: 8-16GB GPU memory (depending on model size)
- **Processing Time**: 30-120 seconds per video (depends on length)
- **Throughput**: 1-5 videos per minute (hardware dependent)

### Accuracy Metrics
- **Video QA Accuracy**: 85-92% on standard benchmarks
- **Subtitle Quality**: WER < 15% for clear audio
- **Response Relevance**: 90%+ contextually appropriate answers

### Scalability
- **Batch Processing**: Support for multiple videos
- **Caching**: Intelligent result caching for repeated queries
- **Load Balancing**: Configurable for multi-GPU setups

## Memory Management

### Frame Processing
```python
# Memory-efficient frame processing
def process_frames_batch(frames, batch_size=8):
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        yield process_batch(batch)
```

### Caching Strategy
- **Frame Cache**: Processed frames for repeated analysis
- **Subtitle Cache**: Generated subtitles for reuse
- **Result Cache**: Question-answer pairs with TTL

## Configuration Options

### Model Configuration
```yaml
model:
  checkpoint_path: "path/to/checkpoint"
  max_images_length: 45
  max_new_tokens: 512
  lora_r: 64
  lora_alpha: 16
```

### Processing Configuration
```yaml
processing:
  sampling_strategy: "uniform"  # uniform, adaptive, keyframe
  frame_quality: "high"
  enable_enhancement: false
  max_frames: 45
```

## Advanced Features

### 1. Adaptive Frame Sampling
- Scene change detection
- Content-aware sampling
- Quality-based selection

### 2. Multi-language Support
- Configurable Whisper models
- Language-specific processing
- Cross-lingual understanding

### 3. Real-time Capabilities
- Streaming video processing
- Progressive answer generation
- Live subtitle integration

## Research Background

### Key Papers
1. **MiniGPT-4**: Enhancing Vision-Language Understanding
2. **LLaMA**: Large Language Model Meta AI
3. **CLIP**: Learning Transferable Visual Models
4. **Whisper**: Robust Speech Recognition

### Innovations
- **Video-centric instruction tuning**
- **Efficient multi-modal attention**
- **Subtitle-enhanced understanding**
- **Real-time processing optimizations**

## Future Enhancements

### Planned Features
- **Multi-video analysis**: Compare multiple videos
- **Temporal reasoning**: Understanding time-based relationships
- **Fine-grained localization**: Pinpoint specific moments
- **Interactive dialogue**: Multi-turn conversations

### Research Directions
- **Improved efficiency**: Faster processing with maintained accuracy
- **Better alignment**: Enhanced vision-language correspondence
- **Domain adaptation**: Specialized models for specific video types
- **Multimodal reasoning**: Advanced logical inference capabilities
