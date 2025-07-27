"""
Configuration Management for VULCAN
Handles configuration loading and validation
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for VULCAN"""
    
    DEFAULT_CONFIG = {
        'model': {
            'checkpoint_path': 'pretrained_models/video_llama_checkpoint_last.pth',
            'config_path': 'test_configs/llama2_test_config.yaml',
            'max_images_length': 45,
            'max_subtitle_length': 400,
            'max_new_tokens': 512,
            'lora_r': 64,
            'lora_alpha': 16
        },
        'processing': {
            'add_subtitles': True,
            'sampling_strategy': 'uniform',
            'frame_quality': 'high',
            'enable_caching': True,
            'cache_dir': 'cache'
        },
        'whisper': {
            'model_size': 'base',
            'language': 'english',
            'audio_format': 'mp3',
            'audio_bitrate': '320k'
        },
        'interface': {
            'title': 'VULCAN - Video Understanding and Language-based Contextual Answer Network',
            'description': 'Upload a video and ask any question related to it. The model processes the video and generates an answer.',
            'theme': 'default',
            'share': False,
            'debug': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'vulcan.log'
        },
        'security': {
            'max_file_size_mb': 500,
            'allowed_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
            'sanitize_filenames': True
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        self.validate_config()
    
    def load_config(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            file_ext = Path(config_path).suffix.lower()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if file_ext in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif file_ext == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
            
            # Merge with default config
            self.config = self._merge_configs(self.config, loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            logger.info("Using default configuration")
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """
        Recursively merge configurations
        
        Args:
            default: Default configuration
            loaded: Loaded configuration
            
        Returns:
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_config(self):
        """Validate configuration values"""
        try:
            # Validate model settings
            if self.config['model']['max_images_length'] <= 0:
                self.config['model']['max_images_length'] = 45
            
            if self.config['model']['max_new_tokens'] <= 0:
                self.config['model']['max_new_tokens'] = 512
            
            # Validate paths
            required_dirs = [
                self.config['processing']['cache_dir'],
                'workspace/inference_subtitles'
            ]
            
            for dir_path in required_dirs:
                os.makedirs(dir_path, exist_ok=True)
            
            # Validate file size limits
            max_size = self.config['security']['max_file_size_mb']
            if max_size <= 0 or max_size > 1000:  # Reasonable limits
                self.config['security']['max_file_size_mb'] = 500
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Path to configuration key (e.g., 'model.max_tokens')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
            
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Path to configuration key
            value: Value to set
        """
        try:
            keys = key_path.split('.')
            config_ref = self.config
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set the value
            config_ref[keys[-1]] = value
            
        except Exception as e:
            logger.error(f"Error setting config value {key_path}: {str(e)}")
    
    def save_config(self, output_path: str, format: str = 'yaml'):
        """
        Save current configuration to file
        
        Args:
            output_path: Path to save configuration
            format: Output format ('yaml' or 'json')
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(self.config, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get_model_config(self) -> Dict:
        """Get model-specific configuration"""
        return self.config['model']
    
    def get_processing_config(self) -> Dict:
        """Get processing-specific configuration"""
        return self.config['processing']
    
    def get_whisper_config(self) -> Dict:
        """Get Whisper-specific configuration"""
        return self.config['whisper']
    
    def get_interface_config(self) -> Dict:
        """Get interface-specific configuration"""
        return self.config['interface']
    
    def get_security_config(self) -> Dict:
        """Get security-specific configuration"""
        return self.config['security']

class EnvironmentManager:
    """Manages environment-specific settings"""
    
    @staticmethod
    def setup_environment():
        """Setup environment variables and paths"""
        # Set up CUDA environment if available
        try:
            import torch
            if torch.cuda.is_available():
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("CUDA not available, using CPU")
        except ImportError:
            logger.info("PyTorch not available")
        
        # Set up cache directories
        cache_dirs = [
            'cache',
            'workspace/inference_subtitles',
            'workspace/inference_subtitles/mp3',
            'logs'
        ]
        
        for cache_dir in cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)
    
    @staticmethod
    def setup_logging(config: Config):
        """
        Setup logging configuration
        
        Args:
            config: Configuration object
        """
        log_config = config.get_logging_config()
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Logging configured successfully")
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check if required dependencies are available
        
        Returns:
            Dictionary of dependency status
        """
        dependencies = {}
        
        required_packages = [
            'torch', 'torchvision', 'transformers', 'opencv-python',
            'moviepy', 'whisper', 'gradio', 'webvtt-py', 'soundfile'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                dependencies[package] = True
            except ImportError:
                dependencies[package] = False
                logger.warning(f"Missing dependency: {package}")
        
        return dependencies
