#!/usr/bin/env python3
"""
VULCAN - Video Understanding and Language-based Contextual Answer Network
Main application entry point
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config, EnvironmentManager
from src.interface.gradio_app import launch_interface

def setup_logging(config: Config):
    """Setup logging configuration"""
    log_config = config.config.get('logging', {})
    
    # Ensure logs directory exists
    log_file = log_config.get('file', 'logs/vulcan.log')
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file) if log_config.get('log_to_file', True) else logging.NullHandler(),
            logging.StreamHandler() if log_config.get('log_to_console', True) else logging.NullHandler()
        ]
    )

def check_requirements():
    """Check if all required dependencies are installed"""
    try:
        from src.utils.config import EnvironmentManager
        dependencies = EnvironmentManager.check_dependencies()
        
        missing = [pkg for pkg, available in dependencies.items() if not available]
        
        if missing:
            print("❌ Missing required dependencies:")
            for pkg in missing:
                print(f"   - {pkg}")
            print("\n🔧 Please install missing dependencies using:")
            print("   pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies are available")
        return True
        
    except ImportError as e:
        print(f"❌ Error checking dependencies: {e}")
        print("🔧 Please install requirements: pip install -r requirements.txt")
        return False

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="VULCAN - Video Understanding and Language-based Contextual Answer Network"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check if all dependencies are installed"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Port for web interface"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create public Gradio link"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        check_requirements()
        return
    
    try:
        print("🚀 Starting VULCAN - Video Understanding and Language-based Contextual Answer Network")
        print("=" * 80)
        
        # Check dependencies
        if not check_requirements():
            sys.exit(1)
        
        # Load configuration
        print(f"📁 Loading configuration from: {args.config}")
        config = Config(args.config)
        
        # Override config with command line arguments
        if args.port:
            config.set('interface.server_port', args.port)
        if args.share:
            config.set('interface.share', True)
        if args.debug:
            config.set('interface.debug', True)
            config.set('logging.level', 'DEBUG')
        
        # Setup environment
        print("🔧 Setting up environment...")
        EnvironmentManager.setup_environment()
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("VULCAN application starting...")
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Print configuration summary
        print("\n📋 Configuration Summary:")
        print(f"   Model Path: {config.get('model.checkpoint_path', 'Not specified')}")
        print(f"   Max Frames: {config.get('model.max_images_length', 45)}")
        print(f"   Subtitles: {'Enabled' if config.get('processing.add_subtitles', True) else 'Disabled'}")
        print(f"   Caching: {'Enabled' if config.get('processing.enable_caching', True) else 'Disabled'}")
        print(f"   Port: {config.get('interface.server_port', 7860)}")
        print(f"   Share: {'Yes' if config.get('interface.share', False) else 'No'}")
        
        # Launch interface
        print("\n🌐 Launching web interface...")
        print(f"🔗 Local URL: http://localhost:{config.get('interface.server_port', 7860)}")
        
        if config.get('interface.share', False):
            print("🌍 Public URL will be displayed once interface is ready")
        
        print("\n✨ Ready for video analysis!")
        print("📝 Upload a video and ask questions about its content")
        print("⏹️  Press Ctrl+C to stop the application")
        print("=" * 80)
        
        # Launch the interface
        launch_interface(args.config)
        
    except KeyboardInterrupt:
        print("\n👋 VULCAN application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting VULCAN: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
