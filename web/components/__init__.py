# Archivo para hacer que components sea un paquete de Python

from .ui_components import setup_page_config, render_header, render_sidebar
from .video_handler import VideoUploader, VideoProcessor
from .results_display import ResultsDisplay

__all__ = [
    'setup_page_config',
    'render_header', 
    'render_sidebar',
    'VideoUploader',
    'VideoProcessor',
    'ResultsDisplay'
]