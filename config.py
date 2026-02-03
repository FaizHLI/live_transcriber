"""
Configuration settings for Live Transcriber
Modify these values to customize the application behavior.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Audio Settings
# =============================================================================

# Audio device name to capture from
AUDIO_DEVICE_NAME = "BlackHole 2ch"

# Sample rate in Hz (16kHz is optimal for Whisper)
SAMPLE_RATE = 16000

# Number of audio channels (1 = mono, recommended for transcription)
CHANNELS = 1

# Audio block size for streaming
BLOCK_SIZE = 1024


# =============================================================================
# Processing Settings
# =============================================================================

# Duration of the sliding window buffer in seconds
BUFFER_DURATION = 5

# How often to process the buffer in seconds
# Smaller values = more responsive but more CPU usage
PROCESS_INTERVAL = 5


# =============================================================================
# Transcription Settings (MLX-Whisper)
# =============================================================================

# MLX-Whisper model to use
# Options: 
#   - "mlx-community/whisper-large-v3-turbo" (recommended, fastest)
#   - "mlx-community/whisper-large-v3"
#   - "mlx-community/whisper-medium"
#   - "mlx-community/whisper-small"
#   - "mlx-community/whisper-base"
#   - "mlx-community/whisper-tiny"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"

# Language for transcription
# Set to None for auto-detection
# Common codes: 'en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'zh', 'ko'
TRANSCRIPTION_LANGUAGE = "en"


# =============================================================================
# Speaker Diarization Settings (pyannote.audio)
# =============================================================================

# Hugging Face token for pyannote models
# Can also be set via HF_TOKEN or HUGGINGFACE_TOKEN environment variable
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

# Use MPS (Apple Silicon GPU) for diarization
# Set to False to use CPU instead
USE_MPS = True

# Maximum number of speakers to detect
# Set to None for automatic detection
MAX_SPEAKERS = 4

# Minimum number of speakers (optional)
MIN_SPEAKERS = None


# =============================================================================
# UI Settings
# =============================================================================

# Maximum number of transcript entries to keep in memory
MAX_TRANSCRIPT_ENTRIES = 100

# Auto-refresh interval in seconds when running
UI_REFRESH_INTERVAL = 2.0


# =============================================================================
# Speaker Colors
# =============================================================================

# Colors for different speakers (cycling)
SPEAKER_COLORS = [
    "#1E90FF",  # Blue
    "#32CD32",  # Green
    "#FF6347",  # Tomato
    "#9370DB",  # Purple
    "#FF8C00",  # Orange
    "#20B2AA",  # Teal
    "#FF69B4",  # Pink
    "#FFD700",  # Gold
]

# Keyword highlight color
KEYWORD_HIGHLIGHT_COLOR = "#FFD700"  # Gold
KEYWORD_HIGHLIGHT_TEXT_COLOR = "black"


# =============================================================================
# Debug Settings
# =============================================================================

# Enable verbose logging
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Print audio levels (useful for debugging audio capture)
PRINT_AUDIO_LEVELS = False
