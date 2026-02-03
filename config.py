"""
Configuration settings for Live Transcriber
Modify these values to customize the application behavior.
"""


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
BUFFER_DURATION = 10.0

# How often to process the buffer in seconds
# Smaller values = more responsive but more CPU usage
PROCESS_INTERVAL = 10.0


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
# UI Settings
# =============================================================================

# Maximum number of transcript entries to keep in memory
MAX_TRANSCRIPT_ENTRIES = 100

# Auto-refresh interval in seconds when running
UI_REFRESH_INTERVAL = 5.0


# =============================================================================
# Keyword Settings
# =============================================================================

# Keyword highlight color
KEYWORD_HIGHLIGHT_COLOR = "#FFD700"  # Gold
KEYWORD_HIGHLIGHT_TEXT_COLOR = "black"

# Keyword alert border color
KEYWORD_ALERT_BORDER_COLOR = "#FFD700"  # Gold
KEYWORD_ALERT_BG_COLOR = "#2D2D1F"  # Dark gold tint


# =============================================================================
# Debug Settings
# =============================================================================

# Enable verbose logging
DEBUG = False

# Print audio levels (useful for debugging audio capture)
PRINT_AUDIO_LEVELS = False
