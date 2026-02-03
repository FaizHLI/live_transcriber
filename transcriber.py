"""
MLX-Whisper Transcription Module
Optimized for Apple Silicon using MLX framework
"""

import numpy as np
from typing import Optional, Dict, Any
import time


class MLXWhisperTranscriber:
    """
    Transcription using MLX-Whisper optimized for Apple Silicon.
    Uses the large-v3-turbo model for high-speed, accurate transcription.
    """
    
    MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the MLX-Whisper transcriber.
        
        Args:
            model_name: Optional custom model name. Defaults to large-v3-turbo.
        """
        self.model_name = model_name or self.MODEL_NAME
        self._model = None
        self._is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the MLX-Whisper model.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            import mlx_whisper
            
            print(f"Loading MLX-Whisper model: {self.model_name}")
            start_time = time.time()
            
            # The model is loaded on first transcription call
            # We do a dummy transcription to preload
            self._model = mlx_whisper
            self._is_loaded = True
            
            load_time = time.time() - start_time
            print(f"MLX-Whisper ready (initialization took {load_time:.2f}s)")
            
            return True
            
        except ImportError as e:
            print(f"Error: mlx_whisper not installed. Run: pip install mlx-whisper")
            print(f"Details: {e}")
            return False
            
        except Exception as e:
            print(f"Error loading MLX-Whisper model: {e}")
            return False
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio using MLX-Whisper.
        
        Args:
            audio: Audio samples as numpy array (float32, normalized -1 to 1)
            sample_rate: Sample rate of the audio (default 16kHz)
            language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
            task: 'transcribe' or 'translate'
            
        Returns:
            Dictionary with transcription results:
            {
                'text': str,           # Full transcription text
                'segments': list,      # List of segment dictionaries
                'language': str,       # Detected/used language
                'duration': float,     # Processing time in seconds
            }
        """
        if not self._is_loaded:
            if not self.load_model():
                return {
                    'text': '',
                    'segments': [],
                    'language': language,
                    'duration': 0.0,
                    'error': 'Model not loaded'
                }
        
        try:
            import mlx_whisper
            
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            start_time = time.time()
            
            # Perform transcription with MLX-Whisper
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_name,
                language=language,
                task=task,
                word_timestamps=True,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            # Format the result
            return {
                'text': result.get('text', '').strip(),
                'segments': result.get('segments', []),
                'language': result.get('language', language),
                'duration': processing_time
            }
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return {
                'text': '',
                'segments': [],
                'language': language,
                'duration': 0.0,
                'error': str(e)
            }
    
    def transcribe_with_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> list:
        """
        Transcribe audio and return segments with word-level timestamps.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of segments with timestamps:
            [
                {
                    'start': float,
                    'end': float,
                    'text': str,
                    'words': [{'word': str, 'start': float, 'end': float}, ...]
                },
                ...
            ]
        """
        result = self.transcribe(audio, sample_rate)
        return result.get('segments', [])
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


class FallbackWhisperTranscriber:
    """
    Fallback transcriber using standard Whisper if MLX-Whisper is not available.
    Uses torch with MPS backend for Apple Silicon acceleration.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize fallback Whisper transcriber.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self._model = None
        self._is_loaded = False
        
    def load_model(self) -> bool:
        """Load the Whisper model."""
        try:
            import whisper
            import torch
            
            # Use MPS if available (Apple Silicon GPU)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Loading Whisper {self.model_size} on {device}")
            
            self._model = whisper.load_model(self.model_size, device=device)
            self._is_loaded = True
            
            print(f"Whisper model loaded on {device}")
            return True
            
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            return False
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Transcribe audio using standard Whisper."""
        if not self._is_loaded:
            if not self.load_model():
                return {'text': '', 'segments': [], 'error': 'Model not loaded'}
        
        try:
            import torch
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            start_time = time.time()
            
            result = self._model.transcribe(
                audio,
                language="en",
                word_timestamps=True,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            return {
                'text': result.get('text', '').strip(),
                'segments': result.get('segments', []),
                'language': result.get('language', 'en'),
                'duration': processing_time
            }
            
        except Exception as e:
            return {'text': '', 'segments': [], 'error': str(e)}


def get_transcriber(prefer_mlx: bool = True):
    """
    Get the best available transcriber.
    
    Args:
        prefer_mlx: If True, prefer MLX-Whisper over standard Whisper
        
    Returns:
        Transcriber instance
    """
    if prefer_mlx:
        try:
            import mlx_whisper
            return MLXWhisperTranscriber()
        except ImportError:
            print("MLX-Whisper not available, falling back to standard Whisper")
    
    return FallbackWhisperTranscriber()


def test_transcriber():
    """Test the transcriber with a simple audio sample."""
    print("Testing MLX-Whisper Transcriber...")
    print("-" * 40)
    
    transcriber = get_transcriber()
    
    if transcriber.load_model():
        # Create a simple test audio (silence)
        test_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds of silence
        
        result = transcriber.transcribe(test_audio)
        print(f"Test result: {result}")
    else:
        print("Failed to load transcriber model")


if __name__ == "__main__":
    test_transcriber()
