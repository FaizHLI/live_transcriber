"""
Speaker Diarization Module using pyannote.audio
Uses the community-1 model optimized for Apple Silicon using MPS (Metal Performance Shaders)
"""

import numpy as np
import torch
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import time


@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing and speaker ID."""
    speaker: str
    start: float
    end: float
    
    @property
    def duration(self) -> float:
        return self.end - self.start


class PyAnnoteDiarizer:
    """
    Speaker diarization using pyannote.audio (community-1 model).
    Utilizes MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon.
    """
    
    def __init__(self, hf_token: Optional[str] = None, use_mps: bool = True):
        """
        Initialize the pyannote diarizer.
        
        Args:
            hf_token: Hugging Face token for accessing pyannote models.
                      Required for first download. Can also be set via
                      HF_TOKEN or HUGGINGFACE_TOKEN environment variable.
            use_mps: Whether to use MPS (Apple Silicon GPU) acceleration.
        """
        self.hf_token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        self.use_mps = use_mps
        
        self._pipeline = None
        self._device = None
        self._is_loaded = False
        
    def _get_device(self) -> torch.device:
        """
        Get the appropriate torch device.
        Prefers MPS on Apple Silicon, falls back to CPU.
        
        Returns:
            torch.device instance
        """
        if self.use_mps and torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def load_pipeline(self) -> bool:
        """
        Load the pyannote speaker diarization pipeline.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.hf_token:
            print("Error: Hugging Face token required for pyannote models.")
            print(f"  HF_TOKEN env var: {'set' if os.getenv('HF_TOKEN') else 'NOT SET'}")
            print(f"  HUGGINGFACE_TOKEN env var: {'set' if os.getenv('HUGGINGFACE_TOKEN') else 'NOT SET'}")
            print("Set HF_TOKEN environment variable or pass hf_token parameter.")
            print("\nTo get a token:")
            print("1. Go to https://huggingface.co/settings/tokens")
            print("2. Create a new token with 'read' access")
            print("3. Accept the model terms at:")
            print("   - https://huggingface.co/pyannote/speaker-diarization-community-1")
            print("   - https://huggingface.co/pyannote/segmentation-3.0")
            return False
        
        # Debug: show token is being used (masked)
        token_preview = self.hf_token[:10] + "..." if len(self.hf_token) > 10 else "***"
        print(f"Using HF token: {token_preview}")
        
        try:
            from pyannote.audio import Pipeline
            
            self._device = self._get_device()
            print(f"Loading pyannote pipeline on {self._device}...")
            
            start_time = time.time()
            
            # Load the speaker diarization pipeline
            # Using community-1 model which has better performance than 3.1
            # See: https://huggingface.co/pyannote/speaker-diarization-community-1
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=self.hf_token
            )
            
            # Move pipeline to the appropriate device
            # Some pyannote models may have issues with MPS, fallback to CPU if needed
            try:
                self._pipeline.to(self._device)
            except Exception as device_err:
                print(f"Warning: Could not use {self._device}, falling back to CPU: {device_err}")
                self._device = torch.device('cpu')
                self._pipeline.to(self._device)
            
            load_time = time.time() - start_time
            print(f"Pyannote pipeline loaded on {self._device} ({load_time:.2f}s)")
            
            self._is_loaded = True
            return True
            
        except ImportError as e:
            print(f"Error: pyannote.audio not installed. Run: pip install pyannote.audio")
            print(f"Details: {e}")
            return False
            
        except Exception as e:
            print(f"Error loading pyannote pipeline: {e}")
            error_str = str(e).lower()
            if "401" in str(e) or "unauthorized" in error_str:
                print("\n⚠️  Authentication error. Please check:")
                print("1. Your Hugging Face token is valid")
                print("2. You have accepted the model terms (see README for links)")
            elif "could not find" in error_str or "cannot find" in error_str or "locate the file" in error_str:
                print("\n⚠️  Model access error. You need to accept the model licenses:")
                print("1. https://huggingface.co/pyannote/speaker-diarization-community-1")
                print("2. https://huggingface.co/pyannote/segmentation-3.0")
                print("\nClick 'Agree and access repository' on each page.")
                print("Make sure you're logged into the same HF account that created your token.")
            return False
    
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio: Audio samples as numpy array (float32 or int16)
            sample_rate: Sample rate of the audio
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            List of SpeakerSegment objects with speaker labels and timing
        """
        if not self._is_loaded:
            if not self.load_pipeline():
                return []
        
        try:
            # Convert numpy array to torch tensor for direct processing
            # This avoids file I/O and torchcodec/FFmpeg issues
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            # Convert to torch tensor with shape (1, num_samples) for mono
            waveform = torch.from_numpy(audio).unsqueeze(0)
            
            # Prepare audio input as dict (avoids file decoding)
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            
            # Prepare diarization parameters
            params = {}
            if num_speakers is not None:
                params['num_speakers'] = num_speakers
            if min_speakers is not None:
                params['min_speakers'] = min_speakers
            if max_speakers is not None:
                params['max_speakers'] = max_speakers
            
            # Run diarization directly from memory
            output = self._pipeline(audio_input, **params)
            
            # Convert to SpeakerSegment list
            segments = []
            
            # Handle different pyannote output formats
            # community-1 model returns DiarizeOutput with speaker_diarization attribute
            if hasattr(output, 'speaker_diarization'):
                # New community-1 format
                for turn, speaker in output.speaker_diarization:
                    segments.append(SpeakerSegment(
                        speaker=speaker,
                        start=turn.start,
                        end=turn.end
                    ))
            elif hasattr(output, 'itertracks'):
                # Legacy format (3.1 and older)
                for turn, _, speaker in output.itertracks(yield_label=True):
                    segments.append(SpeakerSegment(
                        speaker=speaker,
                        start=turn.start,
                        end=turn.end
                    ))
            else:
                # Try iterating directly (some versions)
                try:
                    for turn, speaker in output:
                        segments.append(SpeakerSegment(
                            speaker=speaker,
                            start=turn.start,
                            end=turn.end
                        ))
                except Exception as iter_err:
                    print(f"Could not parse diarization output: {type(output)}")
                    print(f"Available attributes: {dir(output)}")
            
            return segments
            
        except Exception as e:
            print(f"Diarization error: {e}")
            return []
    
    def diarize_with_timing(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[List[SpeakerSegment], float]:
        """
        Perform diarization and return processing time.
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate
            
        Returns:
            Tuple of (segments list, processing time in seconds)
        """
        start_time = time.time()
        segments = self.diarize(audio, sample_rate)
        processing_time = time.time() - start_time
        
        return segments, processing_time
    
    def get_speaker_at_time(
        self,
        segments: List[SpeakerSegment],
        time_sec: float
    ) -> Optional[str]:
        """
        Get the speaker label at a specific time.
        
        Args:
            segments: List of SpeakerSegment objects
            time_sec: Time in seconds
            
        Returns:
            Speaker label if found, None otherwise
        """
        for segment in segments:
            if segment.start <= time_sec <= segment.end:
                return segment.speaker
        return None
    
    def merge_with_transcription(
        self,
        segments: List[SpeakerSegment],
        transcription_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge diarization results with transcription segments.
        Assigns speaker labels to transcription text.
        
        Args:
            segments: Diarization segments from diarize()
            transcription_segments: Transcription segments with 'start', 'end', 'text'
            
        Returns:
            List of merged segments with speaker labels
        """
        merged = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', 0)
            trans_mid = (trans_start + trans_end) / 2
            
            # Find the speaker at the midpoint of the transcription segment
            speaker = self.get_speaker_at_time(segments, trans_mid)
            
            # If no speaker found at midpoint, try to find any overlapping speaker
            if speaker is None:
                for seg in segments:
                    if seg.start < trans_end and seg.end > trans_start:
                        speaker = seg.speaker
                        break
            
            merged.append({
                'speaker': speaker or 'UNKNOWN',
                'start': trans_start,
                'end': trans_end,
                'text': trans_seg.get('text', ''),
                'words': trans_seg.get('words', [])
            })
        
        return merged
    
    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._is_loaded
    
    @property
    def device(self) -> Optional[torch.device]:
        """Get the current device."""
        return self._device


class SimpleSpeakerTracker:
    """
    Simple speaker tracking for when pyannote is not available.
    Uses basic voice activity detection and clustering.
    """
    
    def __init__(self):
        """Initialize the simple speaker tracker."""
        self._current_speaker = "SPEAKER_00"
        self._last_energy = 0.0
        self._speaker_count = 0
        
    def estimate_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Estimate the current speaker based on audio characteristics.
        This is a simple placeholder that doesn't do real diarization.
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate
            
        Returns:
            Speaker label string
        """
        # Calculate audio energy
        energy = np.sqrt(np.mean(audio ** 2))
        
        # Very basic "speaker change" detection based on energy changes
        # This is NOT real diarization - just a placeholder
        energy_change = abs(energy - self._last_energy)
        
        if energy_change > 0.1 and self._last_energy > 0.01:
            # Potential speaker change
            self._speaker_count = (self._speaker_count + 1) % 10
            self._current_speaker = f"SPEAKER_{self._speaker_count:02d}"
        
        self._last_energy = energy
        return self._current_speaker


def get_diarizer(hf_token: Optional[str] = None, use_mps: bool = True):
    """
    Get the best available diarizer.
    
    Args:
        hf_token: Hugging Face token
        use_mps: Whether to use MPS acceleration
        
    Returns:
        Diarizer instance
    """
    try:
        from pyannote.audio import Pipeline
        return PyAnnoteDiarizer(hf_token=hf_token, use_mps=use_mps)
    except ImportError:
        print("Warning: pyannote.audio not available.")
        print("Speaker diarization will be limited.")
        return SimpleSpeakerTracker()


def test_diarizer():
    """Test the diarizer with a simple audio sample."""
    print("Testing Pyannote Diarizer...")
    print("-" * 40)
    
    # Check for HF token
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    
    if not hf_token:
        print("No HF_TOKEN found. Set environment variable to test diarization.")
        print("Example: export HF_TOKEN='your_token_here'")
        return
    
    # Check device availability
    print(f"\nDevice availability:")
    print(f"  MPS (Apple Silicon): {torch.backends.mps.is_available()}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    
    diarizer = PyAnnoteDiarizer(hf_token=hf_token)
    
    if diarizer.load_pipeline():
        # Create test audio (3 seconds of noise)
        test_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        
        segments, proc_time = diarizer.diarize_with_timing(test_audio)
        
        print(f"\nDiarization completed in {proc_time:.2f}s")
        print(f"Found {len(segments)} speaker segments")
        
        for seg in segments:
            print(f"  {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
    else:
        print("Failed to load diarizer pipeline")


if __name__ == "__main__":
    test_diarizer()
