"""
Audio Capture Module for BlackHole Virtual Audio Driver
Captures system audio at 16kHz using sounddevice
"""

import numpy as np
import sounddevice as sd
from collections import deque
from threading import Lock, Event
from typing import Optional, Callable
import time


class AudioCapture:
    """
    Captures audio from BlackHole 2ch virtual audio driver.
    Uses a sliding window buffer for near-real-time processing.
    """
    
    SAMPLE_RATE = 16000  # 16kHz for Whisper compatibility
    CHANNELS = 1  # Mono for transcription
    BLOCK_SIZE = 1024  # Audio block size
    
    def __init__(
        self,
        buffer_duration: float = 5.0,
        device_name: str = "BlackHole 2ch"
    ):
        """
        Initialize the audio capture.
        
        Args:
            buffer_duration: Duration of sliding window buffer in seconds
            device_name: Name of the audio input device (BlackHole 2ch)
        """
        self.buffer_duration = buffer_duration
        self.device_name = device_name
        self.device_id: Optional[int] = None
        
        # Calculate buffer size
        self.buffer_size = int(self.SAMPLE_RATE * buffer_duration)
        
        # Thread-safe audio buffer using deque
        self._buffer = deque(maxlen=self.buffer_size)
        self._lock = Lock()
        self._stop_event = Event()
        
        # Stream reference
        self._stream: Optional[sd.InputStream] = None
        
        # Callback for new audio data
        self._on_audio_callback: Optional[Callable] = None
        
    def find_blackhole_device(self) -> Optional[int]:
        """
        Find the BlackHole 2ch device ID.
        
        Returns:
            Device ID if found, None otherwise
        """
        devices = sd.query_devices()
        
        for idx, device in enumerate(devices):
            if self.device_name.lower() in device['name'].lower():
                if device['max_input_channels'] > 0:
                    return idx
        
        return None
    
    def list_audio_devices(self) -> list:
        """
        List all available audio input devices.
        
        Returns:
            List of tuples (device_id, device_name, max_input_channels)
        """
        devices = sd.query_devices()
        input_devices = []
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((
                    idx,
                    device['name'],
                    device['max_input_channels']
                ))
        
        return input_devices
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for audio stream.
        Called for each audio block captured.
        """
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Add to buffer (thread-safe)
        with self._lock:
            self._buffer.extend(audio_data.tolist())
        
        # Trigger callback if registered
        if self._on_audio_callback is not None:
            self._on_audio_callback(audio_data)
    
    def start(self) -> bool:
        """
        Start audio capture from BlackHole device.
        
        Returns:
            True if started successfully, False otherwise
        """
        # Find BlackHole device
        self.device_id = self.find_blackhole_device()
        
        if self.device_id is None:
            print(f"Error: Could not find '{self.device_name}' device.")
            print("Available input devices:")
            for dev_id, name, channels in self.list_audio_devices():
                print(f"  [{dev_id}] {name} ({channels} ch)")
            return False
        
        print(f"Using audio device: {sd.query_devices(self.device_id)['name']}")
        
        try:
            self._stop_event.clear()
            
            # Create and start the input stream
            self._stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                blocksize=self.BLOCK_SIZE,
                callback=self._audio_callback
            )
            
            self._stream.start()
            print(f"Audio capture started at {self.SAMPLE_RATE}Hz")
            return True
            
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            return False
    
    def stop(self):
        """Stop audio capture."""
        self._stop_event.set()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("Audio capture stopped")
    
    def get_buffer(self) -> np.ndarray:
        """
        Get the current audio buffer as a numpy array.
        
        Returns:
            Numpy array of audio samples (float32, normalized)
        """
        with self._lock:
            audio = np.array(list(self._buffer), dtype=np.float32)
        
        return audio
    
    def get_buffer_duration(self) -> float:
        """
        Get the current buffer duration in seconds.
        
        Returns:
            Duration of audio in buffer
        """
        with self._lock:
            return len(self._buffer) / self.SAMPLE_RATE
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        with self._lock:
            self._buffer.clear()
    
    def set_callback(self, callback: Callable):
        """
        Set a callback function to be called when new audio is received.
        
        Args:
            callback: Function that takes audio_data (numpy array) as argument
        """
        self._on_audio_callback = callback
    
    def is_running(self) -> bool:
        """Check if audio capture is running."""
        return self._stream is not None and self._stream.active
    
    @property
    def sample_rate(self) -> int:
        """Get the sample rate."""
        return self.SAMPLE_RATE


def test_audio_capture():
    """Test function to verify audio capture is working."""
    print("Testing Audio Capture...")
    print("-" * 40)
    
    # List available devices
    capture = AudioCapture()
    print("\nAvailable input devices:")
    for dev_id, name, channels in capture.list_audio_devices():
        marker = " <-- BlackHole" if "blackhole" in name.lower() else ""
        print(f"  [{dev_id}] {name} ({channels} ch){marker}")
    
    # Try to start capture
    print("\nAttempting to start BlackHole capture...")
    if capture.start():
        print("Capturing audio for 5 seconds...")
        time.sleep(5)
        
        audio = capture.get_buffer()
        print(f"\nCaptured {len(audio)} samples ({len(audio)/capture.SAMPLE_RATE:.2f}s)")
        print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
        print(f"Audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")
        
        capture.stop()
    else:
        print("Failed to start audio capture.")
        print("\nMake sure BlackHole 2ch is installed and configured.")


if __name__ == "__main__":
    test_audio_capture()
