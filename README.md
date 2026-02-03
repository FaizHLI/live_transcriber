# 🎙️ Live Transcriber

**Real-time Speaker Diarization & Keyword Tracking for Apple Silicon**

A Streamlit dashboard that captures live system audio, performs real-time transcription using MLX-Whisper, and identifies speakers using pyannote.audio. Track keywords and see them highlighted in the live feed.

![Apple Silicon Optimized](https://img.shields.io/badge/Apple%20Silicon-Optimized-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red)

## ✨ Features

- **Real-time Transcription**: Uses MLX-Whisper (large-v3-turbo) optimized for Apple Silicon
- **Speaker Diarization**: Identifies different speakers using pyannote.audio (community-1 model)
- **Keyword Tracking**: Highlight and track specific words/phrases in real-time
- **Apple Silicon Optimization**: Leverages MPS (Metal Performance Shaders) for GPU acceleration
- **5-Second Sliding Window**: Near-real-time processing with overlapping buffers
- **Beautiful UI**: Modern Streamlit dashboard with speaker color-coding

## 📋 Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.10+** (recommended: Python 3.11)
- **BlackHole 2ch** virtual audio driver
- **Hugging Face account** (for pyannote models)

## 🔧 Setup Guide

### Step 1: Install BlackHole Virtual Audio Driver

BlackHole is a virtual audio driver that lets us capture system audio.

```bash
# Install via Homebrew
brew install blackhole-2ch
```

Or download from: https://existential.audio/blackhole/

### Step 2: Configure macOS Audio MIDI

This is **critical** for capturing system audio. You need to create a Multi-Output Device.

1. **Open Audio MIDI Setup**
   - Press `Cmd + Space`, type "Audio MIDI Setup", press Enter
   - Or go to: `/Applications/Utilities/Audio MIDI Setup.app`

2. **Create a Multi-Output Device**
   - Click the **+** button in the bottom-left corner
   - Select **"Create Multi-Output Device"**

3. **Configure the Multi-Output Device**
   - Check **"BlackHole 2ch"** ✅
   - Check your **regular output device** (e.g., "MacBook Pro Speakers" or your headphones) ✅
   - Make sure BlackHole 2ch is listed **FIRST** (drag to reorder if needed)
   - Optionally rename it to "System + BlackHole"

4. **Set as System Output**
   - Right-click (or Control-click) on your new Multi-Output Device
   - Select **"Use This Device For Sound Output"**
   
   Alternatively:
   - Go to **System Settings > Sound > Output**
   - Select your **Multi-Output Device**

5. **Verify Setup**
   - Play some audio (YouTube, Spotify, etc.)
   - You should still hear it through your speakers/headphones
   - The audio is now being mirrored to BlackHole

> **Note**: When using the Multi-Output Device, you cannot adjust volume from the menu bar. Use the volume controls in your app or use the keyboard volume buttons.

### Step 3: Get a Hugging Face Token

The pyannote.audio speaker diarization models require authentication.

1. **Create a Hugging Face account** (if you don't have one)
   - Go to: https://huggingface.co/join

2. **Create an access token**
   - Go to: https://huggingface.co/settings/tokens
   - Click **"New token"**
   - Name it (e.g., "live-transcriber")
   - Select **"Read"** access
   - Click **"Generate token"**
   - **Copy the token** (you'll need it later)

3. **Accept the model licenses**
   
   You must accept the terms for these models (click **"Agree and access repository"** on each):
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
   
   > **Note**: Make sure you're logged into the same Hugging Face account that generated your token.

4. **Set the environment variable** (optional but recommended)
   ```bash
   # Add to your ~/.zshrc or ~/.bashrc
   export HF_TOKEN="hf_your_token_here"
   
   # Reload your shell
   source ~/.zshrc
   ```

### Step 4: Install Python Dependencies

```bash
# Navigate to the project directory
cd /path/to/live_transcriber

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🚀 Usage

1. **Enter your Hugging Face Token** in the sidebar (if not set as environment variable)

2. **Add Keywords** to track in the sidebar (comma-separated)
   - Example: `meeting, deadline, action item, next steps`

3. **Click "Initialize"** to load the transcription and diarization models
   - This may take 30-60 seconds on first run (models are downloaded)

4. **Click "Start"** to begin live transcription

5. **Watch the Live Feed** for real-time transcriptions with:
   - Speaker labels (SPEAKER_00, SPEAKER_01, etc.)
   - Highlighted keywords
   - Timestamps

6. **Click "Stop"** when done

## 🏗️ Architecture

```
live_transcriber/
├── app.py              # Main Streamlit application
├── audio_capture.py    # BlackHole audio capture module
├── transcriber.py      # MLX-Whisper transcription module
├── diarizer.py         # pyannote.audio speaker diarization
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Key Components

- **Audio Capture**: Uses `sounddevice` to capture from BlackHole at 16kHz
- **Transcription**: MLX-Whisper (large-v3-turbo) optimized for Apple Silicon
- **Diarization**: pyannote.audio (community-1 model) with MPS (GPU) acceleration
- **Processing**: 5-second sliding window with 2.5-second overlap

## ⚙️ Configuration

### Audio Settings

In `audio_capture.py`:
```python
SAMPLE_RATE = 16000  # 16kHz for Whisper
CHANNELS = 1         # Mono
BLOCK_SIZE = 1024    # Audio block size
```

### Processing Settings

In `app.py`:
```python
BUFFER_DURATION = 5.0    # 5-second sliding window
PROCESS_INTERVAL = 2.5   # Process every 2.5 seconds
```

### Model Settings

In `transcriber.py`:
```python
MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
```

## 🔍 Troubleshooting

### BlackHole Not Found

```
Error: Could not find 'BlackHole 2ch' device.
```

**Solutions:**
1. Make sure BlackHole is installed: `brew install blackhole-2ch`
2. Restart your Mac after installation
3. Check Audio MIDI Setup to verify BlackHole appears

### No Audio Being Captured

**Solutions:**
1. Verify Multi-Output Device is set as system output
2. Make sure audio is playing from an app
3. Check that BlackHole is checked in the Multi-Output Device

### Hugging Face Authentication Error

```
401 Client Error: Unauthorized
```
or
```
An error happened while trying to locate the file on the Hub
```

**Solutions:**
1. Verify your token is correct
2. Make sure you accepted **ALL** model licenses:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Ensure you're logged into the same HF account that created the token
4. Try regenerating your token with "Write" permissions

### MPS Not Available

If you see the model running on CPU instead of MPS:

```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

**Solutions:**
1. Update PyTorch: `pip install --upgrade torch`
2. Make sure you're on macOS 12.3+ with Apple Silicon

### Slow Transcription

**Solutions:**
1. Ensure you're using MLX-Whisper (not standard Whisper)
2. Close other GPU-intensive applications
3. Try a smaller model if needed

## 📊 Performance Tips

1. **Close browser tabs** playing audio/video to reduce noise
2. **Use headphones** to prevent feedback loops
3. **Adjust volume levels** in the app generating audio
4. **Keep the window focused** on the Streamlit app for better refresh rates

## 🔒 Privacy Note

- All processing happens **locally** on your Mac
- No audio is sent to external servers
- Hugging Face is only used for model downloads (one-time)
- Your transcriptions stay on your machine

## 📝 License

MIT License - Feel free to use and modify as needed.

## 🙏 Credits

- [MLX-Whisper](https://github.com/ml-explore/mlx-examples) - Apple's MLX framework
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [BlackHole](https://existential.audio/blackhole/) - Virtual audio driver
- [Streamlit](https://streamlit.io/) - Web application framework
