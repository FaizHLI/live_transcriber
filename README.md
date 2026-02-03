# 🎙️ Live Transcriber

**Real-time Transcription & Keyword Tracking for Apple Silicon**

A Streamlit dashboard that captures live system audio and performs real-time transcription using MLX-Whisper. Track keywords and see them highlighted in the live feed with alerts.

![Apple Silicon Optimized](https://img.shields.io/badge/Apple%20Silicon-Optimized-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red)

## ✨ Features

- **Real-time Transcription**: Uses MLX-Whisper (large-v3-turbo) optimized for Apple Silicon
- **Keyword Tracking**: Highlight and track specific words/phrases in real-time
- **Keyword Alerts**: Visual and console alerts when tracked keywords are detected
- **Plural Matching**: "baby" matches "babies", "tax" matches "taxes"
- **Compound Words**: "shut down" matches "shutdown", "shut-down"
- **Apple Silicon Optimization**: Leverages MLX framework for GPU acceleration
- **Beautiful UI**: Modern Streamlit dashboard with keyword highlighting

## 📋 Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+** (recommended: Python 3.11)
- **BlackHole 2ch** virtual audio driver

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
   - Set the sample rate to **48000 Hz**
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

### Step 3: Install Python Dependencies

```bash
# Navigate to the project directory
cd /path/to/live_transcriber

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🚀 Usage

1. **Add Keywords** to track in the sidebar
   - One keyword/phrase per line
   - Use ` / ` to separate variations (e.g., `AI / Artificial Intelligence`)
   - Example:
     ```
     AI / Artificial Intelligence
     Tax Cut
     Immigration / Immigrant
     ```

2. **Click "Initialize"** to load the MLX-Whisper model
   - This may take 30-60 seconds on first run (model is downloaded)

3. **Click "Start"** to begin live transcription

4. **Watch the Live Feed** for real-time transcriptions with:
   - 🔔 Keyword alerts (highlighted entries)
   - Timestamps for each transcription
   - Keyword badges showing which words matched

5. **Click "Stop"** when done

## 🏗️ Architecture

```
live_transcriber/
├── app.py              # Main Streamlit application
├── audio_capture.py    # BlackHole audio capture module
├── transcriber.py      # MLX-Whisper transcription module
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Key Components

- **Audio Capture**: Uses `sounddevice` to capture from BlackHole at 16kHz
- **Transcription**: MLX-Whisper (large-v3-turbo) optimized for Apple Silicon
- **Keyword Matching**: Whole-word matching with plural and compound word support

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
BUFFER_DURATION = 10.0   # 10-second sliding window
PROCESS_INTERVAL = 5.0   # Process every 5 seconds
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
- Your transcriptions stay on your machine

## 📝 License

MIT License - Feel free to use and modify as needed.

## 🙏 Credits

- [MLX-Whisper](https://github.com/ml-explore/mlx-examples) - Apple's MLX framework
- [BlackHole](https://existential.audio/blackhole/) - Virtual audio driver
- [Streamlit](https://streamlit.io/) - Web application framework
