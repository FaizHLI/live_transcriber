"""
Live Transcriber - Real-time Transcription & Keyword Tracking Dashboard
Optimized for Apple Silicon with MLX-Whisper

This Streamlit application captures live system audio via BlackHole 2ch,
performs real-time transcription, and highlights user-specified keywords
in the live feed with alerts.
"""

import streamlit as st
import numpy as np
import time
import re
from collections import deque
from threading import Thread, Event, Lock
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import our custom modules
from audio_capture import AudioCapture
from transcriber import get_transcriber


# Page configuration
st.set_page_config(
    page_title="Live Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@dataclass
class TranscriptEntry:
    """Represents a single transcript entry with text and keywords."""
    text: str
    timestamp: float
    entry_id: str = ""  # Unique ID for tracking confirmations
    keywords_found: List[str] = field(default_factory=list)
    has_keyword: bool = False  # Flag for keyword alerts


class LiveTranscriber:
    """
    Main class that orchestrates audio capture, transcription, and keyword tracking.
    """
    
    BUFFER_DURATION = 10.0  # 10-second sliding window
    PROCESS_INTERVAL = 5.0  # Process every 5 seconds
    
    def __init__(self):
        """Initialize the live transcriber."""
        self.audio_capture = AudioCapture(buffer_duration=self.BUFFER_DURATION)
        self.transcriber = None
        
        # Processing state
        self._is_running = False
        self._stop_event = Event()
        self._process_thread: Optional[Thread] = None
        
        # Transcript storage (thread-safe)
        self._transcript_lock = Lock()
        self._transcript: deque = deque(maxlen=100)  # Keep last 100 entries
        
        # Keywords
        self._keywords: List[str] = []
        
        # Status
        self._status = "Idle"
        self._last_process_time = 0.0
        
    def initialize(self) -> bool:
        """
        Initialize transcription model.
        
        Returns:
            True if initialization successful
        """
        self._status = "Loading transcription model..."
        
        # Initialize transcriber (MLX-Whisper)
        try:
            self.transcriber = get_transcriber(prefer_mlx=True)
            if not self.transcriber.load_model():
                self._status = "Failed to load transcription model"
                return False
        except Exception as e:
            self._status = f"Transcriber error: {e}"
            return False
        
        self._status = "Ready"
        return True
    
    def set_keywords(self, keywords: List[str]):
        """Set the keywords to track."""
        self._keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
    
    def start(self) -> bool:
        """Start the live transcription."""
        if self._is_running:
            return True
        
        # Start audio capture
        if not self.audio_capture.start():
            self._status = "Failed to start audio capture"
            return False
        
        # Start processing thread
        self._stop_event.clear()
        self._is_running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        self._status = "Running"
        return True
    
    def stop(self):
        """Stop the live transcription."""
        self._stop_event.set()
        self._is_running = False
        
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        
        self.audio_capture.stop()
        self._status = "Stopped"
    
    def _process_loop(self):
        """Main processing loop for transcription and diarization."""
        while not self._stop_event.is_set():
            try:
                # Wait for enough audio
                buffer_duration = self.audio_capture.get_buffer_duration()
                
                if buffer_duration >= self.BUFFER_DURATION * 0.8:  # 80% of buffer
                    self._process_audio()
                
                # Wait before next processing
                time.sleep(self.PROCESS_INTERVAL)
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(1.0)
    
    def _process_audio(self):
        """Process the current audio buffer."""
        start_time = time.time()
        
        # Get audio from buffer
        audio = self.audio_capture.get_buffer()
        
        if len(audio) < 16000:  # Less than 1 second
            return
        
        # Transcribe
        result = self.transcriber.transcribe(
            audio,
            sample_rate=self.audio_capture.sample_rate
        )
        
        text = result.get('text', '').strip()
        
        if not text:
            return
        
        # Find keywords in text
        found_keywords = self._find_keywords(text)
        has_keyword = len(found_keywords) > 0
        
        # Alert on keyword detection
        if has_keyword:
            print(f"🔔 KEYWORD ALERT: {found_keywords} detected at {time.strftime('%H:%M:%S')}")
        
        # Create transcript entry with unique ID
        entry_id = f"{time.time():.3f}"
        entry = TranscriptEntry(
            text=text,
            timestamp=time.time(),
            entry_id=entry_id,
            keywords_found=found_keywords,
            has_keyword=has_keyword
        )
        
        # Add to transcript (thread-safe)
        with self._transcript_lock:
            self._transcript.append(entry)
        
        self._last_process_time = time.time() - start_time
    
    def _find_keywords(self, text: str) -> List[str]:
        """Find keywords in the text using whole word matching with compound word support."""
        found = []
        
        for keyword in self._keywords:
            # Use the same pattern logic as highlighting
            # This handles compound words like "shut down" / "shutdown"
            pattern_str = get_keyword_pattern(keyword)
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(text):
                found.append(keyword)
        
        return found
    
    def get_transcript(self) -> List[TranscriptEntry]:
        """Get the current transcript entries."""
        with self._transcript_lock:
            return list(self._transcript)
    
    def clear_transcript(self):
        """Clear the transcript."""
        with self._transcript_lock:
            self._transcript.clear()
    
    @property
    def status(self) -> str:
        """Get the current status."""
        return self._status
    
    @property
    def is_running(self) -> bool:
        """Check if transcription is running."""
        return self._is_running


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def get_plural_variations(word: str) -> List[str]:
    """
    Generate common plural/singular variations of a word.
    
    Args:
        word: The word to generate variations for
        
    Returns:
        List of variations including the original word
    """
    word = word.lower().strip()
    variations = {word}
    
    # Common irregular plurals
    irregulars = {
        'child': 'children', 'children': 'child',
        'person': 'people', 'people': 'person',
        'man': 'men', 'men': 'man',
        'woman': 'women', 'women': 'woman',
        'foot': 'feet', 'feet': 'foot',
        'tooth': 'teeth', 'teeth': 'tooth',
        'mouse': 'mice', 'mice': 'mouse',
        'goose': 'geese', 'geese': 'goose',
    }
    
    if word in irregulars:
        variations.add(irregulars[word])
    
    # If word ends in 's', 'es', 'ies' - try to find singular
    if word.endswith('ies') and len(word) > 3:
        # babies -> baby
        variations.add(word[:-3] + 'y')
    elif word.endswith('es') and len(word) > 2:
        # taxes -> tax, boxes -> box, watches -> watch
        variations.add(word[:-2])  # taxes -> tax
        variations.add(word[:-1])  # sometimes just remove 's'
    elif word.endswith('s') and len(word) > 1 and not word.endswith('ss'):
        # cars -> car
        variations.add(word[:-1])
    
    # Generate plurals from potential singular
    if not word.endswith('s'):
        # Standard plural: add 's'
        variations.add(word + 's')
        
        # Words ending in y (preceded by consonant) -> ies
        if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
            variations.add(word[:-1] + 'ies')
        
        # Words ending in s, x, z, ch, sh -> add 'es'
        if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            variations.add(word + 'es')
    
    return list(variations)


def get_keyword_pattern(keyword: str) -> str:
    """
    Generate a regex pattern that matches keyword variations:
    - "shut down" matches "shut down", "shut-down", "shutdown"
    - "shutdown" matches "shutdown", "shut-down", "shut down"
    - Handles plurals: "baby" matches "babies", "tax" matches "taxes"
    - Single words use word boundaries to avoid substring matches
    
    Args:
        keyword: The keyword to create a pattern for
        
    Returns:
        Regex pattern string
    """
    keyword = keyword.strip().lower()
    
    # Check if keyword contains spaces or hyphens (compound word)
    if ' ' in keyword or '-' in keyword:
        # Split by space or hyphen
        parts = re.split(r'[\s\-]+', keyword)
        if len(parts) > 1:
            # For compound words, generate plural variations of the last word
            last_word_variations = get_plural_variations(parts[-1])
            
            # Create patterns for each variation
            patterns = []
            for variation in last_word_variations:
                modified_parts = parts[:-1] + [variation]
                escaped_parts = [re.escape(p) for p in modified_parts if p]
                pattern = r'[\s\-]?'.join(escaped_parts)
                patterns.append(pattern)
            
            # Join all patterns with OR
            combined = '|'.join(f'(?:{p})' for p in patterns)
            return r'\b(?:' + combined + r')\b'
    
    # Single word - generate plural variations
    variations = get_plural_variations(keyword)
    escaped_variations = [re.escape(v) for v in variations]
    
    # Join variations with OR: (baby|babies)
    return r'\b(?:' + '|'.join(escaped_variations) + r')\b'


def highlight_keywords(text: str, keywords: List[str]) -> str:
    """
    Highlight keywords in text using HTML markup.
    - Only matches whole words (e.g., "ice" won't match in "sacrifice")
    - Handles compound words (e.g., "shut down" matches "shutdown")
    
    Args:
        text: Original text
        keywords: List of keywords to highlight
        
    Returns:
        HTML string with highlighted keywords
    """
    # First escape any HTML in the original text
    result = escape_html(text)
    
    if not keywords:
        return result
    
    # Sort keywords by length (longest first) to avoid partial replacements
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    for keyword in sorted_keywords:
        # Get the pattern for this keyword (handles compound words)
        pattern_str = get_keyword_pattern(keyword)
        pattern = re.compile(pattern_str, re.IGNORECASE)
        
        def replace_match(match):
            # Preserve the original case from the text
            return f'<mark style="background-color:#FFD700;color:black;font-weight:bold;padding:2px 4px;border-radius:3px;">{match.group(0)}</mark>'
        
        result = pattern.sub(replace_match, result)
    
    return result


def format_timestamp(timestamp: float) -> str:
    """Format a timestamp for display."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%H:%M:%S")


def main():
    """Main Streamlit application."""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .status-running {
        color: #32CD32;
        font-weight: bold;
    }
    .status-stopped {
        color: #FF6347;
        font-weight: bold;
    }
    .transcript-entry {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        background-color: #1E1E1E;
        border-left: 4px solid;
    }
    .speaker-label {
        font-weight: bold;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .transcript-text {
        font-size: 1rem;
        line-height: 1.5;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #888;
        margin-left: 10px;
    }
    .keyword-badge {
        background-color: #FFD700;
        color: black;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    .keyword-group {
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    # Check if transcriber needs to be recreated (e.g., after code update)
    if 'transcriber' not in st.session_state or not hasattr(st.session_state.transcriber, '_keywords'):
        st.session_state.transcriber = LiveTranscriber()
        st.session_state.initialized = False  # Force re-initialization with new instance
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []  # Flat list of all keyword variations
    if 'keyword_groups' not in st.session_state:
        st.session_state.keyword_groups = []  # List of (primary, [variations]) tuples
    if 'confirmed_alerts' not in st.session_state:
        st.session_state.confirmed_alerts = set()  # Set of confirmed entry_id:keyword pairs
    
    transcriber = st.session_state.transcriber
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Keywords
        st.subheader("🔍 Keywords")
        
        # Default keywords in the new format
        default_keywords = """AI / Artificial Intelligence
Manufacturing
Marijuana / Weed / Cannabis
Minimum Wage
Democracy
ICE / National Guard
Tax Cut
New Jersey
Trump
Hydrogen
Inflation
Immigrant / Immigration
Tariff"""
        
        # Convert stored keyword groups back to text format for display
        if st.session_state.keyword_groups:
            current_text = "\n".join([
                " / ".join([primary] + variations) if variations else primary
                for primary, variations in st.session_state.keyword_groups
            ])
        else:
            current_text = default_keywords
        
        keywords_input = st.text_area(
            "Track Keywords",
            value=current_text,
            height=300,
            help="One keyword per line. Use ' / ' to separate variations (e.g., 'AI / Artificial Intelligence')"
        )
        
        if keywords_input:
            # Parse the new format: one group per line, variations separated by " / "
            keyword_groups = []
            all_keywords = []
            
            for line in keywords_input.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Split by " / " to get variations
                variations = [v.strip() for v in line.split(' / ') if v.strip()]
                
                if variations:
                    primary = variations[0]
                    other_variations = variations[1:] if len(variations) > 1 else []
                    keyword_groups.append((primary, other_variations))
                    all_keywords.extend(variations)
            
            st.session_state.keyword_groups = keyword_groups
            st.session_state.keywords = all_keywords
            transcriber.set_keywords(all_keywords)
        
        if st.session_state.keyword_groups:
            st.write("**Tracking:**")
            for primary, variations in st.session_state.keyword_groups:
                if variations:
                    # Show primary with variations count
                    variation_text = f" (+{len(variations)} variations)"
                    st.markdown(
                        f'<span class="keyword-badge">{primary}</span>'
                        f'<span style="color: #888; font-size: 0.75rem;">{variation_text}</span>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f'<span class="keyword-badge">{primary}</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # Audio Device Info
        st.subheader("🎤 Audio Device")
        audio_capture = AudioCapture()
        devices = audio_capture.list_audio_devices()
        
        blackhole_found = False
        for dev_id, name, channels in devices:
            if 'blackhole' in name.lower():
                st.success(f"✅ {name}")
                blackhole_found = True
                break
        
        if not blackhole_found:
            st.error("❌ BlackHole 2ch not found")
            st.info("Install BlackHole: brew install blackhole-2ch")
        
        with st.expander("All Audio Devices"):
            for dev_id, name, channels in devices:
                st.text(f"[{dev_id}] {name} ({channels}ch)")
        
        st.divider()
        
        # Status
        st.subheader("📊 Status")
        status = transcriber.status
        
        if transcriber.is_running:
            st.markdown(f'<p class="status-running">● {status}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="status-stopped">● {status}</p>', unsafe_allow_html=True)
        
        # Keyword Alerts Section
        if st.session_state.initialized:
            entries = transcriber.get_transcript()
            keyword_entries = [e for e in entries if e.has_keyword]
            
            if keyword_entries:
                st.divider()
                st.subheader("🔔 Keyword Alerts")
                
                # Count confirmed vs total
                total_alerts = sum(len(e.keywords_found) for e in keyword_entries)
                confirmed_count = len(st.session_state.confirmed_alerts)
                
                col_count, col_clear = st.columns([3, 2])
                with col_count:
                    st.caption(f"{confirmed_count}/{total_alerts} confirmed")
                with col_clear:
                    if st.button("Reset", key="clear_confirmations", help="Clear all confirmations"):
                        st.session_state.confirmed_alerts = set()
                        st.rerun()
                
                # Show each keyword detection with checkbox
                for entry in reversed(keyword_entries):  # Most recent first
                    entry_time = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
                    
                    for kw in entry.keywords_found:
                        alert_key = f"{entry.entry_id}:{kw}"
                        is_confirmed = alert_key in st.session_state.confirmed_alerts
                        
                        # Create checkbox
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            if st.checkbox("", value=is_confirmed, key=f"chk_{alert_key}", label_visibility="collapsed"):
                                st.session_state.confirmed_alerts.add(alert_key)
                            else:
                                st.session_state.confirmed_alerts.discard(alert_key)
                        
                        with col2:
                            # Show keyword with color based on confirmation
                            if alert_key in st.session_state.confirmed_alerts:
                                # Confirmed - green
                                st.markdown(
                                    f'<span style="color:#32CD32;font-weight:bold;">✓ {kw}</span> '
                                    f'<span style="color:#666;font-size:0.8rem;">{entry_time}</span>',
                                    unsafe_allow_html=True
                                )
                            else:
                                # Unconfirmed - gold/yellow
                                st.markdown(
                                    f'<span style="color:#FFD700;font-weight:bold;">{kw}</span> '
                                    f'<span style="color:#666;font-size:0.8rem;">{entry_time}</span>',
                                    unsafe_allow_html=True
                                )
    
    # Main content
    st.markdown('<h1 class="main-header">🎙️ Live Transcriber</h1>', unsafe_allow_html=True)
    st.markdown("Real-time transcription with keyword alerts — optimized for Apple Silicon")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Initialize", disabled=st.session_state.initialized):
            with st.spinner("Loading MLX-Whisper model..."):
                if transcriber.initialize():
                    st.session_state.initialized = True
                    st.success("Ready! Click Start to begin transcription.")
                else:
                    st.error(f"Initialization failed: {transcriber.status}")
    
    with col2:
        start_disabled = not st.session_state.initialized or transcriber.is_running
        if st.button("▶️ Start", disabled=start_disabled):
            if transcriber.start():
                st.success("Transcription started!")
            else:
                st.error(f"Failed to start: {transcriber.status}")
    
    with col3:
        if st.button("⏹️ Stop", disabled=not transcriber.is_running):
            transcriber.stop()
            st.info("Transcription stopped")
    
    with col4:
        if st.button("🗑️ Clear"):
            transcriber.clear_transcript()
            st.info("Transcript cleared")
    
    st.divider()
    
    # Live Feed
    st.subheader("📜 Live Feed")
    
    # Create a placeholder for the live feed
    feed_placeholder = st.empty()
    
    # Auto-refresh when running
    if transcriber.is_running:
        st.markdown("*Auto-refreshing every 2 seconds...*")
    
    # Display transcript entries
    entries = transcriber.get_transcript()
    
    if entries:
        # Display in reverse order (newest first)
        feed_html = ""
        
        for entry in reversed(entries):
            timestamp_str = format_timestamp(entry.timestamp)
            
            # Highlight keywords in text
            highlighted_text = highlight_keywords(entry.text, entry.keywords_found)
            
            # Build entry HTML - keywords badges for matches found
            # Check which keywords are confirmed
            keywords_html = ""
            all_confirmed = True
            if entry.keywords_found:
                badges = []
                for kw in entry.keywords_found:
                    alert_key = f"{entry.entry_id}:{kw}"
                    if alert_key in st.session_state.confirmed_alerts:
                        # Confirmed - green badge
                        badges.append(f'<span style="background-color:#32CD32;color:black;padding:2px 6px;border-radius:3px;font-size:0.8rem;margin-right:5px;">✓ {escape_html(kw)}</span>')
                    else:
                        # Unconfirmed - gold badge
                        badges.append(f'<span class="keyword-badge">{escape_html(kw)}</span>')
                        all_confirmed = False
                keywords_html = " ".join(badges)
            
            # Style based on whether keywords were found and confirmed
            if entry.has_keyword:
                if all_confirmed and entry.keywords_found:
                    # All confirmed - green
                    border_color = "#32CD32"  # Green
                    bg_color = "#1F2D1F"  # Darker green tint
                    alert_icon = "✓ "
                else:
                    # Unconfirmed - gold
                    border_color = "#FFD700"  # Gold
                    bg_color = "#2D2D1F"  # Darker gold tint
                    alert_icon = "🔔 "
            else:
                # Normal entry
                border_color = "#444"
                bg_color = "#1E1E1E"
                alert_icon = ""
            
            # Build compact HTML
            feed_html += (
                f'<div class="transcript-entry" style="border-left-color:{border_color};background-color:{bg_color};">'
                f'<div class="speaker-label" style="color:#888;">'
                f'{alert_icon}<span class="timestamp">{timestamp_str}</span> {keywords_html}'
                f'</div>'
                f'<div class="transcript-text">{highlighted_text}</div>'
                f'</div>'
            )
        
        feed_placeholder.markdown(feed_html, unsafe_allow_html=True)
    else:
        feed_placeholder.info("No transcriptions yet. Click 'Initialize' then 'Start' to begin.")
    
    # Auto-refresh mechanism
    if transcriber.is_running:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
