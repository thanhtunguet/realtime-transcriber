#!/usr/bin/env python3
import argparse
import queue
import threading
import sys
import time
from collections import deque
from datetime import datetime
import numpy as np
import sounddevice as sd
import mlx_whisper

# Map friendly model sizes to pre-converted MLX models on the Hugging Face Hub.
MLX_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

class RealtimeTranscriber:
    # ANSI color codes for cycling through colors
    COLORS = [
        '\033[91m',  # Red
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[94m',  # Blue
        '\033[95m',  # Magenta
        '\033[96m',  # Cyan
        '\033[97m',  # White
        '\033[38;5;214m',  # Bright Orange
    ]
    RESET = '\033[0m'

    def __init__(
        self,
        model_size: str = "base",
        chunk_duration: float = 3.0,
        sample_rate: int = 16000,
        refinement_history: int = 3,
    ):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.refinement_history = max(1, refinement_history)
        self.chunk_history: deque[np.ndarray] = deque(maxlen=self.refinement_history)
        self.committed_text: str = ""  # All text that has been output to user
        self.last_refined_text: str = ""  # Last refined text for comparison
        self._live_line_active = False
        self._live_line_length = 0
        self._color_index = 0

        # Resolve the MLX model repo. MLX Whisper runs on the Apple Silicon GPU
        # natively, so there is no device/fp16 selection to make.
        self.model_repo = MLX_MODEL_REPOS.get(model_size, model_size)

        print(f"Loading MLX Whisper model '{self.model_repo}'...")
        # Warm up / trigger download so the first live chunk isn't penalized.
        mlx_whisper.transcribe(
            np.zeros(self.sample_rate, dtype=np.float32),
            path_or_hf_repo=self.model_repo,
        )

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_recording = False

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input stream"""
        if status:
            print(f"Audio callback status: {status}")

        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]

        self.audio_queue.put(audio_data.copy())

    def get_system_audio_device(self):
        """Find system audio device for macOS"""
        devices = sd.query_devices()

        # Look for system audio devices (typically contain "BlackHole" or "Soundflower" on macOS)
        system_devices = []
        for i, device in enumerate(devices):
            name = device['name'].lower()
            if ('blackhole' in name or 'soundflower' in name or
                'loopback' in name or 'virtual' in name):
                if device['max_input_channels'] > 0:
                    system_devices.append((i, device))

        if system_devices:
            device_id, device_info = system_devices[0]
            print(f"Using system audio device: {device_info['name']}")
            return device_id
        else:
            print("No virtual audio device found. Install BlackHole or Soundflower for system audio capture.")
            print("Using default input device instead...")
            return None

    def start_recording(self, use_system_audio: bool = True):
        """Start recording audio"""
        device_id = None
        if use_system_audio:
            device_id = self.get_system_audio_device()

        self.is_recording = True

        # Start audio stream
        self.stream = sd.InputStream(
            device=device_id,
            samplerate=self.sample_rate,
            channels=1 if device_id is None else None,
            callback=self.audio_callback,
            blocksize=1024,
        )

        self.stream.start()
        print(f"Recording started... (sample rate: {self.sample_rate} Hz)")

    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self._clear_live_line()
        print("Recording stopped.")

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a chunk of audio"""
        try:
            # Ensure audio is float32 and normalized
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            # Normalize audio
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))

            # Transcribe with MLX Whisper (runs on the Apple Silicon GPU)
            result = mlx_whisper.transcribe(
                audio_chunk,
                path_or_hf_repo=self.model_repo,
                language=None,  # Auto-detect
                task="transcribe",
            )

            text = result.get("text", "").strip()
            return text

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def _get_next_color(self) -> str:
        """Get the next color in the cycle."""
        color = self.COLORS[self._color_index]
        return color

    def _advance_color(self) -> None:
        """Advance to the next color in the cycle."""
        self._color_index = (self._color_index + 1) % len(self.COLORS)

    def _normalize_word(self, word: str) -> str:
        """Normalize a word for comparison by lowercasing and stripping punctuation."""
        return word.lower().strip('.,!?;:\'"()[]{}–—-')

    def _find_overlap_length(self, committed_words: list[str], new_words: list[str]) -> int:
        """
        Find how many words at the start of new_words overlap with the end of committed_words.
        Uses fuzzy matching to handle minor transcription differences.
        """
        if not committed_words or not new_words:
            return 0

        # Check overlap lengths from longest to shortest
        max_overlap = min(len(committed_words), len(new_words), 30)  # Limit search

        for overlap_len in range(max_overlap, 2, -1):  # Require at least 3 words overlap
            committed_suffix = committed_words[-overlap_len:]
            new_prefix = new_words[:overlap_len]

            # Count matching words
            matches = 0
            for w1, w2 in zip(committed_suffix, new_prefix):
                if self._normalize_word(w1) == self._normalize_word(w2):
                    matches += 1

            # Require 70% match for this overlap length
            if matches >= overlap_len * 0.7:
                return overlap_len

        return 0

    def _diff_refined_text(self, refined_text: str) -> tuple[str, bool]:
        """
        Return the portion of text that is new compared to ALL committed text.
        Uses word-based overlap detection to handle sliding window overlap.
        """
        if not refined_text:
            return "", False

        refined_text = refined_text.strip()

        # First time - no committed text yet
        if not self.committed_text:
            self.committed_text = refined_text
            self.last_refined_text = refined_text
            return refined_text, False

        # Split into words for comparison
        committed_words = self.committed_text.split()
        refined_words = refined_text.split()
        refined_original_words = refined_text.split()  # Keep original for output

        # Find overlap between end of committed text and start of refined text
        overlap_len = self._find_overlap_length(committed_words, refined_words)

        if overlap_len > 0:
            # We found overlap - only output words after the overlap
            new_words = refined_original_words[overlap_len:]
            if new_words:
                new_text = ' '.join(new_words)
                # Append to committed text
                self.committed_text = self.committed_text + ' ' + new_text
                self.last_refined_text = refined_text
                return new_text, True
            else:
                # Everything was overlap, no new content
                self.last_refined_text = refined_text
                return "", False
        else:
            # No overlap found - this might be a new segment or transcription jump
            # Check if the refined text is substantially contained in committed text
            # (to avoid re-outputting the same content with different wording)

            # If more than 70% of the refined text words already appeared in recent committed text, skip
            refined_word_set = set(self._normalize_word(w) for w in refined_words if len(w) > 2)
            committed_word_set = set(self._normalize_word(w) for w in committed_words[-50:] if len(w) > 2)

            if refined_word_set and committed_word_set:
                overlap_ratio = len(refined_word_set & committed_word_set) / len(refined_word_set)
                if overlap_ratio > 0.7:  # 70% of words already appeared
                    self.last_refined_text = refined_text
                    return "", False

            # New content - append to committed
            self.committed_text = self.committed_text + ' ' + refined_text
            self.last_refined_text = refined_text
            return refined_text, False

    def _print_live_line(self, timestamp: str, text: str) -> None:
        """Show immediate transcription while waiting for refinement."""
        color = self._get_next_color()
        line = f"{color}[{timestamp}] (live) {text}{self.RESET}"
        # Store length without ANSI codes for proper clearing
        line_plain = f"[{timestamp}] (live) {text}"
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        self._live_line_active = True
        self._live_line_length = len(line_plain)

    def _replace_live_line(self, timestamp: str, text: str, is_refinement: bool) -> None:
        """Replace the live line with the refined result."""
        if not text.strip():
            self._clear_live_line()
            return

        color = self._get_next_color()
        prefix = "Refined: " if is_refinement else ""
        line = f"{color}[{timestamp}] {prefix}{text}{self.RESET}"
        if self._live_line_active:
            sys.stdout.write("\r" + " " * self._live_line_length + "\r")
            sys.stdout.flush()
        print(line, flush=True)
        self._live_line_active = False
        self._live_line_length = 0
        self._advance_color()

    def _clear_live_line(self) -> None:
        """Clear any live line when no refined text is produced."""
        if not self._live_line_active:
            return
        sys.stdout.write("\r" + " " * self._live_line_length + "\r")
        sys.stdout.flush()
        self._live_line_active = False
        self._live_line_length = 0

    def process_audio(self):
        """Process audio chunks from the queue"""
        audio_buffer = np.array([], dtype=np.float32)
        self.chunk_history.clear()
        self.committed_text = ""
        self.last_refined_text = ""
        self._live_line_active = False
        self._live_line_length = 0

        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio data from queue (with timeout)
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, audio_data])

                # Process when we have enough audio
                if len(audio_buffer) >= self.chunk_size:
                    # Extract chunk
                    chunk = audio_buffer[:self.chunk_size]
                    audio_buffer = audio_buffer[self.chunk_size:]

                    # Check if chunk has enough audio content
                    if np.max(np.abs(chunk)) > 0.01:  # Minimum volume threshold
                        chunk = chunk.astype(np.float32, copy=True)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        live_text = self.transcribe_chunk(chunk)
                        if live_text:
                            self._print_live_line(timestamp, live_text)
                        else:
                            self._clear_live_line()

                        self.chunk_history.append(chunk)
                        combined_audio = np.concatenate(list(self.chunk_history))
                        text = self.transcribe_chunk(combined_audio)

                        if text:
                            new_text, is_refinement = self._diff_refined_text(text)
                            if self._live_line_active:
                                final_text = new_text if new_text else (live_text or "")
                                self._replace_live_line(timestamp, final_text, is_refinement)
                            elif new_text:
                                color = self._get_next_color()
                                prefix = "Refined: " if is_refinement else ""
                                print(f"{color}[{timestamp}] {prefix}{new_text}{self.RESET}")
                                self._advance_color()
                        else:
                            if self._live_line_active:
                                self._replace_live_line(timestamp, live_text or "", False)

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break

    def run(self, use_system_audio: bool = True):
        """Run the real-time transcriber"""
        try:
            self.start_recording(use_system_audio)

            # Start processing thread
            process_thread = threading.Thread(target=self.process_audio)
            process_thread.daemon = True
            process_thread.start()

            print("Real-time transcription started. Press Ctrl+C to stop...")
            print("=" * 60)

            # Keep main thread alive
            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping transcription...")
        finally:
            self.stop_recording()
            # Wait a bit for the processing thread to finish
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                # If interrupted during cleanup, just exit immediately
                pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using MLX Whisper"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help=(
            "Whisper model size (tiny, base, small, medium, large, large-v2, "
            "large-v3, large-v3-turbo) or a Hugging Face MLX repo id (default: base)"
        ),
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Audio chunk duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--refinement-chunks",
        type=int,
        default=3,
        help="Number of recent audio chunks to use for refinement (default: 3)",
    )
    parser.add_argument(
        "--use-mic",
        action="store_true",
        help="Use microphone instead of system audio",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    transcriber = RealtimeTranscriber(
        model_size=args.model,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate,
        refinement_history=args.refinement_chunks,
    )

    use_system_audio = not args.use_mic
    transcriber.run(use_system_audio=use_system_audio)

if __name__ == "__main__":
    main()
