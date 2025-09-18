"""
Whisper-based live STT repeater with configurable VAD and model/runtime options.

Notes:
- Device "cuda" requires a CUDA-enabled PyTorch build; falls back to CPU if unavailable.
- Language uses ISO-639-1 codes (e.g., "pt", "en", "es", "fr").
- Non-speaking cut-off is controlled by Recognizer.non_speaking_duration; pause detection by pause_threshold.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import numpy as np
import pyttsx3
import speech_recognition as sr
import torch
import whisper

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

DEFAULT_MODEL = "small"                 # CLI: --model
DEFAULT_DEVICE = "cpu"                  # CLI: --device {cpu,cuda}
DEFAULT_LOG_LEVEL = "info"              # CLI: --log-level
DEFAULT_LANGUAGE = "pt"                 # CLI: --language

DEFAULT_TIMEOUT = 5.0                   # CLI: --timeout
DEFAULT_AMBIENT_DURATION = 1.0          # CLI: --ambient-duration
PAUSE_THRESHOLD_SECONDS = 2.0           # CLI: --pause-threshold
NON_SPEAKING_DURATION = 2.0             # CLI: --non-speaking-duration

DEFAULT_SAMPLE_RATE = 16000             # Internal audio resample rate (Hz)

# Whisper decoding heuristics (exposed as flags)
DEFAULT_TEMPERATURE = 0.0               # CLI: --temperature
DEFAULT_NO_SPEECH_THRESHOLD = 0.45      # CLI: --no-speech-threshold
DEFAULT_LOGPROB_THRESHOLD = -1.0        # CLI: --logprob-threshold
DEFAULT_CONDITION_ON_PREV = False       # CLI: --condition-on-previous-text
DEFAULT_FP16_MODE = "auto"              # CLI: --fp16 {auto,true,false}

# Energy adaptation knobs (kept as sensible defaults; usually you won’t need to change)
DEFAULT_DYNAMIC_ENERGY = True
DEFAULT_ENERGY_DAMPING = 0.15
DEFAULT_ENERGY_RATIO = 1.5

def parse_log_level(level_str: str) -> int:
    """
    Map user-provided log level string to logging.* constant, case-insensitive.
    """
    mapping = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET,
    }
    key = level_str.strip().lower()
    if key not in mapping:
        raise ValueError(f"Invalid log level: {level_str}")
    return mapping[key]


def decide_device(requested: str) -> str:
    """
    Return a safe device string. If 'cuda' is requested but unavailable, fall back to 'cpu' with a warning.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        logging.getLogger("Device").warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return requested


def resolve_fp16(mode: str, device: str) -> bool:
    """
    Resolve fp16 behavior based on mode and device.
    - 'auto': True on CUDA, False on CPU
    - 'true' / 'false': force-enable/disable
    """
    key = mode.strip().lower()
    if key == "auto":
        return device.startswith("cuda")
    if key in ("true", "t", "1", "yes", "y"):
        return True
    if key in ("false", "f", "0", "no", "n"):
        return False
    raise ValueError(f"Invalid fp16 mode: {mode} (use 'auto', 'true', or 'false')")


def setup_logging(level_name: str) -> None:
    """
    Configure a root logger with a single stdout handler and a uniform format.
    """
    level = parse_log_level(level_name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


class SpeechRepeater:
    """
    Captures speech from the default microphone, transcribes with Whisper, and speaks the result.

    The recognizer uses energy-based VAD:
    - pause_threshold          : seconds of silence that mark the end of a phrase
    - non_speaking_duration    : minimal silence duration to be considered non-speaking
    Both interacting; typically set pause_threshold ≈ 2.0 and non_speaking_duration ≈ 2.0 for "stop after ~2s silence."
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        language: str = DEFAULT_LANGUAGE,
        timeout: float = DEFAULT_TIMEOUT,
        ambient_duration: float = DEFAULT_AMBIENT_DURATION,
        pause_threshold: float = PAUSE_THRESHOLD_SECONDS,
        non_speaking_duration: float = NON_SPEAKING_DURATION,
        temperature: float = DEFAULT_TEMPERATURE,
        no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD,
        logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
        condition_on_previous_text: bool = DEFAULT_CONDITION_ON_PREV,
        fp16_mode: str = DEFAULT_FP16_MODE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        dynamic_energy: bool = DEFAULT_DYNAMIC_ENERGY,
        energy_damping: float = DEFAULT_ENERGY_DAMPING,
        energy_ratio: float = DEFAULT_ENERGY_RATIO,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.device = decide_device(device)
        self.model_name = model_name
        self.language = language
        self.timeout = float(timeout)
        self.ambient_duration = float(ambient_duration)
        self.pause_threshold = float(pause_threshold)
        self.non_speaking_duration = float(non_speaking_duration)
        self.temperature = float(temperature)
        self.no_speech_threshold = float(no_speech_threshold)
        self.logprob_threshold = float(logprob_threshold)
        self.condition_on_previous_text = bool(condition_on_previous_text)
        self.sample_rate = int(sample_rate)

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = self.pause_threshold
        self.recognizer.non_speaking_duration = self.non_speaking_duration
        self.recognizer.dynamic_energy_threshold = bool(dynamic_energy)
        self.recognizer.dynamic_energy_adjustment_damping = float(energy_damping)
        self.recognizer.dynamic_energy_ratio = float(energy_ratio)

        self.fp16 = resolve_fp16(fp16_mode, self.device)

        self.logger.info("Loading Whisper model=%s on device=%s (fp16=%s)", self.model_name, self.device, self.fp16)
        self.whisper = whisper.load_model(self.model_name, device=self.device)

        self.engine = pyttsx3.init()

    def speak(self, text: str) -> None:
        """
        Say the provided text using the system TTS engine.
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_once(self) -> sr.AudioData:
        """
        Capture a single utterance from the microphone, ending after VAD detects ~pause_threshold seconds of silence.
        Raises:
            sr.WaitTimeoutError: if the user does not start speaking within `self.timeout` seconds.
        """
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=self.ambient_duration)
            audio = self.recognizer.listen(source, timeout=self.timeout, phrase_time_limit=None)
        return audio

    def _audio_to_float32(self, audio: sr.AudioData) -> np.ndarray:
        """
        Convert SpeechRecognition AudioData to mono float32 PCM at self.sample_rate in range [-1, 1].
        """
        pcm16 = audio.get_raw_data(convert_rate=self.sample_rate, convert_width=2)
        return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe_whisper(self, audio: sr.AudioData) -> str:
        """
        Transcribe captured audio using Whisper. Returns a lowercased string (possibly empty if nothing was decoded).
        """
        samples = self._audio_to_float32(audio)
        result = self.whisper.transcribe(
            audio=samples,
            language=self.language,
            fp16=self.fp16,
            condition_on_previous_text=self.condition_on_previous_text,
            temperature=self.temperature,
            no_speech_threshold=self.no_speech_threshold,
            logprob_threshold=self.logprob_threshold,
        )
        return result.get("text", "").strip().lower()

    def run(self) -> None:
        """
        Main loop: listen → transcribe → log/speak. Ctrl+C to exit.
        """
        self.logger.info(
            "Starting | model=%s device=%s lang=%s timeout=%.1fs ambient=%.1fs pause=%.1fs non_speaking=%.1fs",
            self.model_name,
            self.device,
            self.language,
            self.timeout,
            self.ambient_duration,
            self.pause_threshold,
            self.non_speaking_duration,
        )
        while True:
            try:
                audio = self.listen_once()
                text = self.transcribe_whisper(audio)
                if text:
                    self.logger.info("Heard: %s", text)
                    self.speak(text)
                else:
                    self.logger.info("No transcription returned")
            except sr.WaitTimeoutError:
                self.logger.info("No speech within %.1fs; continuing", self.timeout)
                continue
            except KeyboardInterrupt:
                self.logger.info("Interrupted by user. Exiting.")
                break
            except Exception:
                self.logger.exception("Unexpected error")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser with all supported flags and defaults.
    """
    p = argparse.ArgumentParser(description="Whisper STT repeater with configurable VAD and runtime.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size/name (tiny, base, small, medium, large).")
    p.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE, help="Compute device.")
    p.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["critical", "error", "warning", "info", "debug", "notset"],
        help="Logging verbosity.",
    )
    p.add_argument("--language", default=DEFAULT_LANGUAGE, help="ISO-639-1 language code (e.g., pt, en, es).")
    p.add_argument(
        "--non-speaking-duration",
        type=float,
        default=NON_SPEAKING_DURATION,
        help="Seconds of minimal silence to consider as non-speaking (VAD).",
    )
    p.add_argument(
        "--pause-threshold",
        type=float,
        default=PAUSE_THRESHOLD_SECONDS,
        help="Seconds of silence that end a phrase.",
    )
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Seconds to wait for speech to start.")
    p.add_argument(
        "--ambient-duration",
        type=float,
        default=DEFAULT_AMBIENT_DURATION,
        help="Seconds for ambient noise calibration.",
    )
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Whisper decoding temperature.")
    p.add_argument(
        "--no-speech-threshold",
        type=float,
        default=DEFAULT_NO_SPEECH_THRESHOLD,
        help="Whisper prob. threshold to treat a segment as no-speech.",
    )
    p.add_argument(
        "--logprob-threshold",
        type=float,
        default=DEFAULT_LOGPROB_THRESHOLD,
        help="Minimum average logprob for decoding to be accepted.",
    )
    p.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        default=DEFAULT_CONDITION_ON_PREV,
        help="If set, condition on previous text for next segment.",
    )
    p.add_argument(
        "--fp16",
        default=DEFAULT_FP16_MODE,
        choices=["auto", "true", "false"],
        help="Use 16-bit floats: auto uses True on CUDA and False on CPU.",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Internal resample rate for microphone audio.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    """
    Parse CLI options, configure logging, and launch the app.
    """
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level)

    app = SpeechRepeater(
        model_name=args.model,
        device=args.device,
        language=args.language,
        timeout=args.timeout,
        ambient_duration=args.ambient_duration,
        pause_threshold=args.pause_threshold,
        non_speaking_duration=args.non_speaking_duration,
        temperature=args.temperature,
        no_speech_threshold=args.no_speech_threshold,
        logprob_threshold=args.logprob_threshold,
        condition_on_previous_text=args.condition_on_previous_text,
        fp16_mode=args.fp16,
        sample_rate=args.sample_rate,
    )
    app.run()


if __name__ == "__main__":
    main()
