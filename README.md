# BPM-Detector

A real-time BPM detector with Python 3 (UTF-8 encoding). It uses key libraries: numpy 1.24.3 for numerical/signal processing, pygame 2.5.2 for audio playback, and pydub 0.25.1 for format conversion; Python’s standard libraries (tkinter, threading, struct, os/sys) handle GUI building, background task processing, WAV parsing, and system operations.

Core tech includes a BPM detection algorithm (energy + zero-crossing rate), digital signal processing (frame segmentation, peak detection), IQR-based outlier filtering, and data smoothing for stable BPM. It enables audio play/pause/position control (via pygame) and WAV conversion, while tkinter builds a responsive GUI with Chinese font support and real-time status updates.

Structured with OOP (BPMAnalyzer for BPM calculation, BPMGUIApp for UI), multi-threading (background audio analysis), layered design, and error handling. It supports 6 audio formats, auto-analyzes BPM (with tempo descriptions), shows playback progress, visualizes BPM changes, and has pause/resume.
