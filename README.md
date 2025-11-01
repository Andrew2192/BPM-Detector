# BPM Analyzer (Reference + Real-time Microphone)

A desktop app for analyzing BPM (beats per minute) of an audio file and comparing it with real-time microphone input. It provides interactive charts, play/pause/seek controls for both the reference audio and microphone recording, and exports BPM time series for further analysis.

## Features
- Analyze BPM of a selected audio file with visualizations.
- Interactive charts with seek bars and progress labels.
- Reference audio controls: play/pause, seek, reset to start.
- Real-time microphone monitoring with live BPM chart.
- Microphone playback controls: play/pause, seek, reset to start.
- Compare recorded microphone BPM against reference BPM.
- Export BPM time series (reference and microphone) to files.
- Clean UI styled with ttk, Matplotlib charts embedded via TkAgg.

## Requirements
- Python `3.11+` (tested with CPython 3.13 bytecode artifacts in repo)
- macOS (tested), Linux and Windows should also work with matching dependencies.
- System dependencies:
  - `PortAudio` for `sounddevice` (on macOS, install via Homebrew if needed: `brew install portaudio`).
  - A working audio device and microphone.
- Python packages (installed via `requirements.txt`):
  - `pygame`, `pydub`, `numpy`, `matplotlib`, `scipy`, `sounddevice`.

## Installation
1. Create and activate a virtual environment:
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
   - Windows (PowerShell):
     - `py -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. (macOS only) If microphone access prompts appear, allow access for Python.

## Run
- `python main.py`

## Usage
- Reference Audio
  - Click `Browse` to select an audio file.
  - Click `Analyze BPM` to process and visualize.
  - Use the reference section controls:
    - `▶/⏸` play/pause reference audio.
    - Drag the seek bar to move to a position; time label updates.
    - Click `↺` reset to the beginning; labels and slider return to `00:00`.
- Real-time Microphone BPM
  - Click the microphone icon `🎤` to start/stop monitoring.
  - While monitoring, the app records audio to a temporary WAV and displays live BPM timeseries.
  - Use the microphone section controls:
    - `▶/⏸` play/pause the recorded mic audio.
    - Drag the seek bar to navigate within the mic recording.
    - Click `↺` reset to the beginning; labels and slider return to `00:00`.
- Comparison & Export
  - Click `Show Detailed BPM Data` in either section for detailed plots.
  - Use `Compare BPM` to analyze differences between reference and microphone recordings.
  - Use export actions to save BPM time series to files.

## File Outputs
- Temporary playback files: `temp_playback_YYYYMMDDHHMMSS.wav` (reference), `temp_mic_playback_YYYYMMDDHHMMSS.wav` (mic).
- Saved microphone recordings: `mic_recording_YYYYMMDDHHMMSS.wav` (when monitoring is stopped and finalized).

## Project Structure
- `main.py` — entry point that initializes the UI (`BPMGUIApp.main()`).
- `analyzer.py` — GUI and playback logic, chart controls, seek/reset, mic monitoring.
- `bpm_core.py` — BPM analysis core algorithms.
- `bpm_visuals.py` — plotting helpers for BPM timeseries, distributions, heatmaps.
- `plot_config.py` — Matplotlib theme and style settings.
- `config.json` — app configuration.
- `requirements.txt` — Python dependencies.

## Troubleshooting
- Playback starts from the beginning after seek:
  - `pygame.mixer.music.set_pos()` can be unreliable for some formats; the app falls back to start when a format doesn’t support sample-accurate seeking. Converting to WAV improves reliability.
- Microphone reset warns about missing recording:
  - The reset control works even without a recording; it returns UI to `00:00`.
- No microphone detected or errors from `sounddevice`:
  - Ensure microphone permission is granted and a valid input device is selected.
  - On macOS, check System Settings → Privacy & Security → Microphone.
- Charts or labels don’t update while playing:
  - UI update runs on a timer; heavy CPU load may slow redraw. Reduce other tasks or close background apps.

## Development Notes
- Playback state is managed via:
  - `current_playback_file`, `playback_position`, `playing`, `last_update_time`.
  - Reference and mic contexts are separated to keep sliders, labels, and charts in sync.
- Seek handling prefers in-place position updates and falls back to restart when needed.
- Temporary files are cleaned up on exit; recorded mic files are preserved.

## License
- No explicit license provided. Please consult the repository owner before redistribution.