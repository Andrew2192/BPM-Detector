# Advanced BPM Analyzer

A sophisticated desktop application for analyzing BPM (beats per minute) of audio files and comparing it with real-time microphone input. Built with PySide6, it provides interactive charts, play/pause/seek controls, and AI-powered feedback.

## Features

- **Audio File Analysis**: Analyze BPM of any audio file with high accuracy visualization
- **Interactive Charts**: Real-time BPM variation charts with seek bars and progress labels
- **Audio Playback Controls**: 
  - Reference audio: play/pause, seek, reset to start
  - Microphone recording: play/pause, seek, reset to start
- **Real-time Microphone Monitoring**: Live BPM analysis with interactive chart
- **BPM Comparison**: Compare recorded microphone BPM against reference BPM
- **AI-powered Feedback**: Generate AI summaries using DeepSeek API
- **Clean Modern UI**: Styled with PySide6 and Matplotlib charts
- **Modular Architecture**: Well-organized codebase for easy maintenance and extension

## Requirements

- Python `3.11+` (tested with CPython 3.13)
- macOS (tested), Linux and Windows should also work with matching dependencies
- System dependencies:
  - `PortAudio` for microphone functionality (macOS: `brew install portaudio`)
  - Working audio device and microphone
- Python packages (install via `requirements.txt`):
  - PySide6, pygame, pydub, numpy, matplotlib, scipy, sounddevice, openpyxl, deepseek-sdk

## Installation

1. **Create and activate a virtual environment**:
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     py -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Permissions**:
   - macOS: Allow microphone access for Python when prompted
   - Windows/Linux: Ensure microphone permissions are granted

## Usage

### Run the Application
```bash
python main.py
```

### Basic Operations

#### Reference Audio Analysis
1. Click `Browse` to select an audio file
2. Click `Calculate BPM` to process and visualize
3. Use reference section controls:
   - `▶/⏸` Play/Pause reference audio
   - Drag the seek bar to navigate
   - Click `⟲` Reset to beginning

#### Real-time Microphone BPM
1. Click the microphone icon `🎤` to start/stop monitoring
2. View live BPM chart while recording
3. Use microphone section controls:
   - `▶/⏸` Play/Pause recorded audio
   - Drag the seek bar to navigate
   - Click `⟲` Reset to beginning

#### BPM Comparison
1. Analyze a reference audio file
2. Record some audio with the microphone
3. Click `Compare BPM` to view detailed comparison charts
4. Click `Generate` to get AI feedback using DeepSeek API

## Project Structure

```
bpm_analyzer-v2/
├── main.py                    # Application entry point
├── analyzer.py                # Main GUI application class
├── bpm_core.py                # Core BPM analysis algorithms
├── bpm_visuals.py             # Visualization functions for BPM data
├── audio_processing.py        # Audio file processing and conversion
├── playback.py                # Audio playback management
├── mic_recording.py           # Microphone recording and real-time BPM
├── charting.py                # Chart drawing and management
├── utils.py                   # Utility functions
├── plot_config.py             # Chart styling configuration
├── config.json                # Application configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Module Descriptions

### Core Modules
- **main.py**: Entry point that initializes the PySide6 application
- **analyzer.py**: Central controller for the GUI, manages all UI components and events
- **bpm_core.py**: Implements advanced BPM detection algorithms

### Audio Processing
- **audio_processing.py**: Handles audio file loading, conversion, and segmentation
- **playback.py**: Manages audio playback using pygame.mixer
- **mic_recording.py**: Handles microphone recording and real-time BPM analysis

### Visualization
- **charting.py**: Manages BPM variation charts and updates
- **bpm_visuals.py**: Creates BPM comparison charts, heatmaps, and distribution plots
- **plot_config.py**: Defines chart styling and themes

### Utilities
- **utils.py**: Provides common functions like time formatting and file validation

## Technical Architecture

### GUI Framework
- **PySide6**: Modern, cross-platform GUI framework
- **Matplotlib**: Embedded charts with Qt5Agg backend
- **Threading**: Background threads for audio processing and BPM analysis

### Audio Processing
- **pydub**: Audio file loading and conversion
- **pygame.mixer**: Audio playback
- **pyaudio**: Microphone recording
- **scipy.signal**: Audio signal processing for BPM detection

### BPM Analysis Algorithm
1. **Beat Detection**: Uses spectral flux and energy envelope analysis
2. **Interval Calculation**: Calculates beat intervals
3. **BPM Candidates**: Generates BPM candidates from intervals
4. **Outlier Filtering**: Removes outliers using IQR method
5. **Smoothing**: Applies moving average for better visualization

## Troubleshooting

### Common Issues
1. **Microphone not working**:
   - Check system microphone permissions
   - Ensure PortAudio is installed correctly
   - Restart the application after granting permissions

2. **Audio playback issues**:
   - Some formats may not support seeking, try converting to WAV first
   - Ensure pygame.mixer is properly initialized

3. **DeepSeek API errors**:
   - Check that DEEPSEEK_API_KEY is set in environment variables or config.json
   - Ensure network connectivity

4. **Chart updates slow**:
   - Reduce other CPU-intensive tasks
   - Ensure Matplotlib is using the correct backend

### Resetting Application State
If the application becomes unresponsive or displays unexpected behavior:
1. Click the microphone button to stop recording
2. Stop all audio playback
3. Close and restart the application

## Development Notes

### Code Style
- Follows PEP 8 guidelines
- Modular design with single responsibility principle
- Well-documented functions and classes

### Adding New Features
1. Create a new module if the feature is self-contained
2. Update analyzer.py to integrate the new feature
3. Add tests if applicable
4. Update requirements.txt if new dependencies are added

### Testing
- Manual testing recommended for GUI functionality
- Unit tests can be added for core algorithms
- Test with various audio formats and microphone inputs

## License

This project is open source. Please consult the repository owner before redistribution.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

- Built with PySide6 for the modern UI
- Uses Matplotlib for beautiful data visualization
- Leverages DeepSeek API for AI-powered feedback
- Inspired by various BPM analysis tools and techniques

---

Enjoy analyzing BPM with this advanced tool! 🎵
