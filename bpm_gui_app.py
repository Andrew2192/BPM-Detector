import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import struct
import numpy as np
import pygame
from pydub import AudioSegment
from pygame import font

class BPMAnalyzer:
    """
    BPM Analyzer class, responsible for beat detection and BPM calculation from audio data
    """
    
    def __init__(self, frame_size=2048, hop_size=512):
        """
        Initialize BPM analyzer with specified parameters
        
        Parameters:
            frame_size: Audio frame size for analysis
            hop_size: Hop size between consecutive frames
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
    
    def analyze_audio_data(self, audio_data, sample_rate):
        """
        Analyze audio data to detect beats and calculate BPM
        
        Parameters:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Detected BPM value or None if detection failed
        """
        # Check if audio data is valid
        if len(audio_data) < self.frame_size:
            return None
        
        # Detect beats
        beats = self._detect_beats(audio_data, sample_rate)
        
        if not beats:
            return None
        
        # Calculate BPM
        bpm = self._calculate_bpm(beats, sample_rate)
        
        return bpm
    
    def _bpm_to_category(self, bpm):
        """
        Classify BPM value into music categories
        
        Parameters:
            bpm: BPM value
            
        Returns:
            Category description of the BPM
        """
        if bpm < 60:
            return "Very slow (meditation, relaxation)"
        elif bpm < 80:
            return "Slow (ballads, folk, slow rock)"
        elif bpm < 100:
            return "Moderately slow (soft pop)"
        elif bpm < 120:
            return "Medium (standard pop)"
        elif bpm < 140:
            return "Fast (electronic, dance, rock)"
        elif bpm < 160:
            return "Very fast (House, Trance)"
        elif bpm < 180:
            return "Extremely fast (Hardstyle, Techno)"
        else:
            return "Ultra fast (hardcore, metal)"
    
    def analyze_audio_file(self, file_path):
        """
        Analyze audio file to calculate its BPM
        
        Parameters:
            file_path: Path to the audio file
            
        Returns:
            BPM value or None if analysis failed
        """
        try:
            # Read audio file using pydub
            audio = AudioSegment.from_file(file_path)
            
            # Convert to mono if stereo
            if audio.channels == 2:
                audio = audio.set_channels(1)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Normalize
            max_val = 2 ** (audio.sample_width * 8 - 1)
            samples = samples / max_val
            
            # Analyze BPM
            bpm = self.analyze_audio_data(samples, audio.frame_rate)
            
            return bpm
        except Exception as e:
            print(f"Error analyzing audio file: {str(e)}")
            return None
    
    def _detect_beats(self, audio_data, sample_rate):
        """
        Detect beats in audio data using energy and zero-crossing rate
        
        Parameters:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of beat positions in samples
        """
        beats = []
        
        # Process audio in frames
        for i in range(0, len(audio_data) - self.frame_size, self.hop_size):
            frame = audio_data[i:i + self.frame_size]
            
            # Calculate energy (E = 1/N Σx_i²)
            energy = np.sum(frame ** 2) / len(frame)
            
            # Calculate zero-crossing rate (ZCR = 1/2 Σ|sign(x_{i+1}) - sign(x_i)|)
            zero_crossing = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            
            # Apply threshold to detect beats
            # Calculate a more reasonable energy threshold based on recent frames
            if i > 0:
                # Get recent frames to calculate a more accurate average energy
                recent_frames_start = max(0, i - 5 * self.hop_size)
                recent_frames = audio_data[recent_frames_start:i + self.frame_size]
                # Process in frames to calculate average energy
                recent_energies = []
                for j in range(0, len(recent_frames) - self.frame_size, self.hop_size):
                    recent_frame = recent_frames[j:j + self.frame_size]
                    recent_energies.append(np.sum(recent_frame ** 2) / len(recent_frame))
                
                if recent_energies:
                    energy_threshold = 1.2 * np.mean(recent_energies)
                else:
                    energy_threshold = 0.01  # Default threshold if no recent frames
            else:
                energy_threshold = 0.01  # Default threshold for first frame
                
            zero_crossing_threshold = 30  # Lowered threshold for better detection
            
            # Detect peaks in energy
            if i > 0 and i + self.frame_size + self.hop_size < len(audio_data):
                prev_frame = audio_data[i - self.hop_size:i - self.hop_size + self.frame_size]
                next_frame = audio_data[i + self.hop_size:i + self.hop_size + self.frame_size]
                
                prev_energy = np.sum(prev_frame ** 2) / len(prev_frame)
                next_energy = np.sum(next_frame ** 2) / len(next_frame)
                
                # Check if current frame is a local peak and exceeds thresholds
                if (energy > prev_energy and energy > next_energy and 
                    energy > energy_threshold and zero_crossing > zero_crossing_threshold):
                    beats.append(i)
        
        return beats
    
    def _calculate_bpm(self, beat_positions, sample_rate):
        """
        Calculate BPM from detected beat positions
        
        Parameters:
            beat_positions: List of beat positions in samples
            sample_rate: Sample rate of the audio
            
        Returns:
            Calculated BPM value
        """
        if len(beat_positions) < 2:
            return None
        
        # Calculate intervals between consecutive beats in seconds
        intervals = []
        for i in range(1, len(beat_positions)):
            # Calculate time difference between consecutive beats
            time_diff = (beat_positions[i] - beat_positions[i-1]) / sample_rate
            intervals.append(time_diff)
        
        # Filter outliers using IQR method
        if intervals:
            # Calculate quartiles
            q1 = np.percentile(intervals, 25)
            q3 = np.percentile(intervals, 75)
            iqr = q3 - q1
            
            # Define outlier boundaries
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter out outliers
            filtered_intervals = [interval for interval in intervals if lower_bound <= interval <= upper_bound]
            
            # If all intervals were filtered out, use the original intervals
            if not filtered_intervals:
                filtered_intervals = intervals
            
            # Calculate BPM: BPM = 60 / average_interval
            if filtered_intervals:
                avg_interval = np.mean(filtered_intervals)
                bpm = 60 / avg_interval
                
                # Ensure BPM is within a reasonable range (40-220 BPM)
                if 40 <= bpm <= 220:
                    return bpm
        
        return None

class BPMGUIApp:
    """
    GUI application for BPM detection and audio playback
    """
    
    def __init__(self, root):
        """
        Initialize GUI application
        
        Parameters:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Audio BPM Analyzer")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Configure font settings
        pygame.font.init()
        # Set default font for pygame
        self.font = pygame.font.SysFont(None, 10)
        
        # Set fonts
        self.style = ttk.Style()
        try:
            self.style.configure(
                "TLabel",
                font=("Arial", 10)
            )
            self.style.configure(
                "TButton",
                font=("Arial", 10)
            )
        except Exception as e:
            print(f"Error setting fonts: {str(e)}")
        
        # Initialize variables
        self.audio_file = None
        self.analyzer = BPMAnalyzer()
        self.bpm_result = None
        self.is_playing = False
        self.stop_playback = False
        self.current_position = 0
        self.analysis_thread = None
        self.time_bpm_pairs = []  # Store time points and BPM values
        
        # Initialize UI variables
        self.bpm_var = tk.StringVar(value="--")
        self.bpm_desc_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.time_var = tk.StringVar(value="00:00 / 00:00")  # Ensure correct initial time format
        self.file_path_var = tk.StringVar()
        
        # Initialize pygame
        pygame.init()
        pygame.mixer.init()
        
        # Create UI
        self._create_widgets()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_widgets(self):
        """
        Create GUI components with optimized layout and visual display
        """
        # Set window theme color
        self.root.configure(bg="#f0f0f0")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section - more modern design
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Use grid layout for file selection area
        file_frame.columnconfigure(1, weight=1)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=5)
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_btn.grid(row=0, column=1, sticky="e", pady=5)
        
        # Analysis results section - two-column layout
        result_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Left and right columns
        left_col = ttk.Frame(result_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_col = ttk.Frame(result_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, ipady=5)
        
        # Left side - BPM results display
        bpm_label_frame = ttk.Frame(left_col)
        bpm_label_frame.pack(pady=20)
        
        ttk.Label(bpm_label_frame, text="Detected BPM: ", font=("Arial", 12)).pack(side=tk.LEFT)
        ttk.Label(bpm_label_frame, textvariable=self.bpm_var, font= ("Arial", 32, "bold"), foreground="#0078d7").pack(side=tk.LEFT)
        
        # BPM description label
        self.bpm_desc_var = tk.StringVar(value="")
        bpm_desc_label = ttk.Label(left_col, textvariable=self.bpm_desc_var, font=("Arial", 10), wraplength=300)
        bpm_desc_label.pack(pady=10)
        
        # Status and progress area
        status_frame = ttk.Frame(left_col)
        status_frame.pack(fill=tk.X, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=300)
        progress_bar.pack(pady=(0, 5), fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="#333333", font=("Arial", 10))
        status_label.pack(pady=(5, 0), anchor="w")
        
        # Right side - reserved for chart
        self.chart_frame = ttk.LabelFrame(right_col, text="BPM Visualization", padding="10")
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chart placeholder label
        self.chart_placeholder = ttk.Label(self.chart_frame, text="BPM chart will be displayed after analysis", foreground="#777777", font=("Arial", 10))
        self.chart_placeholder.pack(expand=True, fill=tk.BOTH)
        
        # Function buttons area
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Use grid layout for buttons
        control_frame.columnconfigure(4, weight=1)  # Make last column expand
        
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_playback, width=10)
        self.play_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_playback, state=tk.DISABLED, width=10)
        self.reset_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.show_timeseries_button = ttk.Button(control_frame, text="Show Detailed Data", command=self.show_bpm_timeseries, state=tk.DISABLED, width=20)
        self.show_timeseries_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Time display
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        time_label = ttk.Label(control_frame, textvariable=self.time_var, font=("Arial", 10, "bold"))
        time_label.grid(row=0, column=3, padx=10, pady=5, sticky="w")
        
        # Help text
        help_text = (
            "Instructions:\n"
            "1. Click 'Browse' to select an audio file\n"
            "2. The program will automatically analyze the audio and display BPM value\n"
            "3. You can play the audio to verify BPM accuracy\n"
            "4. Click 'Show Detailed Data' to view BPM changes throughout the audio"
        )
        help_frame = ttk.LabelFrame(main_frame, text="Help", padding="10")
        help_frame.pack(fill=tk.BOTH, expand=True)
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT, font=("Arial", 9), foreground="#555555")
        help_label.pack(fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        """
        Browse and select an audio file
        """
        file_types = [
            ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.aac *.m4a"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=file_types
        )
        
        if file_path:
            self.audio_file = file_path
            self.file_path_var.set(file_path)
            self.status_var.set("Analyzing audio...")
            self.bpm_var.set("--")
            self.show_timeseries_button.config(state=tk.DISABLED)
            self.time_bpm_pairs = []  # Clear previous data
            
            # Clear cached audio duration
            if hasattr(self, 'audio_duration'):
                delattr(self, 'audio_duration')
            
            # Immediately set initial time display in main thread
            self.time_var.set("00:00 / 00:00")
            
            # Calculate audio duration in main thread
            self._calculate_and_display_duration()
            
            # Analyze file in background thread
            self.analysis_thread = threading.Thread(target=self._analyze_file_thread)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
    def _calculate_and_display_duration(self):
        """
        Calculate and display the total duration of the audio file (executed in main thread)
        """
        print(f"Starting to calculate duration for file {self.audio_file}")
        try:
            # First try generic method using pydub for all formats
            try:
                print("Attempting to calculate duration with pydub")
                audio = AudioSegment.from_file(self.audio_file)
                self.audio_duration = len(audio) / 1000  # Convert to seconds
                print(f"Successfully calculated duration with pydub: {self.audio_duration} seconds")
            except Exception as pydub_error:
                # If pydub fails and it's a WAV file, try file header parsing
                if self.audio_file.lower().endswith('.wav'):
                    print(f"pydub failed, trying WAV file header parsing: {str(pydub_error)}")
                    try:
                        with open(self.audio_file, 'rb') as f:
                            # Check if file header is RIFF/WAVE format
                            header = f.read(12)
                            if len(header) < 12 or header[:4] != b'RIFF' or header[8:12] != b'WAVE':
                                raise ValueError("Invalid WAV file format")
                            
                            # Look for fmt and data chunks
                            while True:
                                # Read chunk header
                                subchunk_id = f.read(4)
                                if not subchunk_id:
                                    raise ValueError("Data chunk not found")
                                
                                subchunk_size = struct.unpack('<I', f.read(4))[0]
                                
                                if subchunk_id == b'fmt ':
                                    # Read fmt chunk content
                                    fmt_data = f.read(subchunk_size)
                                    if len(fmt_data) >= 16:
                                        channels = struct.unpack('<H', fmt_data[2:4])[0]
                                        sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                                        bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                                elif subchunk_id == b'data':
                                    # Read data chunk size
                                    data_size = subchunk_size
                                    break
                                else:
                                    # Skip other chunks
                                    f.seek(subchunk_size, 1)
                            
                            # Calculate duration (seconds)
                            self.audio_duration = data_size / (sample_rate * channels * (bits_per_sample / 8))
                            print(f"Successfully calculated duration with WAV header parsing: {self.audio_duration} seconds")
                    except Exception as wav_error:
                        print(f"WAV file parsing failed: {str(wav_error)}")
                        raise
                else:
                    print(f"Non-WAV file, pydub calculation failed: {str(pydub_error)}")
                    raise
            
            # Ensure duration is valid
            if self.audio_duration <= 0:
                raise ValueError(f"Invalid audio duration: {self.audio_duration}")
            
            # Update time display directly in main thread
            duration_str = self._format_time(self.audio_duration)
            print(f"Updating time display to: 00:00 / {duration_str}")
            self.time_var.set(f"00:00 / {duration_str}")
        except Exception as e:
            print(f"Error calculating audio duration: {str(e)}")
            self.audio_duration = 0
            # Set default value directly in main thread
            self.time_var.set("00:00 / 00:00")
            print("Set default time display")
    
    def _analyze_file_thread(self):
        """
        Analyze BPM of audio file in a background thread
        """
        try:
            # Update progress bar
            self.root.after(0, lambda: self.progress_var.set(0))
            
            # Read audio file
            if self.audio_file.lower().endswith('.wav'):
                # Read WAV file directly
                with open(self.audio_file, 'rb') as f:
                    # Skip WAV header
                    f.seek(44)
                    
                    # Read audio data
                    audio_data = f.read()
                    
                    # Parse WAV file format information
                    f.seek(22)
                    channels = struct.unpack('<H', f.read(2))[0]
                    
                    f.seek(24)
                    sample_rate = struct.unpack('<I', f.read(4))[0]
                    
                    f.seek(34)
                    bits_per_sample = struct.unpack('<H', f.read(2))[0]
                    
                    f.seek(40)
                    data_size = struct.unpack('<I', f.read(4))[0]
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(20))
                    
                    # Convert audio data based on bit depth
                    samples = []
                    if bits_per_sample == 8:
                        # 8-bit WAV file
                        samples = np.frombuffer(audio_data, dtype=np.uint8)
                        # Normalize to -1.0 to 1.0
                        samples = (samples / 127.5) - 1.0
                    elif bits_per_sample == 16:
                        # 16-bit WAV file
                        samples = np.frombuffer(audio_data, dtype=np.int16)
                        # Normalize to -1.0 to 1.0
                        samples = samples / 32768.0
                    elif bits_per_sample == 24:
                        # 24-bit WAV file
                        samples = []
                        for i in range(0, len(audio_data), 3):
                            # Read 24-bit sample
                            b = audio_data[i:i+3]
                            # Convert to integer
                            sample = (b[2] << 16) | (b[1] << 8) | b[0]
                            # Handle sign bit
                            if sample & 0x800000:
                                sample -= 0x1000000
                            samples.append(sample)
                        samples = np.array(samples)
                        # Normalize to -1.0 to 1.0
                        samples = samples / 8388608.0
                    elif bits_per_sample == 32:
                        # 32-bit WAV file (assuming float)
                        samples = np.frombuffer(audio_data, dtype=np.float32)
                    else:
                        # Unsupported bit depth
                        self.root.after(0, lambda: self.status_var.set(f"Unsupported bit depth: {bits_per_sample} bits"))
                        return
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(40))
                    
                    # If stereo, convert to mono
                    if channels == 2:
                        # Check if data length is divisible by number of channels
                        if len(samples) % 2 != 0:
                            # If not divisible, truncate to divisible length
                            samples = samples[:len(samples) - (len(samples) % 2)]
                        # Reshape array to 2 columns
                        samples = samples.reshape(-1, 2)
                        # Take average as mono
                        samples = np.mean(samples, axis=1)
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(60))
                    
                    # Segment analysis for BPM
                    self.time_bpm_pairs = []
                    segment_size = int(sample_rate * 5)  # 5 seconds per segment
                    total_segments = max(1, len(samples) // segment_size)
                    
                    for i in range(total_segments):
                        start_idx = i * segment_size
                        end_idx = min((i + 1) * segment_size, len(samples))
                        segment = samples[start_idx:end_idx]
                        
                        # Analyze BPM for current segment
                        bpm = self.analyzer.analyze_audio_data(segment, sample_rate)
                        
                        # Calculate time point for current segment
                        current_time_seconds = i * 5  # Each segment is 5 seconds
                        
                        # Save time point and BPM value
                        if bpm is not None:
                            self.time_bpm_pairs.append((current_time_seconds, bpm))
                        
                        # Update progress bar
                        progress = 60 + (30 * (i + 1) / total_segments)
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    
                    # Calculate overall BPM (can be mean or median)
                    if self.time_bpm_pairs:
                        bpm_values = [bpm for _, bpm in self.time_bpm_pairs]
                        overall_bpm = np.mean(bpm_values)
                        self.bpm_result = overall_bpm
                        
                        # Update UI display
                        self.root.after(0, lambda bpm_val=overall_bpm: self.bpm_var.set(f"{bpm_val:.1f}"))
                        self.root.after(0, lambda bpm_val=overall_bpm: self._update_bpm_description(bpm_val))
                        self.root.after(0, lambda: self.status_var.set("Analysis completed"))
                        self.root.after(0, lambda: self.show_timeseries_button.config(state=tk.NORMAL))
                        self.root.after(0, lambda: self._create_bpm_chart())
                    else:
                        # If no segments detected BPM, try analyzing the entire audio file
                        self.root.after(0, lambda: self.status_var.set("Trying full audio analysis..."))
                        full_bpm = self.analyzer.analyze_audio_data(samples, sample_rate)
                        
                        if full_bpm is not None:
                            self.bpm_result = full_bpm
                            self.time_bpm_pairs.append((0, full_bpm))
                            self.root.after(0, lambda bpm_val=full_bpm: self.bpm_var.set(f"{bpm_val:.1f}"))
                            self.root.after(0, lambda bpm_val=full_bpm: self._update_bpm_description(bpm_val))
                            self.root.after(0, lambda: self.status_var.set("Analysis completed (full audio)"))
                            self.root.after(0, lambda: self.show_timeseries_button.config(state=tk.NORMAL))
                            self.root.after(0, lambda: self._create_bpm_chart())
                        else:
                            self.root.after(0, lambda: self.status_var.set("Unable to detect BPM"))
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(100))
            else:
                # Use pydub for other formats
                try:
                    # Read audio file
                    audio = AudioSegment.from_file(self.audio_file)
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(20))
                    
                    # Convert to mono
                    audio = audio.set_channels(1)
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(40))
                    
                    # Convert to numpy array
                    samples = np.array(audio.get_array_of_samples())
                    
                    # Normalize
                    max_val = 2 ** (audio.sample_width * 8 - 1)
                    samples = samples / max_val
                    
                    sample_rate = audio.frame_rate
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(60))
                    
                    # Segment analysis for BPM
                    self.time_bpm_pairs = []
                    segment_size = int(sample_rate * 5)  # 5 seconds per segment
                    total_segments = max(1, len(samples) // segment_size)
                    
                    for i in range(total_segments):
                        start_idx = i * segment_size
                        end_idx = min((i + 1) * segment_size, len(samples))
                        segment = samples[start_idx:end_idx]
                        
                        # Analyze BPM for current segment
                        bpm = self.analyzer.analyze_audio_data(segment, sample_rate)
                        
                        # Calculate time point for current segment
                        current_time_seconds = i * 5  # Each segment is 5 seconds
                        
                        # Save time point and BPM value
                        if bpm is not None:
                            self.time_bpm_pairs.append((current_time_seconds, bpm))
                        
                        # Update progress bar
                        progress = 60 + (30 * (i + 1) / total_segments)
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    
                    # Calculate overall BPM (can be mean or median)
                    if self.time_bpm_pairs:
                        bpm_values = [bpm for _, bpm in self.time_bpm_pairs]
                        overall_bpm = np.mean(bpm_values)
                        self.bpm_result = overall_bpm
                        
                        # Update UI display
                        self.root.after(0, lambda: self.bpm_var.set(f"{overall_bpm:.1f}"))
                        self.root.after(0, lambda bpm_val=overall_bpm: self._update_bpm_description(bpm_val))
                        self.root.after(0, lambda: self.status_var.set("Analysis completed"))
                        self.root.after(0, lambda: self.show_timeseries_button.config(state=tk.NORMAL))
                        self.root.after(0, lambda: self._create_bpm_chart())
                    else:
                        self.root.after(0, lambda: self.status_var.set("Unable to detect BPM"))
                    
                    # Update progress bar
                    self.root.after(0, lambda: self.progress_var.set(100))
                    
                except Exception as e:
                    self.root.after(0, lambda error_msg=str(e): self.status_var.set(f"Error processing audio file: {error_msg}"))
                    return
        except Exception as e:
            self.root.after(0, lambda error_msg=str(e): self.status_var.set(f"Error analyzing audio: {error_msg}"))
            return
            
    def _update_bpm_description(self, bpm):
        """
        Generate music style and mood description based on BPM value
        
        Parameters:
            bpm: Detected BPM value
        """
        if bpm < 60:
            description = "Very slow tempo, possibly meditation, relaxation, or ambient music."
        elif bpm < 80:
            description = "Slow tempo, common in ballads, folk, or slow rock."
        elif bpm < 100:
            description = "Moderately slow, suitable for gentle pop music or soft rock."
        elif bpm < 120:
            description = "Medium tempo, common range for pop music."
        elif bpm < 140:
            description = "Fast tempo, common in electronic music, dance, or rock."
        elif bpm < 160:
            description = "Very fast tempo, such as House, Trance, or other electronic dance music."
        elif bpm < 180:
            description = "Extremely fast tempo, common in Hardstyle or some Techno music."
        else:
            description = "Ultra-fast tempo, possibly hardcore electronic music or certain metal music."
        
        self.bpm_desc_var.set(description)
        
    def _create_bpm_chart(self):
        """
        Create BPM variation chart and display in the reserved chart area
        """
        # Clear placeholder label
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # If no data, show prompt
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            ttk.Label(self.chart_frame, text="No BPM data available", foreground="#777777").pack(pady=20)
            return
        
        # Extract time and BPM data
        times = [time for time, bpm in self.time_bpm_pairs]
        bpms = [bpm for time, bpm in self.time_bpm_pairs]
        
        # Calculate window size
        width = 300
        height = 200
        
        # Create Canvas for drawing
        canvas = tk.Canvas(self.chart_frame, width=width, height=height, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw axes
        padding = 30
        x_min, x_max = min(times), max(times)
        y_min, y_max = min(bpms) * 0.9, max(bpms) * 1.1  # 10% margin
        
        # Draw background grid
        # Vertical lines
        num_vertical_lines = 5
        for i in range(num_vertical_lines + 1):
            x = padding + (width - 2 * padding) * i / num_vertical_lines
            canvas.create_line(x, padding, x, height - padding, fill="#f0f0f0")
            
            # Labels
            if i < num_vertical_lines:
                time_val = x_min + (x_max - x_min) * i / num_vertical_lines
                minutes = int(time_val // 60)
                seconds = int(time_val % 60)
                label = f"{minutes}:{seconds:02d}"
                canvas.create_text(x, height - padding + 15, text=label, font=("Arial", 8))
        
        # Horizontal lines
        num_horizontal_lines = 5
        for i in range(num_horizontal_lines + 1):
            y = height - padding - (height - 2 * padding) * i / num_horizontal_lines
            canvas.create_line(padding, y, width - padding, y, fill="#f0f0f0")
            
            # Labels
            if i < num_horizontal_lines:
                bpm_val = y_min + (y_max - y_min) * i / num_horizontal_lines
                canvas.create_text(padding - 5, y, text=f"{int(bpm_val)}", font=("Arial", 8), anchor="e")
        
        # Draw axes
        canvas.create_line(padding, padding, padding, height - padding, width=2)
        canvas.create_line(padding, height - padding, width - padding, height - padding, width=2)
        
        # Draw BPM curve
        if len(times) > 1:
            # Calculate coordinates for each point
            points = []
            for time, bpm in zip(times, bpms):
                x = padding + (time - x_min) * (width - 2 * padding) / (x_max - x_min)
                y = height - padding - (bpm - y_min) * (height - 2 * padding) / (y_max - y_min)
                points.extend([x, y])
            
            # Draw curve
            canvas.create_line(points, fill="#0078d7", width=2)
            
            # Draw small circles at each point
            for i in range(0, len(points), 2):
                x, y = points[i], points[i+1]
                canvas.create_oval(x-3, y-3, x+3, y+3, fill="#0078d7", outline="#0078d7")
        
        # Add chart title
        canvas.create_text(width/2, 10, text="BPM Variation Over Time", font=("Arial", 10, "bold"))
    
    def toggle_playback(self):
        """
        Toggle play/pause state
        """
        if not self.audio_file:
            messagebox.showinfo("Information", "Please select an audio file first")
            return
        
        # Pre-calculate total audio duration
        if not hasattr(self, 'audio_duration'):
            try:
                if self.audio_file.lower().endswith('.wav'):
                    # For WAV files, calculate duration
                    with open(self.audio_file, 'rb') as f:
                        f.seek(24)
                        sample_rate = struct.unpack('<I', f.read(4))[0]
                        f.seek(40)
                        data_size = struct.unpack('<I', f.read(4))[0]
                        f.seek(34)
                        bits_per_sample = struct.unpack('<H', f.read(2))[0]
                        f.seek(22)
                        channels = struct.unpack('<H', f.read(2))[0]
                        
                        # Calculate duration (seconds)
                        self.audio_duration = data_size / (sample_rate * channels * (bits_per_sample / 8))
                else:
                    # Use pydub to get duration
                    audio = AudioSegment.from_file(self.audio_file)
                    self.audio_duration = len(audio) / 1000  # Convert to seconds
                
                # Update time display (only show total duration part)
                duration_str = self._format_time(self.audio_duration)
                self.time_var.set(f"00:00 / {duration_str}")
            except Exception as e:
                print(f"Error calculating audio duration: {str(e)}")
                self.audio_duration = 0
        
        if not self.is_playing:
            # Start playing or resume
            self.is_playing = True
            self.stop_playback = False
            self.play_button.config(text="Pause")
            self.reset_button.config(state=tk.NORMAL)
            
            # Check if temporary file exists, create if not
            temp_file = "temp_playback.wav"
            if not os.path.exists(temp_file):
                # Convert to WAV format using pydub
                try:
                    audio = AudioSegment.from_file(self.audio_file)
                    audio.export(temp_file, format="wav")
                except Exception as e:
                    messagebox.showerror("Error", f"Error converting audio file: {str(e)}")
                    self._stop_playback()
                    return
            
            # Play audio - key improvement here
            try:
                # Check if audio is already loaded (e.g., resuming from pause)
                if pygame.mixer.music.get_busy() or self.current_position > 0:
                    # Audio is already playing or was paused before, try to continue from paused position
                    if pygame.mixer.music.get_busy():
                        # If currently playing, stop first
                        pygame.mixer.music.stop()
                    
                    # Load audio and play from saved position
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play(start=self.current_position)
                else:
                    # New playback, start from beginning
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                
                # Start updating timer
                self._update_timer()
            except Exception as e:
                messagebox.showerror("Error", f"Error playing audio: {str(e)}")
                self._stop_playback()
        else:
            # Pause playback
            # Save current playback position
            if pygame.mixer.music.get_busy():
                # Get current playback time (milliseconds) and convert to seconds
                self.current_position += pygame.mixer.music.get_pos() / 1000
            
            pygame.mixer.music.pause()
            self.is_playing = False
            self.play_button.config(text="Resume")
    
    def reset_playback(self):
        """
        Reset playback position
        """
        pygame.mixer.music.stop()
        self.current_position = 0
        self.is_playing = False
        self.play_button.config(text="Play")
        self.reset_button.config(state=tk.DISABLED)
        
        # Clear cached audio duration to recalculate when new file is selected
        if hasattr(self, 'audio_duration'):
            delattr(self, 'audio_duration')
        
        # Update time display
        self.time_var.set("00:00 / 00:00")
    
    def _stop_playback(self):
        """
        Stop playback and reset state
        """
        pygame.mixer.music.stop()
        self.is_playing = False
        self.current_position = 0  # Reset playback position
        self.play_button.config(text="Play")
        self.reset_button.config(state=tk.DISABLED)
    
    def _update_timer(self):
        """
        Update playback timer
        """
        if not self.is_playing or self.stop_playback:
            return
        
        if pygame.mixer.music.get_busy():
            # Get current playback time, and add previously saved position
            current_time = self.current_position + (pygame.mixer.music.get_pos() / 1000)  # Convert to seconds
            
            # Use pre-cached audio total duration
            duration = getattr(self, 'audio_duration', 0)
            
            # Update time display
            current_str = self._format_time(current_time)
            duration_str = self._format_time(duration)
            self.time_var.set(f"{current_str} / {duration_str}")
            
            # Continue updating timer
            self.timer_update_job = self.root.after(1000, self._update_timer)
        else:
            # Playback finished
            self._stop_playback()
    
    def _format_time(self, seconds):
        """
        Convert seconds to minutes:seconds format
        
        Parameters:
            seconds: Number of seconds
            
        Returns:
            Formatted time string (MM:SS)
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def on_closing(self):
        """
        Cleanup when window is closed
        """
        # Stop playback
        self.stop_playback = True
        pygame.mixer.music.stop()
        
        # Delete temporary file
        temp_file = "temp_playback.wav"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Quit pygame
        pygame.quit()
        
        # Close window
        self.root.destroy()
    
    def show_bpm_timeseries(self):
        """
        Show BPM variation data over time
        Create a new window with a table displaying BPM values for each time segment
        """
        # Ensure data exists
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            messagebox.showinfo("Information", "No BPM time series data found")
            return
        
        # Create new window
        timeseries_window = tk.Toplevel(self.root)
        timeseries_window.title("BPM Variation Over Time")
        timeseries_window.geometry("600x400")
        timeseries_window.resizable(True, True)
        
        # Add title
        title_label = ttk.Label(
            timeseries_window, 
            text=f"Audio File BPM Variations ({self.get_filename()})",
            font=("Arial", 12, "bold")
        )
        title_label.pack(pady=10)
        
        # Create frame for table
        table_frame = ttk.Frame(timeseries_window)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create vertical scrollbar
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create horizontal scrollbar
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create table
        tree = ttk.Treeview(
            table_frame,
            columns=("time", "bpm"),
            show="headings",
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )
        
        # Configure columns
        tree.column("time", anchor=tk.CENTER, width=150)
        tree.column("bpm", anchor=tk.CENTER, width=100)
        
        # Set column headings
        tree.heading("time", text="Time (min:sec)")
        tree.heading("bpm", text="BPM")
        
        # Fill data
        for time_seconds, bpm in self.time_bpm_pairs:
            # Convert seconds to min:sec format
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            # Add to table
            tree.insert("", tk.END, values=(time_str, round(bpm, 1)))
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        y_scroll.config(command=tree.yview)
        x_scroll.config(command=tree.xview)
        
        # Add export button
        button_frame = ttk.Frame(timeseries_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        export_btn = ttk.Button(
            button_frame,
            text="Export Data",
            command=lambda: self.export_bpm_timeseries()
        )
        export_btn.pack(side=tk.RIGHT)
        
        # Add statistics information
        stats_label = self._create_bpm_stats_label(timeseries_window)
        stats_label.pack(pady=5)
    
    def _create_bpm_stats_label(self, parent):
        """
        Create label displaying BPM statistics
        
        Parameters:
            parent: Parent widget
            
        Returns:
            Configured label widget
        """
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            return ttk.Label(parent, text="No statistics available")
        
        import numpy as np
        
        # Extract all BPM values
        bpm_values = [bpm for _, bpm in self.time_bpm_pairs]
        
        # Calculate statistics
        avg_bpm = np.mean(bpm_values)
        min_bpm = np.min(bpm_values)
        max_bpm = np.max(bpm_values)
        std_bpm = np.std(bpm_values)
        
        # Create statistics text
        stats_text = f"Statistics: Average BPM = {avg_bpm:.1f}, Minimum BPM = {min_bpm:.1f}, Maximum BPM = {max_bpm:.1f}, Standard Deviation = {std_bpm:.1f}"
        
        return ttk.Label(parent, text=stats_text, foreground="#555555")
    
    def export_bpm_timeseries(self):
        """
        Export BPM time series data to CSV format
        """
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            messagebox.showinfo("Information", "No BPM time series data found")
            return
        
        try:
            # Show save file dialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export BPM Data"
            )
            
            if not file_path:
                return  # User cancelled operation
            
            # Write to CSV file
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write header
                csv_writer.writerow(["Time (seconds)", "Time (min:sec)", "BPM"])
                # Write data
                for time_seconds, bpm in self.time_bpm_pairs:
                    # Convert seconds to min:sec format
                    minutes = int(time_seconds // 60)
                    seconds = int(time_seconds % 60)
                    time_str = f"{minutes:02d}:{seconds:02d}"
                    csv_writer.writerow([time_seconds, time_str, round(bpm, 1)])
            
            messagebox.showinfo("Success", f"BPM data successfully exported to:\n{file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data:\n{str(e)}")
    
    def get_filename(self):
        """
        Get the name of the currently selected file
        
        Returns:
            Filename (without path), or "Unknown File" if no file is selected
        """
        if hasattr(self, 'audio_file') and self.audio_file:
            return os.path.basename(self.audio_file)
        return "Unknown File"

def main():
    """
    Program entry point
    """
    root = tk.Tk()
    app = BPMGUIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()