import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pygame\.pkgdata",
    message=r"pkg_resources is deprecated as an API"
)
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
import threading
import time
from pydub import AudioSegment
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from bpm_visuals import plot_deviation_heatmap, plot_bpm_timeseries, plot_distributions
from plot_config import apply_plot_style
apply_plot_style()
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
from scipy import signal
from datetime import datetime
import wave




from bpm_core import BPMAnalyzer

class BPMGUIApp:
    def __init__(self, root):
        """
        Initialize the BPM Analyzer GUI application
        
        Parameters:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Advanced BPM Analyzer")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Set theme and style
        self.style = ttk.Style()
        self._setup_style()
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # BPM Analyzer instance
        self.analyzer = BPMAnalyzer()
        
        # Variables to store application state
        self.audio_file = None
        
        self.analyzing = False
        self.playing = False
        self.playback_position = 0
        self.temp_wav_file = None
        self.playback_thread = None
        self.update_timer_id = None
        self.time_bpm_pairs = []
        self.ref_audio_duration = 0.0
        self.mic_audio_duration = 0.0
        self.audio_duration = 0.0
        self.last_update_time = 0
        
        # Microphone monitoring state
        self.mic_recording = False
        self.mic_stream = None
        self.mic_buffer = []
        self.mic_bpm_history = []
        self.mic_sample_rate = 44100
        self.mic_chunk_size = 1024
        self.mic_bpm = 0
        
        # BPM comparison state
        self.comparison_active = False
        self.reference_bpm = 0
        self.reference_file = None
        self.comparison_results = []
        
        # Create widgets
        self._create_widgets()
        
        # Configure grid weights for responsive layout
        self._configure_layout()
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _setup_style(self):
        """
        Configure ttk styles for a modern look
        """
        # Set theme
        self.style.theme_use("aqua")
        
        # Configure frame style
        self.style.configure(
            "Modern.TFrame",
            background="#f0f0f0",
            borderwidth=1,
            relief="flat"
        )
        
        # Configure button style
        self.style.configure(
            "Modern.TButton",
            font=("Helvetica", 12),
            padding=6
        )
        self.style.map(
            "Modern.TButton",
            background=[],
            foreground=[]
        )
        
        # Configure label style
        self.style.configure(
            "Title.TLabel",
            font=("Arial", 14, "bold"),
            foreground="#333333",
            background="#f0f0f0"
        )
        self.style.configure(
            "Value.TLabel",
            font=("Arial", 18, "bold"),
            foreground="#4a7abc",
            background="#f0f0f0"
        )
        self.style.configure(
            "Description.TLabel",
            font=("Arial", 10),
            foreground="#666666",
            background="#f0f0f0"
        )
        
        # Configure progressbar style
        self.style.configure(
            "Modern.Horizontal.TProgressbar",
            background="#4a7abc",
            troughcolor="#e0e0e0",
            bordercolor="#d0d0d0"
        )
    
    def _create_widgets(self):
        """
        Create all UI widgets
        """
        # Main container
        main_frame = ttk.Frame(self.root, style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid layout
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=0)
        main_frame.grid_rowconfigure(1, weight=0)
        main_frame.grid_rowconfigure(2, weight=1)  # For reference BPM chart
        main_frame.grid_rowconfigure(3, weight=1)  # For microphone BPM chart
        main_frame.grid_rowconfigure(4, weight=0)
        main_frame.grid_rowconfigure(5, weight=0)
        
        # File selection section
        file_section = ttk.Frame(main_frame, style="Modern.TFrame")
        file_section.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        
        ttk.Label(file_section, text="Audio File:", style="Modern.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.file_entry = ttk.Entry(file_section, width=50)
        self.file_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        file_section.grid_columnconfigure(1, weight=1)
        
        browse_btn = ttk.Button(file_section, text="Browse", command=self.browse_file, style="Modern.TButton")
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # BPM interval dropdown (1–10 seconds), default 3
        self.bpm_interval_var = tk.IntVar(value=3)
        interval_box = ttk.Combobox(file_section, textvariable=self.bpm_interval_var,
                                    values=[1,2,3,4,5,6,7,8,9,10], state="readonly", width=4)
        interval_box.grid(row=0, column=3, padx=5, pady=5)
        interval_box.bind("<<ComboboxSelected>>", lambda e: self._on_bpm_interval_change())
        # Initialize mic sampling interval with dropdown value
        self.mic_bpm_sample_interval = float(self.bpm_interval_var.get())
        
        # Calculate BPM button triggers analysis (no auto-analyze on upload)
        calc_btn = ttk.Button(file_section, text="Calculate BPM", command=self.analyze_file, style="Modern.TButton")
        calc_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Analysis results section
        results_section = ttk.Frame(main_frame, style="Modern.TFrame")
        results_section.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        
        # Create two columns for results
        results_section.grid_columnconfigure(0, weight=1)
        results_section.grid_columnconfigure(1, weight=1)
        
        # Removed BPM and Category displays (no longer needed)
        
        # Progress bar (analysis only)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(results_section, variable=self.progress_var, style="Modern.Horizontal.TProgressbar")
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        
        # Reference BPM Chart section
        ref_viz_section = ttk.LabelFrame(main_frame, text="Reference BPM Variation Chart", style="Modern.TLabelframe")
        ref_viz_section.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5, padx=5)
        
        # Create matplotlib figure for reference BPM chart
        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Container to keep controls visible and chart flexible
        ref_container = ttk.Frame(ref_viz_section, style="Modern.TFrame")
        ref_container.pack(fill=tk.BOTH, expand=True)
        ref_container.grid_columnconfigure(0, weight=1)
        ref_container.grid_rowconfigure(0, weight=1)
        ref_container.grid_rowconfigure(1, weight=0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=ref_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Clear initial plot
        self.ax.clear()
        self.ax.set_title("BPM Variation Over Time", pad=10)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("BPM")
        self.ax.grid(True, alpha=0.3)
        try:
            self.fig.subplots_adjust(top=0.92, bottom=0.20)
        except Exception:
            pass
        # Increase bottom margin to avoid clipping x-axis label
        self.fig.tight_layout(rect=[0, 0.12, 1, 0.95])
        self.canvas.draw()
        
        
        # Add macOS-style playback controls below reference chart
        ref_controls = ttk.Frame(ref_container, style="Modern.TFrame")
        ref_controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))
        # Unified Play/Pause button with icon for reference audio
        self.play_button_ref = ttk.Button(ref_controls, command=self.toggle_ref_playback, style="Modern.TButton", width=2)
        self._update_ref_play_button_icon()
        self.play_button_ref.pack(side=tk.LEFT, padx=2)
        # Reset button (icon)
        self.play_button_ref_reset = ttk.Button(ref_controls, text="↺", command=self._ref_reset, style="Modern.TButton", width=2)
        self.play_button_ref_reset.pack(side=tk.LEFT, padx=2)
        
        # Seek bar and time label for reference audio
        self.seek_var_ref = tk.DoubleVar(value=0.0)
        self.seek_scale_ref = ttk.Scale(ref_controls, variable=self.seek_var_ref, from_=0.0, to=0.0, orient=tk.HORIZONTAL, command=lambda v: self._on_seek_ref_live(float(v)))
        self.seek_scale_ref.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.seek_scale_ref.bind("<ButtonPress-1>", lambda e: setattr(self, '_ref_is_dragging', True))
        self.seek_scale_ref.bind("<ButtonRelease-1>", lambda e: self._on_seek_ref(self.seek_var_ref.get()))
        # Reference playback time label (current / duration)
        self.time_label_ref = ttk.Label(ref_controls, text="00:00 / 00:00", style="Modern.TLabel")
        self.time_label_ref.pack(side=tk.LEFT, padx=4)
        # Reference section detailed data button
        self.detail_btn_ref = ttk.Button(ref_controls, text="Show Detailed BPM Data", command=self.show_bpm_timeseries, style="Modern.TButton")
        self.detail_btn_ref.pack(side=tk.LEFT, padx=4)
        
        # Microphone BPM Chart section
        mic_viz_section = ttk.LabelFrame(main_frame, text="Real-time Microphone BPM Chart", style="Modern.TLabelframe")
        mic_viz_section.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5, padx=5)
        
        # Create matplotlib figure for microphone BPM chart
        self.fig_mic = Figure(figsize=(8, 3), dpi=100)
        self.ax_mic = self.fig_mic.add_subplot(111)

        # Container to keep controls visible and chart flexible
        mic_container = ttk.Frame(mic_viz_section, style="Modern.TFrame")
        mic_container.pack(fill=tk.BOTH, expand=True)
        mic_container.grid_columnconfigure(0, weight=1)
        mic_container.grid_rowconfigure(0, weight=1)
        mic_container.grid_rowconfigure(1, weight=0)

        self.canvas_mic = FigureCanvasTkAgg(self.fig_mic, master=mic_container)
        self.canvas_mic.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Clear initial plot for microphone BPM
        self.ax_mic.clear()
        self.ax_mic.set_title("Real-time Microphone BPM", pad=10)
        self.ax_mic.set_xlabel("Time")
        self.ax_mic.set_ylabel("BPM")
        self.ax_mic.grid(True, alpha=0.3)
        try:
            self.fig_mic.subplots_adjust(top=0.86, bottom=0.22)
        except Exception:
            pass
        # Increase bottom margin to avoid clipping x-axis label
        self.fig_mic.tight_layout(rect=[0, 0.12, 1, 0.92])
        self.canvas_mic.draw()
        
        # Add macOS-style playback controls below microphone chart
        mic_controls = ttk.Frame(mic_container, style="Modern.TFrame")
        mic_controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))
        # Use grid to control left-to-right placement and expansion
        mic_controls.grid_columnconfigure(0, weight=0)
        mic_controls.grid_columnconfigure(1, weight=0)
        mic_controls.grid_columnconfigure(2, weight=0)
        mic_controls.grid_columnconfigure(3, weight=1)
        mic_controls.grid_columnconfigure(4, weight=0)
        mic_controls.grid_columnconfigure(5, weight=0)
        # Mic monitor icon button at far left (narrow)
        self.mic_button = ttk.Button(mic_controls, text="🎤", command=self.toggle_mic_monitor, style="Modern.TButton", width=5)
        self.mic_button.grid(row=0, column=0, padx=4)
        # Mic playback buttons (narrow)
        self.play_button_mic = ttk.Button(mic_controls, command=self.toggle_mic_playback, style="Modern.TButton", width=2)
        self._update_mic_play_button_icon()
        self.play_button_mic.grid(row=0, column=1, padx=2)
        # Reset button (icon) for microphone playback (narrow)
        self.play_button_mic_reset = ttk.Button(mic_controls, text="↺", command=self._mic_reset, style="Modern.TButton", width=2)
        self.play_button_mic_reset.grid(row=0, column=2, padx=4)
        self.seek_var_mic = tk.DoubleVar(value=0.0)
        self.seek_scale_mic = ttk.Scale(mic_controls, variable=self.seek_var_mic, from_=0.0, to=0.0, orient=tk.HORIZONTAL, command=lambda v: self._on_seek_mic_live(float(v)))
        self.seek_scale_mic.grid(row=0, column=3, sticky="ew", padx=6)
        self.seek_scale_mic.bind("<ButtonPress-1>", lambda e: setattr(self, '_mic_is_dragging', True))
        self.seek_scale_mic.bind("<ButtonRelease-1>", lambda e: self._on_seek_mic(self.seek_var_mic.get()))
        # Microphone playback time label (current / duration)
        self.mic_time_label = ttk.Label(mic_controls, text="00:00 / 00:00", style="Modern.TLabel")
        self.mic_time_label.grid(row=0, column=4, padx=4)
        # Microphone section detailed data button
        self.detail_btn_mic = ttk.Button(mic_controls, text="Show Detailed BPM Data", command=self.show_mic_bpm_timeseries, style="Modern.TButton")
        self.detail_btn_mic.grid(row=0, column=5, padx=4)
        
        # Control buttons section
        control_section = ttk.Frame(main_frame, style="Modern.TFrame")
        control_section.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        control_section.grid_columnconfigure(0, weight=1)
        control_section.grid_columnconfigure(1, weight=1)
        control_section.grid_columnconfigure(2, weight=1)
        
        # Removed global Play and Reset buttons to consolidate controls under reference/mic sections
        
        # Removed global 'Show Detailed BPM Data' button; added per-section buttons in reference and mic controls
        
        # Microphone monitoring button moved to far left as icon in mic_controls
        
        # BPM comparison button
        compare_btn = ttk.Button(control_section, text="Compare BPM", command=self.compare_bpm, style="Modern.TButton")
        compare_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Removed global playback time label per UI request
        
        # Removed bottom summary area (Microphone BPM and Comparison Result) per UI request
    
    def _update_play_button_icon(self):
        """
        Update the play/pause button label with icon glyphs to keep macOS-style feel
        """
        if not hasattr(self, 'play_button'):
            return
        # Ensure button style is set and text reflects state with icon
        if not self.playing:
            self.play_button.config(text="▶ Play", style="Modern.TButton")
        else:
            self.play_button.config(text="⏸ Pause", style="Modern.TButton")
    
    def _update_ref_play_button_icon(self):
        """
        Update the reference section play/pause button icon based on current context.
        """
        if hasattr(self, 'play_button_ref'):
            if self.playing and getattr(self, 'current_playback_file', None) == getattr(self, 'temp_wav_file', None):
                self.play_button_ref.config(text="⏸", style="Modern.TButton")
            else:
                self.play_button_ref.config(text="▶", style="Modern.TButton")
    
    def _update_mic_play_button_icon(self):
        """
        Update the microphone section play/pause button icon based on current context.
        """
        if hasattr(self, 'play_button_mic'):
            if self.playing and getattr(self, 'current_playback_file', None) == getattr(self, 'temp_mic_wav_file', None):
                self.play_button_mic.config(text="⏸", style="Modern.TButton")
            else:
                self.play_button_mic.config(text="▶", style="Modern.TButton")
    
    def _configure_layout(self):
        """
        Configure grid weights and responsive layout
        """
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
    
    def _on_bpm_interval_change(self):
        """Sync microphone sampling interval with selected BPM interval."""
        try:
            val = float(self.bpm_interval_var.get())
            self.mic_bpm_sample_interval = val
        except Exception:
            pass
    
    def browse_file(self):
        """
        Open file dialog to select audio file
        """
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.flac *.ogg *.aac *.wma"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            # Reset temporary WAV file and playback state when selecting a new file
            if hasattr(self, 'temp_wav_file') and self.temp_wav_file and os.path.exists(self.temp_wav_file):
                try:
                    os.remove(self.temp_wav_file)
                    print(f"Removed old temp WAV file: {self.temp_wav_file}")
                except Exception as e:
                    print(f"Error removing old temp WAV: {e}")
            self.temp_wav_file = None
            self.playing = False
            self.playback_position = 0
            self.last_update_time = 0
            self._update_play_button_icon()
            
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.audio_file = file_path
            
            # Calculate and display audio duration
            self._calculate_and_display_duration()
            
            print("File loaded. Click 'Calculate BPM' to start analysis.")



    
    def _calculate_and_display_duration(self):
        """
        Calculate audio file duration and display it
        """
        try:
            if not self.audio_file:
                return
                
            # Try to get duration using pydub
            audio = AudioSegment.from_file(self.audio_file)
            self.ref_audio_duration = audio.duration_seconds
            
            # Format duration as MM:SS
            minutes = int(self.ref_audio_duration // 60)
            seconds = int(self.ref_audio_duration % 60)
            duration_str = f"{minutes:02d}:{seconds:02d}"
            
            # Update duration label (label removed; safe-guard)
            if hasattr(self, 'duration_label'):
                self.duration_label.config(text=f"Duration: {duration_str}")
            if hasattr(self, 'time_label'):
                self.time_label.config(text=f"00:00 / {duration_str}")
            
            # Initialize reference chart controls
            try:
                if hasattr(self, 'seek_scale_ref'):
                    self.seek_scale_ref.configure(to=self.ref_audio_duration)
                    self.seek_var_ref.set(0.0)
                if hasattr(self, 'time_label_ref'):
                    self.time_label_ref.config(text=f"00:00 / {duration_str}")
                # Configure single canvas-based range slider bounds
                if hasattr(self, 'ref_range_canvas'):
                    self.ref_range_start_var.set(0.0)
                    self.ref_range_end_var.set(self.ref_audio_duration)
                    self.ref_range_start = 0.0
                    self.ref_range_end = self.ref_audio_duration
                    if hasattr(self, 'ref_range_label_start'):
                        self.ref_range_label_start.config(text=self._format_time(self.ref_range_start))
                    if hasattr(self, 'ref_range_label_end'):
                        self.ref_range_label_end.config(text=self._format_time(self.ref_range_end))
                    # Draw slider and markers on load if data exists
                    try:
                        self._redraw_range_slider()
                    except Exception:
                        pass
                    if hasattr(self, 'time_bpm_pairs') and self.time_bpm_pairs:
                        self._create_bpm_chart()
            except Exception as _:
                pass
            
        except Exception as e:
            print(f"Error calculating duration: {e}")
        
    def _on_ref_range_change(self, kind, value):
        """Handle changes to the dual-handle range selector under the reference chart."""
        try:
            val = float(value)
        except Exception:
            return
        start_val = float(self.ref_range_start_var.get())
        end_val = float(self.ref_range_end_var.get())
        if kind == 'start':
            if val > end_val:
                self.ref_range_end_var.set(val)
                end_val = val
            self.ref_range_start = val
            self.ref_range_end = end_val
        else:
            if val < start_val:
                self.ref_range_start_var.set(val)
                start_val = val
            self.ref_range_start = start_val
            self.ref_range_end = val
        # Update labels
        if hasattr(self, 'ref_range_label_start'):
            self.ref_range_label_start.config(text=self._format_time(self.ref_range_start))
        if hasattr(self, 'ref_range_label_end'):
            self.ref_range_label_end.config(text=self._format_time(self.ref_range_end))
        # Redraw chart with markers
        if hasattr(self, 'time_bpm_pairs') and self.time_bpm_pairs:
            self._create_bpm_chart()
    
    def _redraw_range_slider(self):
        """Redraw the single canvas-based range slider with one handle (End)."""
        if not hasattr(self, 'ref_range_canvas'):
            return
        try:
            canvas = self.ref_range_canvas
            w = max(0, int(canvas.winfo_width()))
            h = max(24, int(canvas.winfo_height()))
            margin = 10
            y = h // 2
            canvas.delete("all")
            # Trough
            canvas.create_line(margin, y, w - margin, y, fill="#cfcfcf", width=4, capstyle=tk.ROUND)
            duration = float(getattr(self, 'ref_audio_duration', 0.0)) or 0.0
            usable = max(1, (w - 2 * margin))
            def x_for(v):
                v = max(0.0, min(duration, float(v)))
                return margin + (v / duration * usable) if duration > 0 else margin
            sx = x_for(getattr(self, 'ref_range_start', 0.0))
            ex = x_for(getattr(self, 'ref_range_end', 0.0))
            # Selected range highlight (from fixed Start to movable End)
            canvas.create_line(sx, y, ex, y, fill="#90CAF9", width=6, capstyle=tk.ROUND)
            # Single handle at End
            r = 6
            canvas.create_oval(ex - r, y - r, ex + r, y + r, fill="#2196F3", outline="")
        except Exception:
            pass
    
    def _value_from_canvas_x(self, x):
        """Map a canvas x-coordinate to a time value in seconds."""
        canvas = getattr(self, 'ref_range_canvas', None)
        if not canvas:
            return 0.0
        w = max(1, int(canvas.winfo_width()))
        margin = 10
        duration = float(getattr(self, 'ref_audio_duration', 0.0)) or 0.0
        usable = max(1, (w - 2 * margin))
        pos = max(margin, min(w - margin, int(x)))
        rel = (pos - margin) / usable
        return max(0.0, min(duration, rel * duration))
    
    def _on_range_canvas_press(self, event):
        """Always select the End handle for single-point control."""
        self._active_range_handle = 'end'
    
    def _on_range_canvas_drag(self, event):
        """Drag the End handle and update the range/labels/markers."""
        new_val = self._value_from_canvas_x(event.x)
        self._on_ref_range_change('end', new_val)
        try:
            self._redraw_range_slider()
        except Exception:
            pass
    
    def analyze_file(self):
        """
        Start audio file analysis in a separate thread
        """
        if not self.audio_file:
            messagebox.showwarning("Warning", "Please select an audio file first")
            return
            
        if not os.path.exists(self.audio_file):
            messagebox.showwarning("Warning", "Selected file does not exist")
            return
            
        # Disable analyze button during analysis
        if hasattr(self, 'analyzing') and self.analyzing:
            return
            
        self.analyzing = True
        
        # Clear previous results (labels may be removed)
        if hasattr(self, 'bpm_value_label'):
            self.bpm_value_label.config(text="Analyzing...")
        if hasattr(self, 'bpm_category_label'):
            self.bpm_category_label.config(text="Processing audio file...")
        
        # Create a new thread for analysis to keep UI responsive
        analysis_thread = threading.Thread(target=self._analyze_file_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _analyze_file_thread(self):
        """
        Thread function for audio file analysis
        """
        try:
            # Load audio file
            audio = AudioSegment.from_file(self.audio_file)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate to 44.1kHz
            audio = audio.set_frame_rate(44100)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Normalize to [-1, 1]
            max_val = 2 ** (audio.sample_width * 8 - 1)
            samples = samples.astype(np.float32) / max_val
            
            # Analyze in segments (user-selected seconds, no overlap)
            try:
                segment_duration = float(self.bpm_interval_var.get()) if hasattr(self, 'bpm_interval_var') else 3.0
            except Exception:
                segment_duration = 3.0
            segment_samples = int(segment_duration * audio.frame_rate)
            overlap_samples = 0  # no overlap to enforce cadence for smoother results
            
            self.time_bpm_pairs = []
            
            # Calculate total segments
            total_segments = max(1, int((len(samples) - segment_samples) / (segment_samples - overlap_samples)) + 1)
            
            for i in range(total_segments):
                # Calculate segment start and end indices
                start_idx = i * (segment_samples - overlap_samples)
                end_idx = start_idx + segment_samples
                
                # Ensure we don't go beyond the audio
                if end_idx > len(samples):
                    end_idx = len(samples)
                    start_idx = max(0, end_idx - segment_samples)
                
                # Extract segment
                segment = samples[start_idx:end_idx]
                
                # Calculate segment time in seconds
                segment_time = start_idx / audio.frame_rate
                
                # Analyze segment
                bpm = self.analyzer.analyze_audio_segment(segment, audio.frame_rate)
                
                # Add to results
                self.time_bpm_pairs.append((segment_time, bpm))
                
                # Update progress bar
                progress_percentage = (i + 1) / total_segments * 100
                self.root.after(0, lambda p=progress_percentage: self.progress_var.set(p))
            
            # Calculate overall BPM
            if self.time_bpm_pairs:
                bpm_values = [bpm for _, bpm in self.time_bpm_pairs]
                avg_bpm = np.mean(bpm_values)
                
                # Update UI with results
                self.root.after(0, lambda: self._update_bpm_display(avg_bpm))
                self.root.after(0, self._update_bpm_description)
                self.root.after(0, self._create_bpm_chart)
            
        except Exception as e:
            print(f"Error in analysis thread: {e}")
            self.root.after(0, lambda error=str(e): messagebox.showerror("Error", f"Analysis error:\n{error}"))
        finally:
            # Reset analyzing flag
            self.analyzing = False
            
            # Set progress to 100% when done
            self.root.after(0, lambda: self.progress_var.set(100))
    
    def _update_bpm_display(self, bpm):
        """
        Update BPM value display
        """
        # Store reference BPM for comparison
        self.reference_bpm = bpm
        
        # Update label if present
        if hasattr(self, 'bpm_value_label'):
            self.bpm_value_label.config(text=f"{bpm:.1f}")
    
    def _update_bpm_description(self):
        """
        Update BPM category description
        """
        if hasattr(self, 'reference_bpm') and self.reference_bpm > 0:
            description = self.analyzer._bpm_to_category(self.reference_bpm)
            if hasattr(self, 'bpm_category_label'):
                self.bpm_category_label.config(text=description)
    
    def _create_bpm_chart(self):
        """
        Create BPM variation chart using matplotlib
        """
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        # Extract data
        times, bpms = zip(*self.time_bpm_pairs)
        times_seconds = list(times)  # Use seconds directly
        
        # Apply smoothing to BPM values for better visualization
        smoothed_bpms = self._smooth_bpm_values(bpms)
        
        # Plot smoothed BPM curve
        self.ax.plot(times_seconds, smoothed_bpms, 'b-', linewidth=2, alpha=0.7, label='BPM')
        
        # Plot original BPM points
        self.ax.scatter(times_seconds, bpms, color='r', s=30, alpha=0.5, label='Raw BPM')
        
        # Add average BPM line
        avg_bpm = np.mean(bpms)
        self.ax.axhline(y=avg_bpm, color='g', linestyle='--', alpha=0.7, label=f'Avg BPM: {avg_bpm:.1f}')
        
        # Configure plot
        self.ax.set_title("BPM Variation Over Time", pad=10)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("BPM")
        
        # Set appropriate y-axis limits
        min_bpm = max(40, np.min(bpms) - 10)
        max_bpm = min(220, np.max(bpms) + 10)
        self.ax.set_ylim(min_bpm, max_bpm)
        
        # Set x-axis limits to include the full audio duration or data extent
        try:
            duration = float(getattr(self, 'ref_audio_duration', 0.0)) or 0.0
        except Exception:
            duration = 0.0
        max_time = max(times_seconds) if times_seconds else 0.0
        right_limit = duration if duration > 0 else (max_time + 2)
        if right_limit < 5:
            right_limit = 5
        self.ax.set_xlim(0, right_limit)
        
        # Add grid and legend
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Ensure title is not clipped
        try:
            self.fig.subplots_adjust(top=0.92)
        except Exception:
            pass
        # Redraw canvas with safe margins for titles
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.draw()
    
    def _smooth_bpm_values(self, bpm_values, window_size=3):
        """
        Apply smoothing to BPM values for better visualization
        """
        if len(bpm_values) < window_size:
            return bpm_values
            
        # Use Gaussian filter for smoothing
        smoothed = signal.wiener(bpm_values, window_size)
        return smoothed
    
    def toggle_playback(self):
        """
        Toggle audio playback (play/pause)
        """
        print("Toggle playback called")
        if not self.audio_file:
            messagebox.showwarning("Warning", "Please select an audio file first")
            print("No audio file selected")
            return
            
        print(f"Audio file: {self.audio_file}")
        
        # Check if audio has been analyzed yet
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            print("Audio not analyzed yet, performing analysis...")
            messagebox.showinfo("Info", "Performing BPM analysis first...")
            # Perform analysis in a blocking way to ensure we have the data
            if not self.analyzing:
                # First set up the analyzer variables
                self.analyzing = True
                if hasattr(self, 'bpm_value_label'):
                    self.bpm_value_label.config(text="Analyzing...")
                if hasattr(self, 'bpm_category_label'):
                    self.bpm_category_label.config(text="Processing audio file...")
                
                try:
                    # Process the audio file directly
                    audio = AudioSegment.from_file(self.audio_file)
                    
                    # Convert to numpy array
                    samples = np.array(audio.get_array_of_samples())
                    
                    # Normalize to [-1, 1]
                    max_val = 2 ** (audio.sample_width * 8 - 1)
                    samples = samples.astype(np.float32) / max_val
                    
                    # Analyze in segments (3 seconds each, no overlap)
                    segment_duration = 3.0  # seconds
                    segment_samples = int(segment_duration * audio.frame_rate)
                    overlap_samples = 0  # no overlap to enforce 3s cadence
                    
                    self.time_bpm_pairs = []
                    
                    # Calculate total segments
                    total_segments = max(1, int((len(samples) - segment_samples) / (segment_samples - overlap_samples)) + 1)
                    
                    for i in range(total_segments):
                        # Calculate segment start and end indices
                        start_idx = i * (segment_samples - overlap_samples)
                        end_idx = start_idx + segment_samples
                        
                        # Ensure we don't go beyond the audio
                        if end_idx > len(samples):
                            end_idx = len(samples)
                            start_idx = max(0, end_idx - segment_samples)
                        
                        # Extract segment
                        segment = samples[start_idx:end_idx]
                        
                        # Calculate segment time in seconds
                        segment_time = start_idx / audio.frame_rate
                        
                        # Analyze segment
                        bpm = self.analyzer.analyze_audio_segment(segment, audio.frame_rate)
                        
                        # Add to results
                        self.time_bpm_pairs.append((segment_time, bpm))
                        
                        # Update progress bar
                        progress_percentage = (i + 1) / total_segments * 100
                        self.progress_var.set(progress_percentage)
                        self.root.update_idletasks()  # Force UI update
                    
                    # Calculate overall BPM
                    if self.time_bpm_pairs:
                        bpm_values = [bpm for _, bpm in self.time_bpm_pairs]
                        avg_bpm = np.mean(bpm_values)
                        
                        # Update UI with results
                        self._update_bpm_display(avg_bpm)
                        self._update_bpm_description()
                        self._create_bpm_chart()
                    
                except Exception as e:
                    print(f"Error in analysis: {e}")
                    messagebox.showerror("Error", f"Analysis error:\n{str(e)}")
                    self.analyzing = False
                    return
                finally:
                    # Reset analyzing flag
                    self.analyzing = False
                    # Set progress to 100%
                    self.progress_var.set(100)
        
        if not hasattr(self, 'temp_wav_file') or self.temp_wav_file is None or not os.path.exists(self.temp_wav_file):
            # Convert audio to WAV for playback if needed
            try:
                print("Converting to WAV for playback...")
                self._convert_to_wav_for_playback()
                print(f"WAV conversion complete: {self.temp_wav_file}")
            except Exception as e:
                print(f"Error in conversion: {e}")
                messagebox.showerror("Error", f"Error preparing audio for playback:\n{str(e)}")
                return
        else:
            print(f"Using existing temp WAV: {self.temp_wav_file}")
        
        print(f"Current playing state: {self.playing}")
        if not self.playing:
            # Start or resume playback
            print("Starting playback...")
            self._start_playback()
        else:
            # Pause playback
            print("Pausing playback...")
            self._pause_playback()
    
    def toggle_ref_playback(self):
        """
        Wrapper to play/pause reference audio from the chart controls
        """
        if not self.audio_file:
            messagebox.showwarning("Warning", "Please select an audio file first")
            return
        # Ensure temp WAV exists
        if not hasattr(self, 'temp_wav_file') or self.temp_wav_file is None or not os.path.exists(self.temp_wav_file):
            try:
                self._convert_to_wav_for_playback()
            except Exception as e:
                messagebox.showerror("Error", f"Error preparing audio for playback:\n{str(e)}")
                return
        # Set current playback file
        self.current_playback_file = self.temp_wav_file
        # Reset any sliced load path and offset
        self.current_playback_load_path = None
        self.playback_offset = 0.0
        # Sync playback_position from reference seek bar before play
        try:
            pos = float(self.seek_var_ref.get()) if hasattr(self, 'seek_var_ref') else getattr(self, 'playback_position', 0.0)
            duration = getattr(self, 'ref_audio_duration', 0.0)
            self.playback_position = max(0.0, min(pos, duration))
        except Exception:
            pass
        # Update seek range for reference
        if hasattr(self, 'ref_audio_duration') and hasattr(self, 'seek_scale_ref'):
            try:
                self.seek_scale_ref.configure(to=self.ref_audio_duration)
            except Exception:
                pass
        # Delegate to existing toggle
        self.toggle_playback()
    
    def _ref_play(self):
        """Play or resume reference audio from the chart controls"""
        if not getattr(self, 'audio_file', None):
            messagebox.showwarning("Warning", "Please select an audio file first")
            return
        # Ensure temp WAV exists
        if not hasattr(self, 'temp_wav_file') or self.temp_wav_file is None or not os.path.exists(self.temp_wav_file):
            try:
                self._convert_to_wav_for_playback()
            except Exception as e:
                messagebox.showerror("Error", f"Error preparing audio for playback:\n{str(e)}")
                return
        # Set playback source to reference file
        self.current_playback_file = self.temp_wav_file
        # Sync playback_position from reference seek bar before play
        try:
            pos = float(self.seek_var_ref.get()) if hasattr(self, 'seek_var_ref') else getattr(self, 'playback_position', 0.0)
            duration = getattr(self, 'ref_audio_duration', 0.0)
            self.playback_position = max(0.0, min(pos, duration))
        except Exception:
            pass
        # Ensure duration and seek range
        try:
            self.ref_audio_duration = AudioSegment.from_file(self.audio_file).duration_seconds
            if hasattr(self, 'seek_scale_ref'):
                self.seek_scale_ref.configure(to=self.ref_audio_duration)
        except Exception:
            pass
        # Start or resume
        self._start_playback()
    
    def _ref_pause(self):
        """Pause reference audio playback"""
        try:
            self._pause_playback()
        except Exception:
            pass
    
    def _ref_reset(self):
        """Reset reference audio playback position to the start"""
        if not getattr(self, 'audio_file', None):
            messagebox.showwarning("Warning", "Please select an audio file first")
            return
        # Set playback source to reference and stop
        self.current_playback_file = getattr(self, 'temp_wav_file', None)
        try:
            self._stop_playback()
        except Exception:
            pass
        self.playback_position = 0.0
        # Refresh duration and UI for reference controls
        try:
            self.ref_audio_duration = AudioSegment.from_file(self.audio_file).duration_seconds
            if hasattr(self, 'seek_var_ref'):
                self.seek_var_ref.set(0.0)
            if hasattr(self, 'seek_scale_ref'):
                self.seek_scale_ref.configure(to=self.ref_audio_duration)
            if hasattr(self, 'time_label_ref'):
                duration_str = self._format_time(self.ref_audio_duration)
                self.time_label_ref.config(text=f"00:00 / {duration_str}")
            # Also refresh global time label if present
            if hasattr(self, 'time_label'):
                self.time_label.config(text=f"00:00 / {self._format_time(self.ref_audio_duration)}")
        except Exception:
            pass
    
    def _on_seek_ref_live(self, value):
        """Live update during dragging on reference seek bar (no reload/play)."""
        try:
            # Clamp and store position
            self.playback_position = max(0.0, min(float(value), getattr(self, 'ref_audio_duration', 0.0)))
            # Keep current playback context to reference
            if hasattr(self, 'temp_wav_file'):
                self.current_playback_file = self.temp_wav_file
            # Update UI time label
            current_str = self._format_time(self.playback_position)
            duration_str = self._format_time(getattr(self, 'ref_audio_duration', 0.0))
            if hasattr(self, 'time_label_ref'):
                self.time_label_ref.config(text=f"{current_str} / {duration_str}")
            # Update chart vertical line immediately
            if hasattr(self, 'time_bpm_pairs') and self.time_bpm_pairs:
                self._highlight_current_bpm_position(self.playback_position)
        except Exception as e:
            print(f"Error in live seek (ref): {e}")

    def _on_seek_ref(self, value):
        """Seek reference playback to a specific position (seconds)"""
        try:
            self.playback_position = max(0.0, min(float(value), getattr(self, 'ref_audio_duration', 0.0)))
            # Set current playback file to ref when seeking from ref controls
            if hasattr(self, 'temp_wav_file'):
                self.current_playback_file = self.temp_wav_file
            if self.playing:
                # Prefer direct seek without reloading to avoid interruptions
                try:
                    pygame.mixer.music.set_pos(self.playback_position)
                    # Reset timer baseline to the new position
                    self.last_update_time = time.time() - self.playback_position
                except Exception as _:
                    # Fallback to restart playback at new position
                    self._start_playback()
            else:
                current_str = self._format_time(self.playback_position)
                duration_str = self._format_time(getattr(self, 'ref_audio_duration', 0.0))
                if hasattr(self, 'time_label_ref'):
                    self.time_label_ref.config(text=f"{current_str} / {duration_str}")
        except Exception as e:
            print(f"Error seeking reference: {e}")
        finally:
            try:
                setattr(self, '_ref_is_dragging', False)
            except Exception:
                pass
    
    def toggle_mic_playback(self):
        """
        Toggle play/pause for recorded microphone audio under mic chart.
        If something else is currently playing, switch to the mic recording instead of pausing.
        """
        # Validate mic data presence (buffer or recorded file)
        if not hasattr(self, 'temp_mic_wav_file') or self.temp_mic_wav_file is None:
            if not hasattr(self, 'mic_buffer') or not self.mic_buffer:
                messagebox.showwarning("Warning", "No microphone recording available yet")
                return
        
        # Ensure mic WAV exists; otherwise convert buffer
        if not os.path.exists(getattr(self, 'temp_mic_wav_file', '') or ''):
            try:
                self._convert_mic_to_wav_for_playback()
            except Exception as e:
                messagebox.showerror("Error", f"Error preparing mic audio for playback:\n{str(e)}")
                return
        
        # Set playback context to mic
        self.current_playback_file = self.temp_mic_wav_file
        
        # Determine accurate duration from the WAV file when available
        try:
            with wave.open(self.temp_mic_wav_file, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or float(self.mic_sample_rate)
                self.mic_audio_duration = frames / float(rate)
        except Exception:
            # Fallback to buffer duration
            self.mic_audio_duration = (len(self.mic_buffer) / float(self.mic_sample_rate)) if hasattr(self, 'mic_buffer') and self.mic_buffer else 0.0
        # Update seek range for mic
        if hasattr(self, 'seek_scale_mic'):
            try:
                self.seek_scale_mic.configure(to=self.mic_audio_duration)
            except Exception:
                pass
        
        # Start or pause: if playing a different source, switch to mic recording
        if not self.playing or (getattr(self, 'current_playback_file', None) != getattr(self, 'temp_mic_wav_file', None)):
            # Preserve current position or use mic seek value when switching context
            try:
                pos = float(self.seek_var_mic.get()) if hasattr(self, 'seek_var_mic') else getattr(self, 'playback_position', 0.0)
                duration = getattr(self, 'mic_audio_duration', 0.0)
                self.playback_position = max(0.0, min(pos, duration))
            except Exception:
                pass
            self._start_playback()
        else:
            self._pause_playback()
    
    def _on_seek_mic_live(self, value):
        """Live update during dragging on microphone seek bar (no reload/play)."""
        try:
            duration = getattr(self, 'mic_audio_duration', 0.0)
            # Clamp and store position using current audio duration
            self.playback_position = max(0.0, min(float(value), duration))
            # Keep current playback context to mic
            if hasattr(self, 'temp_mic_wav_file'):
                self.current_playback_file = self.temp_mic_wav_file
            # Update UI time label
            current_str = self._format_time(self.playback_position)
            duration_str = self._format_time(duration)
            if hasattr(self, 'mic_time_label'):
                self.mic_time_label.config(text=f"{current_str} / {duration_str}")
            # Update mic chart vertical line immediately if data exists
            if hasattr(self, 'mic_time_bpm_pairs') and self.mic_time_bpm_pairs:
                try:
                    self._highlight_current_mic_bpm_position(self.playback_position)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error in live seek (mic): {e}")

    def _on_seek_mic(self, value):
        """Seek microphone playback to a specific position (seconds)"""
        try:
            duration = getattr(self, 'mic_audio_duration', 0.0)
            self.playback_position = max(0.0, min(float(value), duration))
            # Ensure current file is mic
            if hasattr(self, 'temp_mic_wav_file'):
                self.current_playback_file = self.temp_mic_wav_file
            if self.playing:
                # Prefer direct seek without reloading to avoid interruptions
                try:
                    pygame.mixer.music.set_pos(self.playback_position)
                    # Reset timer baseline to the new position
                    self.last_update_time = time.time() - self.playback_position
                except Exception:
                    # Fallback to restart playback at new position
                    self._start_playback()
            else:
                current_str = self._format_time(self.playback_position)
                duration_str = self._format_time(duration)
                if hasattr(self, 'mic_time_label'):
                    self.mic_time_label.config(text=f"{current_str} / {duration_str}")
        except Exception as e:
            print(f"Error seeking microphone: {e}")
        finally:
            try:
                setattr(self, '_mic_is_dragging', False)
            except Exception:
                pass
    
    def _mic_reset(self):
        """Reset microphone playback position to the start and update UI."""
        try:
            # Allow mic reset even when no recording is available; do not block
            # Duration and UI will be updated best-effort below.

            # Flip playback context to mic if we have a temp wav prepared
            self.current_playback_file = getattr(self, 'temp_mic_wav_file', None)

            # Stop any ongoing playback
            try:
                self._stop_playback()
            except Exception:
                pass

            # Reset core playback state
            self.playback_position = 0.0

            # Compute mic duration for label
            mic_duration = 0.0
            try:
                if hasattr(self, 'mic_sample_rate') and self.mic_sample_rate:
                    mic_duration = float(len(self.mic_buffer)) / float(self.mic_sample_rate)
            except Exception:
                mic_duration = 0.0

            # Update mic seek and time label
            try:
                if hasattr(self, 'seek_var_mic'):
                    self.seek_var_mic.set(0.0)
                if hasattr(self, 'mic_time_label'):
                    self.mic_time_label.config(text=f"00:00 / {self._format_time(mic_duration)}")
            except Exception:
                pass

            # Keep icons consistent
            try:
                self._update_play_button_icon()
                self._update_ref_play_button_icon()
                self._update_mic_play_button_icon()
            except Exception:
                pass
        except Exception as e:
            print(f"Mic reset failed: {e}")
 
    def _convert_mic_to_wav_for_playback(self):
        """Convert mic buffer to a temporary WAV file for playback"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.temp_mic_wav_file = f"temp_mic_playback_{timestamp}.wav"
            samples = np.array(self.mic_buffer, dtype=np.float32)
            samples = np.clip(samples, -1.0, 1.0)
            samples_int16 = (samples * 32767).astype(np.int16)
            audio = AudioSegment(
                data=samples_int16.tobytes(),
                sample_width=2,
                frame_rate=self.mic_sample_rate,
                channels=1
            )
            audio.export(self.temp_mic_wav_file, format="wav")
        except Exception as e:
            if hasattr(self, 'temp_mic_wav_file') and self.temp_mic_wav_file and os.path.exists(self.temp_mic_wav_file):
                try:
                    os.remove(self.temp_mic_wav_file)
                except:
                    pass
            self.temp_mic_wav_file = None
            raise e
    
    def _convert_to_wav_for_playback(self):
        """
        Convert audio file to temporary WAV file for playback
        """
        try:
            # Generate unique temporary filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.temp_wav_file = f"temp_playback_{timestamp}.wav"
            
            # Convert to WAV using pydub
            audio = AudioSegment.from_file(self.audio_file)
            audio.export(self.temp_wav_file, format="wav")
            
        except Exception as e:
            # Clean up on error
            if hasattr(self, 'temp_wav_file') and os.path.exists(self.temp_wav_file):
                try:
                    os.remove(self.temp_wav_file)
                except:
                    pass
            self.temp_wav_file = None
            raise e
    
    def _start_playback(self):
        """
        Start audio playback
        """
        try:
            print(f"Starting playback with file: {self.current_playback_file or self.temp_wav_file}")
            print(f"Playback position: {self.playback_position} seconds")
            
            pygame.mixer.music.stop()
            print("Stopped any current playback to reset state")
            
            pygame.mixer.music.load(self.current_playback_file if hasattr(self, 'current_playback_file') and self.current_playback_file else self.temp_wav_file)
            print("Music loaded")
            
            pygame.mixer.music.play()
            print("Playback started")
            
            if self.playback_position > 0:
                try:
                    print(f"Trying to set position to {self.playback_position} seconds")
                    pygame.mixer.music.set_pos(self.playback_position)
                    print("Position set successfully")
                except pygame.error as e:
                    print(f"Warning: Failed to set position: {e}")
                    print("Continuing from beginning instead")
            
            if pygame.mixer.music.get_busy():
                print("Playback verification successful: music is playing")
                print("Updating playback state...")
                self.playing = True
                self.last_update_time = time.time() - self.playback_position
                
                print("Starting update timer...")
                self._update_timer()
                
                print("Updating play button icon...")
                self._update_play_button_icon()
                self._update_ref_play_button_icon()
                self._update_mic_play_button_icon()
            else:
                print("Error: Music is not playing after play()")
                self.playing = False
                self._update_play_button_icon()
                self._update_ref_play_button_icon()
                self._update_mic_play_button_icon()
            
        except Exception as e:
            print(f"Error starting playback: {e}")
            print(f"Exception type: {type(e).__name__}")
            self._stop_playback()
            messagebox.showerror("Error", f"Playback error:\n{str(e)}")
    
    def _pause_playback(self):
        """
        Pause audio playback
        """
        try:
            # Get current position
            self.playback_position = pygame.mixer.music.get_pos() / 1000.0  # Convert to seconds
            
            # Pause playback
            pygame.mixer.music.pause()
            
            # Update state
            self.playing = False
            
            # Stop timer
            if self.update_timer_id:
                self.root.after_cancel(self.update_timer_id)
                self.update_timer_id = None
            
            # Update play button icon
            self._update_play_button_icon()
            self._update_ref_play_button_icon()
            self._update_mic_play_button_icon()
            
        except Exception as e:
            print(f"Error pausing playback: {e}")
    
    def reset_playback(self):
        """
        Reset audio playback to beginning
        """
        # Stop playback if running
        self._stop_playback()
        
        # Reset position
        self.playback_position = 0
        
        # Update time labels separately for reference and mic
        minutes = int(getattr(self, 'ref_audio_duration', 0.0) // 60) if hasattr(self, 'ref_audio_duration') else 0
        seconds = int(getattr(self, 'ref_audio_duration', 0.0) % 60) if hasattr(self, 'ref_audio_duration') else 0
        duration_str_ref = f"{minutes:02d}:{seconds:02d}"
        minutes = int(getattr(self, 'mic_audio_duration', 0.0) // 60) if hasattr(self, 'mic_audio_duration') else 0
        seconds = int(getattr(self, 'mic_audio_duration', 0.0) % 60) if hasattr(self, 'mic_audio_duration') else 0
        duration_str_mic = f"{minutes:02d}:{seconds:02d}"
        if hasattr(self, 'time_label'):
            try:
                if hasattr(self, 'current_playback_file') and hasattr(self, 'temp_mic_wav_file') and self.current_playback_file == self.temp_mic_wav_file:
                    self.time_label.config(text=f"00:00 / {duration_str_mic}")
                else:
                    self.time_label.config(text=f"00:00 / {duration_str_ref}")
            except Exception:
                self.time_label.config(text=f"00:00 / {duration_str_ref}")
        # Reset chart-specific labels and seek bars
        try:
            if hasattr(self, 'seek_scale_ref'):
                self.seek_var_ref.set(0.0)
            if hasattr(self, 'time_label_ref'):
                self.time_label_ref.config(text=f"00:00 / {duration_str_ref}")
            if hasattr(self, 'seek_scale_mic'):
                self.seek_var_mic.set(0.0)
            if hasattr(self, 'mic_time_label'):
                self.mic_time_label.config(text=f"00:00 / {duration_str_mic}")
        except Exception as _:
            pass
    
    def _stop_playback(self):
        """
        Stop audio playback completely
        """
        try:
            # Stop playback
            pygame.mixer.music.stop()
            
            # Update state
            self.playing = False
            
            # Stop timer
            if self.update_timer_id:
                self.root.after_cancel(self.update_timer_id)
                self.update_timer_id = None
            
            # Update play button icon
            self._update_play_button_icon()
            
        except Exception as e:
            print(f"Error stopping playback: {e}")
    
    def _update_timer(self):
        """
        Update playback timer and progress bar
        """
        if not self.playing:
            return
            
        try:
            # Calculate current position
            current_time = time.time() - self.last_update_time
            # Avoid overwriting user-controlled position while dragging
            if not getattr(self, '_ref_is_dragging', False) and not getattr(self, '_mic_is_dragging', False):
                self.playback_position = current_time
            
            # Format time strings
            current_str = self._format_time(current_time)
            # Choose duration based on current playback context
            if hasattr(self, 'current_playback_file') and hasattr(self, 'temp_wav_file') and self.current_playback_file == self.temp_wav_file:
                duration_str = self._format_time(getattr(self, 'ref_audio_duration', 0.0))
            elif hasattr(self, 'current_playback_file') and hasattr(self, 'temp_mic_wav_file') and self.current_playback_file == self.temp_mic_wav_file:
                duration_str = self._format_time(getattr(self, 'mic_audio_duration', 0.0))
            else:
                duration_str = self._format_time(getattr(self, 'ref_audio_duration', 0.0))
            
            # Update time label
            if hasattr(self, 'time_label'):
                self.time_label.config(text=f"{current_str} / {duration_str}")
            
            # Do not update progress bar during playback per UI request
            # Progress bar will only reflect analysis progress elsewhere
            
            # Update chart-specific time labels and seek bars
            try:
                if hasattr(self, 'current_playback_file') and hasattr(self, 'temp_wav_file') and self.current_playback_file == self.temp_wav_file:
                    if not getattr(self, '_ref_is_dragging', False):
                        if hasattr(self, 'seek_scale_ref'):
                            self.seek_scale_ref.configure(to=getattr(self, 'ref_audio_duration', 0.0))
                            self.seek_var_ref.set(current_time)
                        if hasattr(self, 'time_label_ref'):
                            self.time_label_ref.config(text=f"{current_str} / {duration_str}")
                elif hasattr(self, 'current_playback_file') and hasattr(self, 'temp_mic_wav_file') and self.current_playback_file == self.temp_mic_wav_file:
                    if not getattr(self, '_mic_is_dragging', False):
                        if hasattr(self, 'seek_scale_mic'):
                            self.seek_scale_mic.configure(to=getattr(self, 'mic_audio_duration', 0.0))
                            self.seek_var_mic.set(current_time)
                        if hasattr(self, 'mic_time_label'):
                            self.mic_time_label.config(text=f"{current_str} / {duration_str}")
            except Exception as _:
                pass
            
            # Update BPM chart progress lines according to current playback source
            try:
                if hasattr(self, 'current_playback_file') and hasattr(self, 'temp_wav_file') and self.current_playback_file == self.temp_wav_file:
                    if hasattr(self, 'time_bpm_pairs') and self.time_bpm_pairs:
                        if not getattr(self, '_ref_is_dragging', False):
                            self._highlight_current_bpm_position(current_time)
                elif hasattr(self, 'current_playback_file') and hasattr(self, 'temp_mic_wav_file') and self.current_playback_file == self.temp_mic_wav_file:
                        if hasattr(self, 'mic_time_bpm_pairs') and self.mic_time_bpm_pairs:
                            if not getattr(self, '_mic_is_dragging', False):
                                self._highlight_current_mic_bpm_position(current_time)
            except Exception:
                pass
            
            # Schedule next update
            self.update_timer_id = self.root.after(100, self._update_timer)
            
            # Check if playback has ended
            if not pygame.mixer.music.get_busy():
                self._stop_playback()
                
        except Exception as e:
            print(f"Error updating timer: {e}")
    
    def _highlight_current_bpm_position(self, current_time):
        """
        Highlight the current playback position on the BPM chart
        and update the current BPM display
        """
        # Find the current BPM segment
        current_bpm = None
        for time_seconds, bpm in self.time_bpm_pairs:
            if time_seconds > current_time:
                break
            current_bpm = bpm
        
        # Update current BPM display if found
        if current_bpm is not None:
            # Update current BPM label only if original label exists
            if hasattr(self, 'bpm_value_label'):
                if not hasattr(self, 'current_bpm_label'):
                    self.current_bpm_label = ttk.Label(self.bpm_value_label.master, text="", style="Description.TLabel")
                    self.current_bpm_label.pack(pady=2)
                self.current_bpm_label.config(text=f"Current: {current_bpm:.1f}")
            
            # Find the index in our time_bpm_pairs for the current segment
            times, bpms = zip(*self.time_bpm_pairs)
            current_idx = 0
            for i, t in enumerate(times):
                if t > current_time:
                    break
                current_idx = i
            
            # Update the chart with a vertical line
            # First, clear any existing vertical line
            for line in getattr(self, '_vline', []):
                line.remove()
            
            # Add new vertical line at current time (seconds)
            self._vline = [self.ax.axvline(x=current_time, color='red', linestyle=':', alpha=0.8)]
            
            # Add a text label showing current BPM near the vertical line
            y_min, y_max = self.ax.get_ylim()
            text_y_pos = y_min + (y_max - y_min) * 0.9
            self._vline.append(self.ax.text(current_time + 0.01, text_y_pos, 
                               f"{current_bpm:.1f} BPM", color='red', alpha=0.8))
            
            # Redraw the canvas
            self.canvas.draw()
    
    def _highlight_current_mic_bpm_position(self, current_time):
        try:
            if not hasattr(self, 'mic_time_bpm_pairs') or not self.mic_time_bpm_pairs:
                return
            if not hasattr(self, 'ax_mic') or not hasattr(self, 'canvas_mic'):
                return
            times, bpms = zip(*self.mic_time_bpm_pairs)
            closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - current_time))
            # Remove previous mic vertical line(s)
            if hasattr(self, '_vline_mic') and self._vline_mic:
                for line in self._vline_mic:
                    try:
                        line.remove()
                    except Exception:
                        pass
                self._vline_mic = []
            self._vline_mic = [self.ax_mic.axvline(x=current_time, color='red', linestyle=':', alpha=0.8)]
            # Add a text label showing current mic BPM near the vertical line
            y_min, y_max = self.ax_mic.get_ylim()
            text_y_pos = y_min + (y_max - y_min) * 0.9
            current_bpm = bpms[closest_idx]
            self._vline_mic.append(self.ax_mic.text(current_time + 0.01, text_y_pos, f"{current_bpm:.1f} BPM", color='red', alpha=0.8))
            self.canvas_mic.draw()
        except Exception:
            pass

    def _format_time(self, seconds):
        """
        Format time in seconds to MM:SS format
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def toggle_mic_monitor(self):
        """
        Toggle microphone monitoring for real-time BPM detection
        """
        if not self.mic_recording:
            self._start_mic_monitoring()
        else:
            self._stop_mic_monitoring()
    
    def _start_mic_monitoring(self):
        """
        Start microphone monitoring thread and begin real-time WAV recording.
        """
        try:
            # Update button text (mic-off icon)
            self.mic_button.config(text="🚫🎤")
            
            # Update state
            self.mic_recording = True
            self.mic_buffer = []
            self.mic_bpm_history = []
            
            # Initialize time tracking and BPM data for immediate chart display
            self.mic_start_time = time.time()
            self.mic_time_bpm_pairs = [(0, 0)]  # Add initial data point for immediate chart display
            # Set mic BPM sampling interval from the user-selected dropdown (seconds)
            try:
                self.mic_bpm_sample_interval = float(self.bpm_interval_var.get())
            except Exception:
                # Fallback to previous value or default 3.0 if dropdown not available
                self.mic_bpm_sample_interval = float(getattr(self, 'mic_bpm_sample_interval', 3.0))
            self.mic_last_bpm_sample_ts = self.mic_start_time
            print(f"Mic sampling interval set to {self.mic_bpm_sample_interval} seconds")
            
            # Prepare real-time WAV recording
            try:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                self.temp_mic_wav_file = f"mic_recording_{timestamp}.wav"
                self.mic_writer_lock = threading.Lock()
                self.mic_wave_writer = wave.open(self.temp_mic_wav_file, 'wb')
                self.mic_wave_writer.setnchannels(1)
                self.mic_wave_writer.setsampwidth(2)  # 16-bit PCM
                self.mic_wave_writer.setframerate(self.mic_sample_rate)
            except Exception as e:
                print(f"Error preparing real-time mic WAV writer: {e}")
                self.mic_wave_writer = None
                self.mic_writer_lock = None
            
            # Start monitoring thread
            self.mic_thread = threading.Thread(target=self._mic_monitor_thread)
            self.mic_thread.daemon = True
            self.mic_thread.start()
            
            # Immediately update chart to show empty state with fixed axes
            self.root.after(0, self._update_mic_bpm_chart)
            
        except Exception as e:
            print(f"Error starting mic monitoring: {e}")
            messagebox.showerror("Error", f"Failed to start microphone monitoring:\n{str(e)}")
            self._stop_mic_monitoring()
    
    def _stop_mic_monitoring(self):
        """
        Stop microphone monitoring, save BPM information, and perform final BPM analysis
        """
        # Update button text (mic icon)
        self.mic_button.config(text="🎤")
        
        # Update state
        self.mic_recording = False

        # Finalize WAV recording
        if getattr(self, 'mic_wave_writer', None) is not None:
            try:
                if getattr(self, 'mic_writer_lock', None) is not None:
                    with self.mic_writer_lock:
                        self.mic_wave_writer.close()
                else:
                    self.mic_wave_writer.close()
            except Exception as e:
                print(f"Error closing mic WAV writer: {e}")
            finally:
                self.mic_wave_writer = None
        
        # Stop stream if it exists
        if hasattr(self, 'mic_stream') and self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except:
                pass
            self.mic_stream = None
        
        # Save complete microphone BPM time series data for comparison
        if hasattr(self, 'mic_time_bpm_pairs') and self.mic_time_bpm_pairs:
            # Store a copy of the complete BPM time series data
            self.recorded_mic_bpm_data = self.mic_time_bpm_pairs.copy()
            print(f"Saved {len(self.recorded_mic_bpm_data)} BPM data points for analysis")
        else:
            self.recorded_mic_bpm_data = []
        
        # Perform final BPM analysis using the entire recorded buffer
        if hasattr(self, 'mic_buffer') and self.mic_buffer and len(self.mic_buffer) > 0:
            # Run final analysis in UI thread
            self.root.after(0, self._perform_final_mic_analysis)
    
    def _mic_monitor_thread(self):
        """
        Thread function for microphone monitoring
        """
        try:
            # Define callback function for audio stream
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Mic status: {status}")
                
                # Add data to buffer
                self.mic_buffer.extend(indata.flatten())

                # Write to WAV in real-time
                if getattr(self, 'mic_wave_writer', None) is not None:
                    try:
                        # Convert float32 [-1,1] to int16
                        samples = np.clip(indata.flatten(), -1.0, 1.0)
                        pcm16 = (samples * 32767.0).astype(np.int16).tobytes()
                        if getattr(self, 'mic_writer_lock', None) is not None:
                            with self.mic_writer_lock:
                                self.mic_wave_writer.writeframes(pcm16)
                        else:
                            self.mic_wave_writer.writeframes(pcm16)
                    except Exception as e:
                        print(f"Error writing mic frames: {e}")
                
                # Keep buffer to last 10 seconds
                max_buffer_size = self.mic_sample_rate * 10
                if len(self.mic_buffer) > max_buffer_size:
                    self.mic_buffer = self.mic_buffer[-max_buffer_size:]
            
            # Create audio stream
            self.mic_stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.mic_sample_rate,
                blocksize=self.mic_chunk_size
            )
            
            # Start stream
            self.mic_stream.start()
            
            # Process buffer periodically
            while self.mic_recording:
                # Prefer quick initial analysis with ~2s buffer, then switch to stable analysis (~7s)
                if len(self.mic_buffer) >= self.mic_sample_rate * 7:
                    # Take a longer window (7 seconds) for more accurate BPM detection
                    analysis_buffer = self.mic_buffer[-self.mic_sample_rate*7:].copy()
                    
                    # Normalize data with better handling of low volume
                    max_val = np.max(np.abs(analysis_buffer))
                    if max_val > 0:
                        analysis_buffer = analysis_buffer / max_val
                    
                    # Perform multiple analyses on overlapping segments for stability
                    segment_duration = 5  # seconds
                    segment_samples = int(segment_duration * self.mic_sample_rate)
                    overlap_samples = int(segment_samples * 0.5)  # 50% overlap
                    
                    segment_bpms = []
                    # Analyze 3 overlapping segments
                    for i in range(3):
                        start_idx = max(0, len(analysis_buffer) - segment_samples - i * overlap_samples)
                        end_idx = start_idx + segment_samples
                        segment = analysis_buffer[start_idx:end_idx]
                        segment_bpm = self.analyzer.analyze_audio_data(segment, self.mic_sample_rate)
                        if segment_bpm > 0:  # Only include valid BPM values
                            segment_bpms.append(segment_bpm)
                    
                    # Calculate consensus BPM from segments
                    if segment_bpms:
                        # Use median as initial estimate
                        current_bpm = np.median(segment_bpms)
                        
                        # Add to history with confidence weighting
                        self.mic_bpm_history.append(current_bpm)
                        
                        # Keep more history for better smoothing (20 values instead of 10)
                        history_size = 20
                        if len(self.mic_bpm_history) > history_size:
                            self.mic_bpm_history = self.mic_bpm_history[-history_size:]
                        
                        # Apply more sophisticated smoothing:
                        # 1. Use median filter to remove outliers
                        # 2. Apply exponential moving average for responsiveness
                        if len(self.mic_bpm_history) >= 3:
                            # Median filter (3-point)
                            median_filtered = []
                            for i in range(len(self.mic_bpm_history)):
                                if i == 0 or i == len(self.mic_bpm_history) - 1:
                                    median_filtered.append(self.mic_bpm_history[i])
                                else:
                                    window = self.mic_bpm_history[i-1:i+2]
                                    median_filtered.append(np.median(window))
                            
                            # Exponential moving average (alpha=0.3 for more weight to recent values)
                            alpha = 0.3
                            ema_bpm = median_filtered[0]
                            for bpm in median_filtered[1:]:
                                ema_bpm = alpha * bpm + (1 - alpha) * ema_bpm
                            
                            self.mic_bpm = ema_bpm
                        else:
                            self.mic_bpm = np.median(self.mic_bpm_history)
                    else:
                        # If no valid segments, keep last BPM if available
                        if not hasattr(self, 'mic_bpm') or self.mic_bpm == 0:
                            self.mic_bpm = 0
                    
                    # Update UI
                    self.root.after(0, lambda: (self.mic_bpm_label.config(text=f"{self.mic_bpm:.1f}") if hasattr(self, 'mic_bpm_label') else None))
                    # Removed real-time comparison - will be done after mic stops
                    
                    # Update microphone BPM chart at the configured interval
                    now_ts = time.time()
                    last_ts = getattr(self, 'mic_last_bpm_sample_ts', self.mic_start_time)
                    interval = getattr(self, 'mic_bpm_sample_interval', 3.0)
                    if now_ts - last_ts >= interval:
                        current_time = now_ts - self.mic_start_time
                        self.mic_time_bpm_pairs.append((current_time, self.mic_bpm))
                        # Update the chart in the UI thread
                        self.root.after(0, self._update_mic_bpm_chart)
                        self.mic_last_bpm_sample_ts = now_ts
                elif len(self.mic_buffer) >= int(self.mic_sample_rate * 2):
                    # Quick initial BPM estimation on ~2 seconds for immediate plotting
                    analysis_buffer = self.mic_buffer[-int(self.mic_sample_rate*2):].copy()
                    max_val = np.max(np.abs(analysis_buffer))
                    if max_val > 0:
                        analysis_buffer = analysis_buffer / max_val
                    quick_bpm = self.analyzer.analyze_audio_data(analysis_buffer, self.mic_sample_rate)
                    if quick_bpm > 0:
                        self.mic_bpm = quick_bpm
                        self.root.after(0, lambda: (self.mic_bpm_label.config(text=f"{self.mic_bpm:.1f}") if hasattr(self, 'mic_bpm_label') else None))
                        now_ts = time.time()
                        last_ts = getattr(self, 'mic_last_bpm_sample_ts', self.mic_start_time)
                        interval = getattr(self, 'mic_bpm_sample_interval', 3.0)
                        if now_ts - last_ts >= interval:
                            current_time = now_ts - self.mic_start_time
                            self.mic_time_bpm_pairs.append((current_time, self.mic_bpm))
                            self.root.after(0, self._update_mic_bpm_chart)
                            self.mic_last_bpm_sample_ts = now_ts
                
                # Sleep for a short time
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error in mic monitor thread: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Microphone monitoring error:\n{str(e)}"))
        finally:
            self._stop_mic_monitoring()
    
    def compare_bpm(self):
        """
        Compare microphone recorded BPM with reference BPM and show detailed analysis
        """
        # Check if we have reference BPM
        if not hasattr(self, 'reference_bpm') or self.reference_bpm == 0:
            messagebox.showwarning("Warning", "Please analyze an audio file first to get reference BPM")
            return
        
        # Check if we have recorded microphone BPM data
        if not hasattr(self, 'recorded_mic_bpm_data') or not self.recorded_mic_bpm_data:
            # If no recorded data, start microphone monitoring
            if not self.mic_recording:
                messagebox.showinfo("Microphone Monitoring", "Start microphone and perform your test. The comparison will be performed when you stop the microphone.")
                self.comparison_active = True
                self._start_mic_monitoring()
            else:
                messagebox.showinfo("Info", "Microphone is already running. The comparison will be performed when you stop the microphone.")
        else:
            # If we have recorded data, perform detailed comparison immediately
            self._perform_detailed_bpm_comparison()
    
    def _perform_final_mic_analysis(self):
        """
        Perform final BPM analysis on the entire recorded buffer using accurate algorithms
        and update comparison results
        """
        try:
            # Get the entire recorded buffer
            full_buffer = np.array(self.mic_buffer)
            
            # Check if buffer is empty
            if len(full_buffer) == 0:
                self.final_mic_bpm = 0
                self.root.after(0, lambda: (self.mic_bpm_label.config(text="Empty recording") if hasattr(self, 'mic_bpm_label') else None))
                if self.comparison_active:
                    messagebox.showinfo("Analysis Result", "No audio data recorded. Please try again.")
                self.mic_buffer = []
                return
            
            # Use the same optimized analysis approach but on the full buffer
            # 1. Split into overlapping segments for stability
            segment_duration = 5  # seconds
            segment_samples = int(segment_duration * self.mic_sample_rate)
            overlap_samples = int(segment_samples * 0.5)  # 50% overlap
            
            segment_bpms = []
            
            # Calculate how many segments we can get from the full buffer
            if len(full_buffer) >= segment_samples:
                num_segments = max(1, int((len(full_buffer) - segment_samples) / overlap_samples) + 1)
                num_segments = min(num_segments, 5)  # Limit to 5 segments maximum
            else:
                num_segments = 0  # Not enough data for segments
            
            # Analyze multiple overlapping segments
            for i in range(num_segments):
                start_idx = min(i * overlap_samples, len(full_buffer) - segment_samples)
                end_idx = start_idx + segment_samples
                segment = full_buffer[start_idx:end_idx]
                
                # Use the same analyzer as before for consistency
                segment_bpm = self.analyzer.analyze_audio_data(segment, self.mic_sample_rate)
                if segment_bpm > 0:  # Only include valid BPM values
                    segment_bpms.append(segment_bpm)
            
            # If no segments (buffer too small), analyze the whole buffer at once
            if not segment_bpms and len(full_buffer) > 0:
                final_bpm = self.analyzer.analyze_audio_data(full_buffer, self.mic_sample_rate)
                if final_bpm > 0:
                    segment_bpms.append(final_bpm)
            
            # Calculate consensus BPM from all valid segments
            if segment_bpms:
                # Apply 3-point median filter to remove outliers if we have enough segments
                if len(segment_bpms) >= 3:
                    # Sort to easily apply median filtering concept
                    sorted_bpms = sorted(segment_bpms)
                    # Take middle values (remove min and max if we have more than 3)
                    if len(sorted_bpms) > 3:
                        valid_bpms = sorted_bpms[1:-1]
                    else:
                        valid_bpms = sorted_bpms
                    
                    # Use median of the filtered values as final BPM
                    self.final_mic_bpm = np.median(valid_bpms)
                else:
                    # For fewer segments, use median or average
                    self.final_mic_bpm = np.median(segment_bpms)
                
                # Show final BPM in UI
                self.root.after(0, lambda: (self.mic_bpm_label.config(text=f"Final: {self.final_mic_bpm:.1f}") if hasattr(self, 'mic_bpm_label') else None))
                
                # If comparison is active, update the comparison results
                if self.comparison_active and hasattr(self, 'reference_bpm') and self.reference_bpm > 0:
                    self._update_bpm_comparison()
            else:
                # No valid BPM detected
                self.final_mic_bpm = 0
                self.root.after(0, lambda: (self.mic_bpm_label.config(text="No BPM detected") if hasattr(self, 'mic_bpm_label') else None))
                
                if self.comparison_active:
                    messagebox.showinfo("Analysis Result", "Could not detect BPM from the recording. Please try again with clearer audio.")
                    
        except Exception as e:
            print(f"Error in final BPM analysis: {e}")
            self.final_mic_bpm = 0
            self.root.after(0, lambda: (self.mic_bpm_label.config(text="Analysis Error") if hasattr(self, 'mic_bpm_label') else None))
        
        # Clear the buffer for next recording
        self.mic_buffer = []

    def _update_mic_bpm_chart(self):
        """
        Update the real-time microphone BPM chart
        """
        try:
            # Clear previous plot
            self.ax_mic.clear()
            
            # Initialize if not exists
            if not hasattr(self, 'mic_time_bpm_pairs'):
                self.mic_time_bpm_pairs = []
            
            # Always initialize times and bpms to avoid undefined variable errors
            times = []
            bpms = []
            
            # Extract data - ensure times and bpms have the same length
            times = []
            bpms = []
            if self.mic_time_bpm_pairs:
                # Process pairs together to ensure times and bpms stay in sync
                for pair in self.mic_time_bpm_pairs:
                    # Skip invalid pairs
                    if len(pair) < 2:
                        continue
                    
                    # Check if time is valid
                    if isinstance(pair[0], (int, float)) and not np.isnan(pair[0]):
                        # For BPM, accept 0 only for the initial point
                        if isinstance(pair[1], (int, float)) and not np.isnan(pair[1]) and (pair[1] > 0 or (len(times) == 0 and pair[0] == 0 and pair[1] == 0)):
                            times.append(pair[0])
                            bpms.append(pair[1])
            
            # If we have valid data points
            if times and bpms and len(times) == len(bpms):
                # Plot BPM curve with solid line - only if we have more than one point or the point is not the initial (0,0)
                if len(bpms) > 1 or (len(bpms) == 1 and bpms[0] > 0):
                    self.ax_mic.plot(times, bpms, 'b-', linewidth=2.5, alpha=0.8, label='Microphone BPM')
                    
                    # Plot BPM points
                    self.ax_mic.scatter(times, bpms, color='r', s=40, alpha=0.7, label='BPM Samples')
                    
                    # Add average BPM line if we have enough data (and exclude initial 0 value)
                    valid_bpms = [bpm for bpm in bpms if bpm > 0]
                    if len(valid_bpms) > 1:
                        avg_bpm = np.mean(valid_bpms)
                        self.ax_mic.axhline(y=avg_bpm, color='g', linestyle='--', alpha=0.7, 
                                          label=f'Current Avg: {avg_bpm:.1f}')
                    
                    # Set appropriate y-axis limits - use only valid BPM values > 0
                    if valid_bpms:
                        min_bpm = max(40, min(valid_bpms) - 10)
                        max_bpm = min(220, max(valid_bpms) + 10)
                        self.ax_mic.set_ylim(min_bpm, max_bpm)
                
                # Set x-axis limits to keep start point at the left but dynamically expand right side
                max_time = max(times)
                if max_time < 5:  # Initial window when just starting
                    self.ax_mic.set_xlim(0, 5)
                else:
                    # Always keep start at 0, expand right side as time increases
                    self.ax_mic.set_xlim(0, max_time + 2)
            
            # Always configure the basic plot elements
            self.ax_mic.set_title("Real-time Microphone BPM", pad=10)
            self.ax_mic.set_xlabel("Time (seconds)")
            self.ax_mic.set_ylabel("BPM")
            
            # Set default y-limits if no valid data
            if not self.mic_time_bpm_pairs or (len(times) == 1 and times[0] == 0 and bpms[0] == 0):
                self.ax_mic.set_ylim(40, 220)
                self.ax_mic.set_xlim(0, 5)
                # Change text to indicate microphone is active but gathering initial data
                self.ax_mic.text(0.5, 0.5, "Microphone active. Analyzing initial audio...",
                                ha='center', va='center', transform=self.ax_mic.transAxes,
                                color='gray', style='italic')
            elif not times or not bpms:
                self.ax_mic.set_ylim(40, 220)
                self.ax_mic.set_xlim(0, 5)
            
            # Add grid
            self.ax_mic.grid(True, alpha=0.3)
            
            # Only add legend if we have valid data to display
            if times and bpms and len(bpms) > 1 or (len(bpms) == 1 and bpms[0] > 0):
                # Check if there are any elements with labels to show in the legend
                if any(line.get_label() not in ("_nolegend_", "") for line in self.ax_mic.get_lines()):
                    self.ax_mic.legend(loc='upper right')
            
            # Ensure title and x-label are not clipped
            try:
                self.fig_mic.subplots_adjust(top=0.86, bottom=0.22)
            except Exception:
                pass
            # Redraw canvas
            self.fig_mic.tight_layout(rect=[0, 0.12, 1, 0.92])
            self.canvas_mic.draw()
            
        except Exception as e:
            print(f"Error updating microphone BPM chart: {e}")
    
    def _update_bpm_comparison(self):
        """
        Update BPM comparison results
        """
        if not self.comparison_active or not hasattr(self, 'reference_bpm') or self.reference_bpm == 0 or not hasattr(self, 'final_mic_bpm') or self.final_mic_bpm == 0:
            return
            
        # Use final_mic_bpm for comparison instead of mic_bpm
        mic_bpm = self.final_mic_bpm
            
        # Calculate BPM difference and percentage
        bpm_diff = abs(self.reference_bpm - mic_bpm)
        bpm_percent_diff = (bpm_diff / self.reference_bpm) * 100
        
        # Determine similarity level
        if bpm_percent_diff < 2:
            similarity = "Perfect Match"
            color = "green"
            feedback = "Excellent timing! You're perfectly in sync."
        elif bpm_percent_diff < 5:
            similarity = "Very Good Match"
            color = "#4CAF50"  # Darker green
            feedback = "Great job! Your timing is very close."
        elif bpm_percent_diff < 10:
            similarity = "Good Match"
            color = "#8BC34A"  # Light green
            feedback = "Good timing. Slight adjustments could make it perfect."
        elif bpm_percent_diff < 15:
            similarity = "Fair Match"
            color = "#FFC107"  # Yellow
            feedback = "Decent timing. Try to [speed up/slow down] to match better."
            if mic_bpm < self.reference_bpm:
                feedback = feedback.replace("[speed up/slow down]", "speed up")
            else:
                feedback = feedback.replace("[speed up/slow down]", "slow down")
        else:
            similarity = "Not Well Matched"
            color = "#F44336"  # Red
            feedback = "Significant timing difference. Try to [speed up/slow down] considerably."
            if self.mic_bpm < self.reference_bpm:
                feedback = feedback.replace("[speed up/slow down]", "speed up")
            else:
                feedback = feedback.replace("[speed up/slow down]", "slow down")
        
        # Update comparison label
        comparison_text = f"{similarity} ({bpm_diff:.1f} BPM difference)"
        if hasattr(self, 'comparison_label'):
            self.comparison_label.config(text=comparison_text, foreground=color)
        
        # Store comparison result
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.comparison_results.append((timestamp, self.reference_bpm, self.mic_bpm, bpm_diff, similarity))
        
        # Keep only last 50 results
        if len(self.comparison_results) > 50:
            self.comparison_results = self.comparison_results[-50:]
            
        # If we have recorded mic BPM data, show detailed analysis
        if hasattr(self, 'recorded_mic_bpm_data') and self.recorded_mic_bpm_data:
            self._perform_detailed_bpm_comparison()
            
    def _perform_detailed_bpm_comparison(self):
        """
        Perform detailed BPM comparison analysis from multiple perspectives
        and show comprehensive evaluation report
        """
        # Calculate metrics starting from current seek positions of both progress bars
        start_ref = float(self.seek_var_ref.get()) if hasattr(self, 'seek_var_ref') else 0.0
        start_mic = float(self.seek_var_mic.get()) if hasattr(self, 'seek_var_mic') else 0.0
        
        # Slice mic BPM data from its seek position onward
        mic_pairs = self.recorded_mic_bpm_data or []
        mic_pairs = [(t, b) for (t, b) in mic_pairs if t >= start_mic]
        mic_bpms = [bpm for _, bpm in mic_pairs if bpm > 0]
        if not mic_bpms:
            messagebox.showinfo("Comparison Result", "No valid microphone BPM data found from the current mic position onward.")
            return
        
        # Slice reference BPM data from its seek position onward and compute segment average
        ref_pairs = getattr(self, 'time_bpm_pairs', []) or []
        ref_pairs = [(t, b) for (t, b) in ref_pairs if t >= start_ref]
        ref_bpms = [bpm for _, bpm in ref_pairs if bpm > 0]
        ref_bpm_for_compare = (np.mean(ref_bpms) if ref_bpms else float(getattr(self, 'reference_bpm', 0.0))) or 0.0
        if ref_bpm_for_compare <= 0:
            messagebox.showinfo("Comparison Result", "No valid reference BPM data found from the current reference position onward.")
            return
        
        # Speed analysis
        avg_mic_bpm = np.mean(mic_bpms)
        median_mic_bpm = np.median(mic_bpms)
        bpm_diff = abs(avg_mic_bpm - ref_bpm_for_compare)
        bpm_percent_diff = (bpm_diff / ref_bpm_for_compare) * 100 if ref_bpm_for_compare > 0 else 0
        
        # Rhythm stability analysis (standard deviation)
        bpm_std = np.std(mic_bpms)
        stability_score = 100 - min(100, bpm_std * 10)  # Higher is more stable
        
        # Tempo consistency analysis
        # Check how many BPM readings are within certain percentage of the reference segment BPM
        within_2_percent = sum(1 for bpm in mic_bpms if abs(bpm - ref_bpm_for_compare) / ref_bpm_for_compare * 100 <= 2)
        within_5_percent = sum(1 for bpm in mic_bpms if abs(bpm - ref_bpm_for_compare) / ref_bpm_for_compare * 100 <= 5)
        within_10_percent = sum(1 for bpm in mic_bpms if abs(bpm - ref_bpm_for_compare) / ref_bpm_for_compare * 100 <= 10)
        
        consistency_2 = (within_2_percent / len(mic_bpms)) * 100
        consistency_5 = (within_5_percent / len(mic_bpms)) * 100
        consistency_10 = (within_10_percent / len(mic_bpms)) * 100
        
        # Timing progression analysis (detect speeding up/slowing down trends)
        # Split mic data into thirds from the current starting point
        if len(mic_pairs) >= 3:
            third = len(mic_pairs) // 3
            first_third_bpms = [bpm for _, bpm in mic_pairs[:third] if bpm > 0]
            middle_third_bpms = [bpm for _, bpm in mic_pairs[third:2*third] if bpm > 0]
            last_third_bpms = [bpm for _, bpm in mic_pairs[2*third:] if bpm > 0]
            
            first_avg = np.mean(first_third_bpms) if first_third_bpms else 0
            middle_avg = np.mean(middle_third_bpms) if middle_third_bpms else 0
            last_avg = np.mean(last_third_bpms) if last_third_bpms else 0
            
            # Determine trend
            if first_avg > 0 and last_avg > 0:
                trend_diff = last_avg - first_avg
                trend_percent = (abs(trend_diff) / first_avg) * 100 if first_avg > 0 else 0
                
                if trend_diff > 0 and trend_percent > 5:
                    timing_trend = f"Speeding up (+{trend_percent:.1f}%)"
                elif trend_diff < 0 and trend_percent > 5:
                    timing_trend = f"Slowing down (-{trend_percent:.1f}%)"
                else:
                    timing_trend = "Consistent"
            else:
                timing_trend = "Insufficient data"
        else:
            timing_trend = "Insufficient data"
        
        # Generate evaluation and feedback
        evaluation, suggestions = self._generate_evaluation_and_suggestions(
            bpm_percent_diff, stability_score, consistency_10, timing_trend
        )
        
        # Create and show comparison report window
        self._show_comparison_report(
            reference_bpm=ref_bpm_for_compare,
            avg_mic_bpm=avg_mic_bpm,
            median_mic_bpm=median_mic_bpm,
            bpm_diff=bpm_diff,
            bpm_percent_diff=bpm_percent_diff,
            stability_score=stability_score,
            consistency_2=consistency_2,
            consistency_5=consistency_5,
            consistency_10=consistency_10,
            timing_trend=timing_trend,
            evaluation=evaluation,
            suggestions=suggestions
        )
        
    def _generate_evaluation_and_suggestions(self, bpm_percent_diff, stability_score, consistency_10, timing_trend):
        """
        Generate comprehensive evaluation and improvement suggestions based on comparison metrics
        """
        evaluation = []
        suggestions = []
        
        # Speed evaluation
        if bpm_percent_diff < 3:
            evaluation.append("✅ **Speed Accuracy**: Excellent - Your performance speed almost perfectly matches the score")
        elif bpm_percent_diff < 7:
            evaluation.append("✅ **Speed Accuracy**: Good - Your performance speed is very close to the score")
        elif bpm_percent_diff < 12:
            evaluation.append("⚠️ **Speed Accuracy**: Moderate - There is some difference between your performance speed and the score")
        else:
            evaluation.append("❌ **Speed Accuracy**: Low - There is a significant difference between your performance speed and the score")
            if hasattr(self, 'mic_bpm') and self.mic_bpm > self.reference_bpm:
                suggestions.append("- Try slowing down your playing speed to match the score")
            else:
                suggestions.append("- Try speeding up your playing to match the score")
        
        # Rhythm stability evaluation
        if stability_score >= 80:
            evaluation.append("✅ **Rhythm Stability**: Excellent - Your rhythm is very stable")
        elif stability_score >= 60:
            evaluation.append("✅ **Rhythm Stability**: Good - Your rhythm is relatively stable")
        elif stability_score >= 40:
            evaluation.append("⚠️ **Rhythm Stability**: Average - Your rhythm has fluctuations")
        else:
            evaluation.append("❌ **Rhythm Stability**: Poor - Your rhythm has significant fluctuations")
            suggestions.append("- Practice with a metronome to improve your sense of rhythm")
            suggestions.append("- Practice in segments to gradually improve rhythm stability")
        
        # Timing consistency evaluation
        if consistency_10 >= 90:
            evaluation.append("✅ **Timing Precision**: Excellent - Your performance timing is very accurate")
        elif consistency_10 >= 70:
            evaluation.append("✅ **Timing Precision**: Good - Your performance timing is mostly accurate")
        elif consistency_10 >= 50:
            evaluation.append("⚠️ **Timing Precision**: Needs improvement - Your performance timing is somewhat unstable")
        else:
            evaluation.append("❌ **Timing Precision**: Poor - Your performance timing is unstable")
            suggestions.append("- Focus on the accuracy of bar lines and beats")
            suggestions.append("- Practice slowly first, then gradually increase speed to ensure accurate note values")
        
        # Tempo trend evaluation
        if "Consistent" in timing_trend:
            evaluation.append("✅ **Expression & Style**: Excellent - Your performance maintains stable speed, demonstrating professional musical expression")
        elif "Speeding up" in timing_trend:
            evaluation.append("⚠️ **Expression & Style**: Acceleration tendency - Your performance shows a trend of gradually speeding up")
            suggestions.append("- Pay attention to controlling speed, especially during transitions between sections")
            suggestions.append("- Practice with a metronome to develop a stable sense of tempo")
        elif "Slowing down" in timing_trend:
            evaluation.append("⚠️ **Expression & Style**: Deceleration tendency - Your performance shows a trend of gradually slowing down")
            suggestions.append("- Focus on maintaining energy and momentum throughout the performance")
            suggestions.append("- Pay attention to breathing and physical movements to maintain continuity")
        else:
            evaluation.append("ℹ️ **Expression & Style**: Insufficient data - Unable to accurately evaluate expression and style")
        
        # Overall suggestions based on all metrics
        if bpm_percent_diff < 5 and stability_score >= 70 and consistency_10 >= 80:
            suggestions.append("- Keep up the good work! You can try adding more musical expression without affecting rhythm")
            suggestions.append("- You can start focusing on emotional expression and tone variations")
        elif bpm_percent_diff >= 15 or stability_score < 50 or consistency_10 < 60:
            suggestions.append("- It's recommended to start with the basics, practicing with a metronome in stages")
            suggestions.append("- Master accurate rhythm at slow speeds first, then gradually increase the tempo")
        
        return evaluation, suggestions
        
    def _show_comparison_report(self, **metrics):
        """
        Create and display a detailed BPM comparison report window
        """
        # Create new window
        report_window = tk.Toplevel(self.root)
        report_window.title("BPM Comparison Analysis Report")
        report_window.geometry("1125x650")
        report_window.resizable(True, True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(report_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Summary & Metrics (merged)
        summary_tab = ttk.Frame(notebook)
        notebook.add(summary_tab, text="Summary & Metrics")
        
        # Top-right control bar (model dropdown + Generate button)
        control_bar = ttk.Frame(summary_tab)
        control_bar.pack(fill=tk.X, padx=10, pady=(10, 0))
        model_var = tk.StringVar(value="deepseek-v3")
        model_box = ttk.Combobox(control_bar, textvariable=model_var, values=["deepseek-v3", "deepseek-r1"], state="readonly", width=18)
        model_box.pack(side=tk.RIGHT, padx=(6, 0))
        generate_btn = ttk.Button(control_bar, text="Generate")
        generate_btn.pack(side=tk.RIGHT)
        
        # Create summary text widget with scrollbar
        summary_frame = ttk.Frame(summary_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        summary_text = tk.Text(summary_frame, wrap=tk.WORD, font=("Arial", 10))
        summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(summary_frame, command=summary_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        summary_text.config(yscrollcommand=scrollbar.set)
        
        # Wire Generate to call DeepSeek with current data
        generate_btn.config(command=lambda: self._generate_deepseek_summary(metrics, model_var.get(), summary_text))
        
        # Add summary content
        summary_text.insert(tk.END, "BPM COMPARISON ANALYSIS REPORT\n\n")
        
        summary_text.insert(tk.END, "📊 COMPARISON METRICS:\n\n")
        summary_text.insert(tk.END, f"Reference BPM (Score): {metrics['reference_bpm']:.1f}\n")
        summary_text.insert(tk.END, f"Your Average BPM: {metrics['avg_mic_bpm']:.1f}\n")
        summary_text.insert(tk.END, f"Your Median BPM: {metrics['median_mic_bpm']:.1f}\n")
        summary_text.insert(tk.END, f"BPM Difference: {metrics['bpm_diff']:.1f} ({metrics['bpm_percent_diff']:.1f}%)\n\n")
        
        summary_text.insert(tk.END, "🎯 DETAILED EVALUATION:\n\n")
        for item in metrics['evaluation']:
            summary_text.insert(tk.END, f"{item}\n")
        
        summary_text.insert(tk.END, "\n💡 IMPROVEMENT SUGGESTIONS:\n\n")
        for suggestion in metrics['suggestions']:
            summary_text.insert(tk.END, f"{suggestion}\n")
        
        # Make text widget read-only
        summary_text.config(state=tk.DISABLED)
        
        # Append Advanced Metrics to Summary tab
        summary_text.config(state=tk.NORMAL)
        summary_text.insert(tk.END, "\n\nADVANCED PERFORMANCE METRICS\n\n")
        
        summary_text.insert(tk.END, "📈 RHYTHM STABILITY:\n")
        summary_text.insert(tk.END, f"Stability Score: {metrics['stability_score']:.1f}/100\n")
        
        summary_text.insert(tk.END, "\n🎯 TIMING CONSISTENCY:\n")
        summary_text.insert(tk.END, f"Within ±2% of reference: {metrics['consistency_2']:.1f}%\n")
        summary_text.insert(tk.END, f"Within ±5% of reference: {metrics['consistency_5']:.1f}%\n")
        summary_text.insert(tk.END, f"Within ±10% of reference: {metrics['consistency_10']:.1f}%\n")
        
        summary_text.insert(tk.END, "\n⏱️ TEMPO PROGRESSION:\n")
        summary_text.insert(tk.END, f"Performance Trend: {metrics['timing_trend']}\n")
        
        summary_text.insert(tk.END, "\n📊 INTERPRETATION:\n")
        if metrics['stability_score'] >= 80:
            summary_text.insert(tk.END, "• Your rhythm is highly stable, showing professional-level control.\n")
        elif metrics['stability_score'] >= 60:
            summary_text.insert(tk.END, "• Your rhythm is generally stable with minor fluctuations.\n")
        else:
            summary_text.insert(tk.END, "• Your rhythm has noticeable fluctuations that should be addressed.\n")
        
        if metrics['consistency_10'] >= 90:
            summary_text.insert(tk.END, "• You maintained excellent timing throughout the performance.\n")
        elif metrics['consistency_10'] >= 70:
            summary_text.insert(tk.END, "• Your timing was mostly consistent with occasional variations.\n")
        else:
            summary_text.insert(tk.END, "• Your timing needs more consistency to match the score accurately.\n")
        
        summary_text.config(state=tk.DISABLED)
        
        # Tab 3: Visual Comparison
        visual_tab = ttk.Frame(notebook)
        notebook.add(visual_tab, text="Visual Comparison")
        
        # Create figure for visual comparison with distribution subplots below
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#f0f0f0')  # Match Tkinter background
        gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[2, 1, 1, 1])
        ax_ts = fig.add_subplot(gs[0, :])
        ax_violin = fig.add_subplot(gs[1, :])
        ax_box = fig.add_subplot(gs[2, :])
        ax_heatmap = fig.add_subplot(gs[3, :])
        # Increase vertical spacing between subplots and remove line between top and middle
        fig.subplots_adjust(hspace=0.40)
        try:
            ax_ts.spines['bottom'].set_visible(False)
        except Exception:
            pass
        
        # Plot microphone BPM over time (filtered by selected range)
        selected_start = float(getattr(self, 'ref_range_start', 0.0))
        selected_end = float(getattr(self, 'ref_range_end', 0.0))
        mic_pairs = self.recorded_mic_bpm_data or []
        if selected_end > selected_start:
            mic_pairs = [(t, b) for (t, b) in mic_pairs if t >= selected_start and t <= selected_end]
        times = [t for t, _ in mic_pairs]
        mic_bpms = [b for _, b in mic_pairs]
        
        # Filter out zero BPM values
        valid_times = [t for t, b in zip(times, mic_bpms) if b > 0]
        valid_bpms = [b for b in mic_bpms if b > 0]
        
        if valid_times and valid_bpms:
            # Top: real-time microphone BPM + reference + mean + faster/slower fill (extracted)
            sheet_bpm = metrics['reference_bpm']
            ref_series = plot_bpm_timeseries(
                ax_ts,
                valid_times=valid_times,
                valid_bpms=valid_bpms,
                sheet_bpm=sheet_bpm,
                reference_pairs=getattr(self, 'time_bpm_pairs', None)
            )

            # Distributions: violin and box plots (extracted)
            plot_distributions(ax_violin, ax_box, valid_bpms, ref_series)
            
            # Heatmap: segment-wise tempo deviation (%) — extracted to bpm_visuals module
            im = plot_deviation_heatmap(
                ax_heatmap,
                valid_times=valid_times,
                valid_bpms=valid_bpms,
                ref_series=ref_series,
                sheet_bpm=sheet_bpm,
                segment_count=8
            )
            cbar = fig.colorbar(im, ax=ax_heatmap, pad=0.01)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label('Deviation (%)', fontsize=6)
            
            # Adjust layout: leave top margin to avoid title clipping on resize
            fig.subplots_adjust(top=0.92)
            
            # Removed divider line between top time series and lower plots
            
            # Embed plot in a vertically scrollable Tkinter container
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            scroll_container = ttk.Frame(visual_tab)
            scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            tk_canvas = tk.Canvas(scroll_container, highlightthickness=0)
            tk_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=tk_canvas.yview)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            tk_canvas.configure(yscrollcommand=vsb.set)
            
            inner = ttk.Frame(tk_canvas)
            window_id = tk_canvas.create_window((0, 0), window=inner, anchor='nw')
            
            fig_canvas = FigureCanvasTkAgg(fig, master=inner)
            fig_canvas.draw()
            fig_widget = fig_canvas.get_tk_widget()
            fig_widget.pack(fill=tk.X, expand=False)
            
            def _update_scrollregion(event=None):
                tk_canvas.configure(scrollregion=tk_canvas.bbox('all'))
            inner.bind('<Configure>', _update_scrollregion)
            
            def _resize_inner(event):
                tk_canvas.itemconfigure(window_id, width=event.width)
            tk_canvas.bind('<Configure>', _resize_inner)

            # Enable mouse wheel scrolling
            def _on_mousewheel(event):
                delta = event.delta if hasattr(event, 'delta') else 0
                step = -1 if delta > 0 else (1 if delta < 0 else 0)
                if step != 0:
                    tk_canvas.yview_scroll(step, 'units')

            # Bind wheel events to widgets
            tk_canvas.bind('<MouseWheel>', _on_mousewheel)
            inner.bind('<MouseWheel>', _on_mousewheel)
            fig_widget.bind('<MouseWheel>', _on_mousewheel)

            # Linux support for wheel
            tk_canvas.bind('<Button-4>', lambda e: tk_canvas.yview_scroll(-1, 'units'))
            tk_canvas.bind('<Button-5>', lambda e: tk_canvas.yview_scroll(1, 'units'))
            inner.bind('<Button-4>', lambda e: tk_canvas.yview_scroll(-1, 'units'))
            inner.bind('<Button-5>', lambda e: tk_canvas.yview_scroll(1, 'units'))
            fig_widget.bind('<Button-4>', lambda e: tk_canvas.yview_scroll(-1, 'units'))
            fig_widget.bind('<Button-5>', lambda e: tk_canvas.yview_scroll(1, 'units'))

            # Activate wheel capture within scroll area
            def _activate_wheel(event=None):
                tk_canvas.bind_all('<MouseWheel>', _on_mousewheel)
            def _deactivate_wheel(event=None):
                tk_canvas.unbind_all('<MouseWheel>')
            scroll_container.bind('<Enter>', _activate_wheel)
            scroll_container.bind('<Leave>', _deactivate_wheel)
        else:
            no_data_label = ttk.Label(visual_tab, text="No valid BPM data available for visualization.")
            no_data_label.pack(pady=50)
        
        # Add close button
        close_button = ttk.Button(report_window, text="Close", command=report_window.destroy)
        close_button.pack(pady=10)

    def _generate_deepseek_summary(self, metrics, model_name, summary_text_widget):
        """
        Call DeepSeek chat API using current comparison data and append the
        returned feedback to the Summary & Metrics tab. Runs in a background thread.
        """
        def build_prompt():
            ref_bpm = metrics.get('reference_bpm', 0)
            # Prefer recorded microphone series
            if hasattr(self, 'recorded_mic_bpm_data') and self.recorded_mic_bpm_data:
                pairs = self.recorded_mic_bpm_data
            else:
                pairs = getattr(self, 'mic_time_bpm_pairs', [])
            max_items = 60
            pairs_str = ", ".join([f"{round(t,1)}s:{round(b,1)}" for t, b in pairs[:max_items]]) if pairs else "(no data)"
            avg_mic = metrics.get('avg_mic_bpm', 0)
            median_mic = metrics.get('median_mic_bpm', 0)
            diff = metrics.get('bpm_diff', 0)
            percent = metrics.get('bpm_percent_diff', 0)
            instruction = (
                f"Reference BPM (Score): {ref_bpm:.1f}\n"
                f"Recorded BPM time-series (time:value, up to {max_items} items): {pairs_str}\n"
                f"Recorded BPM stats — mean: {avg_mic:.1f}, median: {median_mic:.1f}, difference vs reference: {diff:.1f} ({percent:.1f}%).\n"
                "Please compare the recorded BPM against the reference BPM, provide an evaluation of the performance, and suggest improvements.\n"
                "Respond in English, concise and structured with these sections: Overall Evaluation, Issues Observed, Actionable Improvement Suggestions."
            )
            return instruction

        def request_thread():
            try:
                instruction = build_prompt()
                import os, json, urllib.request, urllib.error
                api_key = os.environ.get('DEEPSEEK_API_KEY')
                if not api_key:
                    # Fallback to project config.json
                    try:
                        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
                        if os.path.exists(cfg_path):
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                            api_key = cfg.get("DEEPSEEK_API_KEY")
                    except Exception:
                        api_key = None
                if not api_key:
                    self.root.after(0, lambda: messagebox.showerror("DeepSeek", "Missing DEEPSEEK_API_KEY. Set environment variable or add to config.json."))
                    return
                url = "https://api.deepseek.com/v1/chat/completions"
                def _map_model(name):
                    n = (name or "").strip().lower()
                    if n in ("deepseek-v3", "deepseek_chat", "deepseek-chat", "v3"):
                        return "deepseek-chat"
                    if n in ("deepseek-r1", "r1", "deepseek-reasoner", "deepseek-reasoner"):
                        return "deepseek-reasoner"
                    return "deepseek-chat"
                payload = {
                    "model": _map_model(model_name),
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant specialized in musical tempo analysis."},
                        {"role": "user", "content": instruction},
                    ],
                    "stream": False
                }
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(url, data=data, method='POST')
                req.add_header('Authorization', f'Bearer {api_key}')
                req.add_header('Content-Type', 'application/json')

                # Use certifi CA bundle to avoid SSL certificate verify failures on macOS
                try:
                    import ssl, certifi
                    ssl_context = ssl.create_default_context(cafile=certifi.where())
                except Exception:
                    import ssl
                    ssl_context = ssl.create_default_context()

                try:
                    with urllib.request.urlopen(req, timeout=180, context=ssl_context) as resp:
                        body = resp.read().decode('utf-8')
                except urllib.error.HTTPError as e:
                    try:
                        body = e.read().decode('utf-8', 'ignore')
                    except Exception:
                        body = ''
                    raise Exception(f"HTTP {e.code}: {body}")
                except urllib.error.URLError as e:
                    raise Exception(f"Network error: {getattr(e, 'reason', e)}")
                result = json.loads(body)
                msg = result.get('choices', [{}])[0].get('message', {})
                content = msg.get('content', '')
                reasoning = msg.get('reasoning', '')
                if reasoning:
                    content = f"### Reasoning\n{reasoning}\n\n### Answer\n{content}" if content else f"### Reasoning\n{reasoning}"
                if not content:
                    content = "(No content returned)"
                def append_text():
                    try:
                        summary_text_widget.config(state=tk.NORMAL)
                        summary_text_widget.insert(tk.END, "\n\nAI Feedback (DeepSeek)\n\n")
                        # Full Markdown rendering: headings, bold/italic, inline code, code blocks,
                        # lists (ordered/unordered with indentation), blockquotes, horizontal rules, links.
                        import re, webbrowser
                        def insert_markdown(widget, md):
                            # Tag setup
                            try:
                                widget.tag_configure('h1', font=('Arial', 14, 'bold'))
                                widget.tag_configure('h2', font=('Arial', 13, 'bold'))
                                widget.tag_configure('h3', font=('Arial', 12, 'bold'))
                                widget.tag_configure('h4', font=('Arial', 11, 'bold'))
                                widget.tag_configure('h5', font=('Arial', 10, 'bold'))
                                widget.tag_configure('h6', font=('Arial', 10))
                                widget.tag_configure('bold', font=('Arial', 10, 'bold'))
                                widget.tag_configure('italic', font=('Arial', 10, 'italic'))
                                widget.tag_configure('code', font=('Courier', 10))
                                widget.tag_configure('codeblock', font=('Courier', 10), background='#f5f5f5')
                                widget.tag_configure('quote', lmargin1=20, lmargin2=20, background='#f9f9f9')
                                widget.tag_configure('hr', foreground='#888888')
                                widget.tag_configure('list1', lmargin1=20, lmargin2=20)
                                widget.tag_configure('list2', lmargin1=40, lmargin2=40)
                                widget.tag_configure('list3', lmargin1=60, lmargin2=60)
                            except Exception:
                                pass

                            link_counter = 0

                            def apply_inline(text):
                                # Returns list of (segment_text, tags, extra)
                                segments = []
                                i = 0
                                while i < len(text):
                                    # Links: [text](url)
                                    m = re.search(r"\[([^\]]+)\]\(([^)]+)\)", text[i:])
                                    if m:
                                        pre = text[i:i+m.start()]
                                        if pre:
                                            segments.append((pre, [], None))
                                        link_text = m.group(1)
                                        link_url = m.group(2)
                                        segments.append((link_text, ['link'], link_url))
                                        i += m.end()
                                        continue
                                    # Inline code: `code`
                                    m = re.search(r"`([^`]+)`", text[i:])
                                    if m:
                                        pre = text[i:i+m.start()]
                                        if pre:
                                            segments.append((pre, [], None))
                                        code_text = m.group(1)
                                        segments.append((code_text, ['code'], None))
                                        i += m.end()
                                        continue
                                    # Bold: **text** or __text__
                                    m = re.search(r"\*\*([^*]+)\*\*|__([^_]+)__", text[i:])
                                    if m:
                                        pre = text[i:i+m.start()]
                                        if pre:
                                            segments.append((pre, [], None))
                                        bold_text = m.group(1) if m.group(1) is not None else m.group(2)
                                        segments.append((bold_text, ['bold'], None))
                                        i += m.end()
                                        continue
                                    # Italic: *text* or _text_
                                    m = re.search(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_", text[i:])
                                    if m:
                                        pre = text[i:i+m.start()]
                                        if pre:
                                            segments.append((pre, [], None))
                                        italic_text = m.group(1) if m.group(1) is not None else m.group(2)
                                        segments.append((italic_text, ['italic'], None))
                                        i += m.end()
                                        continue
                                    # No more markup
                                    segments.append((text[i:], [], None))
                                    break
                                return segments

                            # Create per-link tags with click behavior
                            def insert_with_tags(line_text, base_tag=None):
                                nonlocal link_counter
                                segments = apply_inline(line_text)
                                start_index = widget.index(tk.END)
                                for seg_text, tags, extra in segments:
                                    segment_start = widget.index(tk.END)
                                    widget.insert(tk.END, seg_text)
                                    applied_tags = []
                                    if base_tag:
                                        applied_tags.append(base_tag)
                                    for t in tags:
                                        if t == 'link':
                                            tag_name = f"link_{link_counter}"
                                            link_counter += 1
                                            try:
                                                widget.tag_configure(tag_name, foreground='blue', underline=True)
                                                def _open(url=extra):
                                                    try:
                                                        webbrowser.open(url)
                                                    except Exception:
                                                        pass
                                                widget.tag_bind(tag_name, '<Button-1>', lambda e, f=_open: f())
                                            except Exception:
                                                pass
                                            applied_tags.append(tag_name)
                                        else:
                                            applied_tags.append(t)
                                    for t in applied_tags:
                                        try:
                                            widget.tag_add(t, segment_start, widget.index(tk.END))
                                        except Exception:
                                            pass
                                widget.insert(tk.END, "\n")

                            # Parse block-level elements
                            lines = md.splitlines()
                            in_codeblock = False
                            codeblock_buffer = []
                            for raw in lines:
                                line = raw.rstrip('\n')
                                if in_codeblock:
                                    if line.strip().startswith('```'):
                                        # Flush code block
                                        code_text = "\n".join(codeblock_buffer) + "\n"
                                        start = widget.index(tk.END)
                                        widget.insert(tk.END, code_text)
                                        try:
                                            widget.tag_add('codeblock', start, widget.index(tk.END))
                                        except Exception:
                                            pass
                                        codeblock_buffer = []
                                        in_codeblock = False
                                    else:
                                        codeblock_buffer.append(line)
                                    continue

                                # Start code block
                                if line.strip().startswith('```'):
                                    in_codeblock = True
                                    codeblock_buffer = []
                                    continue

                                if not line.strip():
                                    widget.insert(tk.END, "\n")
                                    continue

                                # Headings
                                m = re.match(r"^(#{1,6})\s+(.*)$", line)
                                if m:
                                    level = len(m.group(1))
                                    text = m.group(2)
                                    tag = f"h{level}"
                                    insert_with_tags(text, base_tag=tag)
                                    continue

                                # Horizontal rule
                                if re.match(r"^\s*(\*{3,}|-{3,}|_{3,})\s*$", line):
                                    widget.insert(tk.END, "-" * 80 + "\n")
                                    continue

                                # Blockquote
                                if re.match(r"^>\s?(.*)$", line):
                                    quote_text = re.sub(r"^>\s?", "", line)
                                    insert_with_tags(quote_text, base_tag='quote')
                                    continue

                                # Lists (unordered and ordered), with indentation
                                lm = re.match(r"^(\s*)([-*+]\s+)(.*)$", line)
                                om = re.match(r"^(\s*)(\d+\.\s+)(.*)$", line)
                                if lm or om:
                                    indent = len((lm or om).group(1)) // 2
                                    indent_tag = 'list1' if indent == 1 else ('list2' if indent == 2 else ('list3' if indent >= 3 else None))
                                    bullet = '• ' if lm else (om.group(2))
                                    content_text = (lm or om).group(3)
                                    insert_with_tags(bullet + content_text, base_tag=indent_tag)
                                    continue

                                # Tables: simple pipe-delimited rows
                                if '|' in line:
                                    # Render as monospaced row
                                    insert_with_tags(line.replace('|', ' | '), base_tag='code')
                                    continue

                                # Paragraph
                                insert_with_tags(line)

                        insert_markdown(summary_text_widget, content.strip())
                        summary_text_widget.config(state=tk.DISABLED)
                    except Exception:
                        pass
                self.root.after(0, append_text)
            except Exception as e:
                self.root.after(0, lambda err=e: messagebox.showerror("DeepSeek", f"API error: {err}"))

        try:
            summary_text_widget.config(state=tk.NORMAL)
            summary_text_widget.insert(tk.END, "\n" + ("-" * 80) + "\nGenerating AI feedback...\n")
            summary_text_widget.config(state=tk.DISABLED)
        except Exception:
            pass
        import threading
        threading.Thread(target=request_thread, daemon=True).start()

    def show_mic_bpm_timeseries(self):
        """Show microphone BPM time series data in a new window"""
        try:
            # Prefer recorded data if available; otherwise use live mic time series
            if hasattr(self, 'recorded_mic_bpm_data') and self.recorded_mic_bpm_data:
                data = self.recorded_mic_bpm_data
            elif hasattr(self, 'mic_time_bpm_pairs') and self.mic_time_bpm_pairs:
                data = self.mic_time_bpm_pairs
            else:
                messagebox.showinfo("Information", "No microphone BPM time series data found")
                return

            timeseries_window = tk.Toplevel(self.root)
            timeseries_window.title("Microphone BPM Variation Over Time")
            timeseries_window.geometry("600x400")
            timeseries_window.resizable(True, True)

            title_label = ttk.Label(
                timeseries_window,
                text="Microphone BPM Variations",
                font=("Arial", 12, "bold")
            )
            title_label.pack(pady=10)

            table_frame = ttk.Frame(timeseries_window)
            table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
            y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

            tree = ttk.Treeview(
                table_frame,
                columns=("time", "bpm"),
                show="headings",
                yscrollcommand=y_scroll.set,
                xscrollcommand=x_scroll.set
            )
            tree.heading("time", text="Time (min:sec)")
            tree.heading("bpm", text="BPM")
            tree.column("time", anchor=tk.CENTER, width=150)
            tree.column("bpm", anchor=tk.CENTER, width=100)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            y_scroll.config(command=tree.yview)
            x_scroll.config(command=tree.xview)

            for pair in data:
                try:
                    t, bpm = pair
                    t_str = self._format_time(float(t))
                    bpm_str = f"{float(bpm):.1f}" if float(bpm) > 0 else "--"
                except Exception:
                    t_str = "--"
                    bpm_str = "--"
                tree.insert("", "end", values=(t_str, bpm_str))

            # Bottom controls: export button and statistics (match file window style)
            button_frame = ttk.Frame(timeseries_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)

            export_btn = ttk.Button(button_frame, text="Export Data", command=self.export_mic_bpm_timeseries)
            export_btn.pack(side=tk.RIGHT)

            try:
                bpm_values = [float(b) for _, b in data if float(b) > 0]
                if bpm_values:
                    avg_bpm = np.mean(bpm_values)
                    min_bpm = np.min(bpm_values)
                    max_bpm = np.max(bpm_values)
                    std_bpm = np.std(bpm_values)
                else:
                    avg_bpm = min_bpm = max_bpm = std_bpm = 0.0
                stats_text = f"Statistics: Average BPM = {avg_bpm:.1f}, Minimum BPM = {min_bpm:.1f}, Maximum BPM = {max_bpm:.1f}, Standard Deviation = {std_bpm:.1f}"
            except Exception:
                stats_text = "Statistics: No data"
            stats_label = ttk.Label(timeseries_window, text=stats_text, foreground="#555555")
            stats_label.pack(pady=5)
        except Exception as e:
            print(f"Error showing mic BPM timeseries: {e}")

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
    
    def export_mic_bpm_timeseries(self):
        """
        Export microphone BPM time series data to CSV format
        """
        # Determine source
        if hasattr(self, 'recorded_mic_bpm_data') and self.recorded_mic_bpm_data:
            pairs = self.recorded_mic_bpm_data
        elif hasattr(self, 'mic_time_bpm_pairs') and self.mic_time_bpm_pairs:
            pairs = self.mic_time_bpm_pairs
        else:
            messagebox.showinfo("Information", "No microphone BPM time series data found")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Microphone BPM Data"
            )
            if not file_path:
                return
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time (seconds)", "Time (min:sec)", "Mic BPM"])
                for time_seconds, bpm in pairs:
                    minutes = int(time_seconds // 60)
                    seconds = int(time_seconds % 60)
                    time_str = f"{minutes:02d}:{seconds:02d}"
                    bpm_out = round(float(bpm), 1) if isinstance(bpm, (int, float)) and float(bpm) > 0 else "--"
                    writer.writerow([time_seconds, time_str, bpm_out])
            messagebox.showinfo("Success", f"Microphone BPM data successfully exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting microphone BPM data:\n{str(e)}")

    def get_filename(self):
        """
        Get the name of the currently selected file
        
        Returns:
            Filename (without path), or "Unknown File" if no file is selected
        """
        if hasattr(self, 'audio_file') and self.audio_file:
            return os.path.basename(self.audio_file)
        return "Unknown File"
    
    def on_closing(self):
        """
        Handle window closing event
        """
        # Stop any ongoing processes
        self.analyzing = False
        self.playing = False
        self.mic_recording = False
        self.comparison_active = False
        
        # Stop playback
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        
        # Remove temporary files
        if hasattr(self, 'temp_wav_file') and self.temp_wav_file and os.path.exists(self.temp_wav_file):
            try:
                os.remove(self.temp_wav_file)
            except:
                pass
        if hasattr(self, 'temp_mic_wav_file') and self.temp_mic_wav_file and os.path.exists(self.temp_mic_wav_file):
            try:
                # Only remove temporary mic playback files, keep recorded mic files
                if os.path.basename(self.temp_mic_wav_file).startswith('temp_mic_playback_'):
                    os.remove(self.temp_mic_wav_file)
            except:
                pass
        
        # Close microphone stream
        if hasattr(self, 'mic_stream') and self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except:
                pass
        
        # Destroy window
        self.root.destroy()

def main():
    """
    Program entry point
    """
    root = tk.Tk()
    app = BPMGUIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()