#!/usr/bin/env python3
"""
BPM Analyzer application with PySide6 GUI, maintaining original interface structure
"""

import os
import sys
import threading
import time
import warnings
import json
import urllib
from datetime import datetime

# Set environment variables to hide unwanted prompts and configure matplotlib
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["QT_API"] = "pyside6"  # Explicitly use PySide6 for Qt bindings

# Suppress deprecated pkg_resources warning from pygame
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# Fix for macOS IMKCFRunLoopWakeUpReliable error
os.environ["QT_MAC_WANTS_LAYER"] = "1"

# Import dependencies
import numpy as np
import pygame

# Import PySide6 first to ensure it's available
from PySide6 import QtCore, QtGui, QtWidgets

# Now import matplotlib with explicit PySide6 backend
import matplotlib
matplotlib.use("Qt5Agg")  # Use Qt5 backend which is compatible with PySide6
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# PySide6 imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QComboBox, QTextEdit, QFrame, QSlider, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QMetaObject, Q_ARG
from PySide6.QtGui import QPalette, QColor, QFont, QIcon

# Import project modules
from bpm_core import BPMAnalyzer
from bpm_visuals import plot_deviation_heatmap, plot_bpm_timeseries, plot_distributions
from plot_config import apply_plot_style
from audio_processing import load_audio_file, process_audio_segment
from playback import AudioPlayer
from mic_recording import MicRecorder
from charting import ChartManager
from utils import format_time, validate_audio_file

# Apply custom plot styling
apply_plot_style()


from PySide6.QtCore import Signal

class BPMGUIApp(QMainWindow):
    """
    BPM Analyzer GUI application using PySide6, maintaining original Tkinter structure
    """
    
    # Define signals for thread-safe UI updates
    bpm_result_ready = Signal(float, str)
    error_occurred = Signal(str)
    
    def __init__(self):
        """
        Initialize the BPM Analyzer GUI application
        """
        super().__init__()
        
        # Set window properties - match original Tkinter window
        self.setWindowTitle("Advanced BPM Analyzer")
        self.setGeometry(100, 100, 900, 800)
        self.setMinimumSize(800, 600)
        
        # Configure macOS style for buttons while maintaining original structure
        self._setup_style()
        
        # Connect signals to slots
        self.bpm_result_ready.connect(self.update_bpm_result)
        self.error_occurred.connect(self.show_error)
        
        # BPM Analyzer instance
        self.analyzer = BPMAnalyzer()
        
        # Initialize managers
        self.audio_player = AudioPlayer()
        self.mic_recorder = MicRecorder(self.analyzer)
        self.chart_manager = ChartManager()
        
        # Variables to store application state - same as original
        self.audio_file = None
        self.analyzing = False
        self.time_bpm_pairs = []
        self.ref_audio_duration = 0.0
        self.mic_audio_duration = 0.0
        self.audio_duration = 0.0
        
        # Microphone monitoring state
        self.mic_bpm_history = []
        self.mic_sample_rate = 44100
        self.mic_chunk_size = 1024
        self.mic_bpm = 0
        
        # BPM comparison state
        self.comparison_active = False
        self.reference_bpm = 0
        self.reference_file = None
        self.comparison_results = []
        
        # Create widgets - maintain original structure
        self._create_widgets()
        
        # Set up window close handler
        self.closeEvent = self.on_closing
    
    def _setup_style(self):
        """
        Configure application style - only optimize buttons for macOS, maintain original structure
        """
        # Set basic macOS color scheme
        mac_colors = {
            "background": "#f5f5f7",
            "foreground": "#1d1d1f",
            "secondary_text": "#6e6e73",
            "accent": "#007aff",
            "border": "#d2d2d7",
            "button_bg": "#ffffff",
            "button_hover": "#f2f2f7",
            "button_pressed": "#ececf1"
        }
        
        # Get current palette
        palette = self.palette()
        
        # Set global colors
        palette.setColor(QPalette.Window, QColor(mac_colors["background"]))
        palette.setColor(QPalette.WindowText, QColor(mac_colors["foreground"]))
        palette.setColor(QPalette.Text, QColor(mac_colors["foreground"]))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(mac_colors["secondary_text"]))
        palette.setColor(QPalette.Base, QColor(mac_colors["background"]))
        palette.setColor(QPalette.AlternateBase, QColor(mac_colors["background"]))
        palette.setColor(QPalette.Highlight, QColor(mac_colors["accent"]))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        
        # Apply palette
        self.setPalette(palette)
        
        # Set font - use system available font
        font = QFont("Helvetica", 13)
        self.setFont(font)
    
    def _create_widgets(self):
        """
        Create all widgets for the application UI - maintain original structure
        """
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Top file selection section
        file_section = QWidget()
        file_layout = QHBoxLayout(file_section)
        file_layout.setSpacing(5)
        
        # Audio file label and input
        file_label = QLabel("Audio File:")
        file_layout.addWidget(file_label)
        
        self.file_path_display = QLabel("No file selected")
        self.file_path_display.setFixedHeight(30)
        self.file_path_display.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px; padding: 5px;")
        # Enable drag and drop for the file path display
        self.file_path_display.setAcceptDrops(True)
        # Set drag enter event handler
        self.file_path_display.dragEnterEvent = self.on_drag_enter
        # Set drag move event handler
        self.file_path_display.dragMoveEvent = self.on_drag_move
        # Set drop event handler
        self.file_path_display.dropEvent = self.on_drop
        file_layout.addWidget(self.file_path_display, 1)
        
        # Browse button - macOS style
        browse_button = QPushButton("Browse")
        browse_button.setFixedHeight(30)
        browse_button.clicked.connect(self.browse_file)
        browse_button.setToolTip("Select an audio file to analyze")
        browse_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                padding: 5px 15px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        file_layout.addWidget(browse_button)
        
        # Segment length dropdown
        self.segment_selector = QComboBox()
        self.segment_selector.setFixedHeight(30)
        # Add interval options (in seconds)
        self.segment_selector.addItems(["1", "2", "3", "4", "5", "6", "8", "10"])
        # Default to 3 seconds
        self.segment_selector.setCurrentText("3")
        self.segment_selector.setToolTip("Set time interval for BPM calculation")
        self.segment_selector.setStyleSheet("""
            QComboBox {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                padding: 5px 15px 5px 8px;
                font-size: 13px;
                color: #1d1d1f;
                width: 40px;
                min-width: 40px;
                max-width: 40px;
            }
            QComboBox:hover {
                background-color: #c7c7cc;
            }
            QComboBox:pressed {
                background-color: #b8b8bd;
            }
            QComboBox::drop-down {
                border: none;
                width: 12px;
                subcontrol-origin: padding;
                subcontrol-position: right center;
            }
            QComboBox QAbstractItemView {
                background-color: #e5e5ea;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                selection-background-color: #d2d2d7;
                selection-color: #1d1d1f;
                font-size: 13px;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 5px 10px;
            }
        """)
        file_layout.addWidget(self.segment_selector)
        
        # Calculate BPM button - macOS style with accent color
        calculate_button = QPushButton("Calculate BPM")
        calculate_button.setFixedHeight(30)
        calculate_button.clicked.connect(self.analyze_bpm)
        calculate_button.setToolTip("Calculate BPM for the selected audio file")
        calculate_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                padding: 5px 15px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        file_layout.addWidget(calculate_button)
        
        main_layout.addWidget(file_section)
        
        # Reference BPM Variation Chart
        ref_chart_section = QGroupBox("Reference BPM Variation Chart")
        ref_chart_layout = QVBoxLayout(ref_chart_section)
        ref_chart_layout.setSpacing(5)
        
        # Progress bar for BPM calculation - inside reference chart section
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #e5e5ea;
                border: none;
                border-radius: 8px;
                text-align: center;
                color: #1d1d1f;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #007aff;
                border-radius: 8px;
            }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(400)
        ref_chart_layout.addWidget(self.progress_bar)
        ref_chart_layout.addSpacing(5)
        
        # Create matplotlib figure and canvas for reference BPM chart
        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFixedHeight(200)
        self.canvas.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        ref_chart_layout.addWidget(self.canvas)
        
        # Initialize chart
        self.ax.clear()
        self.ax.set_title("BPM Variation Over Time", pad=10)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("BPM")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout(rect=[0, 0.12, 1, 0.95])
        self.canvas.draw()
        
        # Chart controls
        ref_controls = QWidget()
        ref_controls_layout = QHBoxLayout(ref_controls)
        ref_controls_layout.setSpacing(5)
        
        # Play button - macOS style
        self.ref_play_button = QPushButton("▶")
        self.ref_play_button.setFixedSize(40, 30)
        self.ref_play_button.clicked.connect(self.toggle_ref_playback)
        self.ref_play_button.setToolTip("Play/Pause reference BPM playback")
        self.ref_play_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        ref_controls_layout.addWidget(self.ref_play_button)
        
        # Reset button - macOS style
        self.ref_reset_button = QPushButton("⟲")
        self.ref_reset_button.setFixedSize(40, 30)
        self.ref_reset_button.clicked.connect(self.reset_ref_playback)
        self.ref_reset_button.setToolTip("Reset reference BPM playback")
        self.ref_reset_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        ref_controls_layout.addWidget(self.ref_reset_button)
        
        # Progress bar
        self.ref_progress = QSlider(Qt.Horizontal)
        self.ref_progress.setRange(0, 100)
        self.ref_progress.valueChanged.connect(self.on_ref_progress_changed)
        ref_controls_layout.addWidget(self.ref_progress, 1)
        
        # Time label
        self.ref_time_label = QLabel("00:00 / 00:00")
        ref_controls_layout.addWidget(self.ref_time_label)
        
        # Show detailed data button - macOS style
        ref_detail_button = QPushButton("Show Detailed BPM Data")
        ref_detail_button.setToolTip("Show detailed BPM data for the reference file")
        ref_detail_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 12px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        ref_detail_button.clicked.connect(self.show_ref_detailed_data)
        ref_controls_layout.addWidget(ref_detail_button)
        
        ref_chart_layout.addWidget(ref_controls)
        
        main_layout.addWidget(ref_chart_section)
        
        # Real-time Microphone BPM Chart
        mic_chart_section = QGroupBox("Real-time Microphone BPM Chart")
        mic_chart_layout = QVBoxLayout(mic_chart_section)
        mic_chart_layout.setSpacing(5)
        
        # Create matplotlib figure and canvas for microphone BPM chart
        self.fig_mic = Figure(figsize=(8, 3), dpi=100)
        self.ax_mic = self.fig_mic.add_subplot(111)
        self.canvas_mic = FigureCanvasQTAgg(self.fig_mic)
        self.canvas_mic.setFixedHeight(200)
        self.canvas_mic.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        mic_chart_layout.addWidget(self.canvas_mic)
        
        # Initialize microphone chart
        self.ax_mic.clear()
        self.ax_mic.set_title("Real-time Microphone BPM")
        self.ax_mic.set_xlabel("Time")
        self.ax_mic.set_ylabel("BPM")
        self.ax_mic.grid(True, alpha=0.3)
        self.fig_mic.tight_layout(rect=[0, 0.12, 1, 0.92])
        self.canvas_mic.draw()
        
        # Chart controls
        mic_controls = QWidget()
        mic_controls_layout = QHBoxLayout(mic_controls)
        mic_controls_layout.setSpacing(5)
        
        # Microphone button - macOS style
        self.mic_button = QPushButton("🎤")
        self.mic_button.setFixedSize(40, 30)
        self.mic_button.clicked.connect(self.toggle_microphone)
        self.mic_button.setToolTip("Toggle microphone recording")
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        mic_controls_layout.addWidget(self.mic_button)
        
        # Play button - macOS style
        self.mic_play_button = QPushButton("▶")
        self.mic_play_button.setFixedSize(40, 30)
        self.mic_play_button.clicked.connect(self.toggle_mic_playback)
        self.mic_play_button.setToolTip("Play/Pause microphone BPM playback")
        self.mic_play_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        mic_controls_layout.addWidget(self.mic_play_button)
        
        # Reset button - macOS style
        self.mic_reset_button = QPushButton("⟲")
        self.mic_reset_button.setFixedSize(40, 30)
        self.mic_reset_button.clicked.connect(self.reset_mic_playback)
        self.mic_reset_button.setToolTip("Reset microphone BPM playback")
        self.mic_reset_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        mic_controls_layout.addWidget(self.mic_reset_button)
        
        # Progress bar
        self.mic_progress = QSlider(Qt.Horizontal)
        self.mic_progress.setRange(0, 100)
        self.mic_progress.valueChanged.connect(self.on_mic_progress_changed)
        mic_controls_layout.addWidget(self.mic_progress, 1)
        
        # Time label
        self.mic_time_label = QLabel("00:00 / 00:00")
        mic_controls_layout.addWidget(self.mic_time_label)
        
        # Show detailed data button - macOS style
        mic_detail_button = QPushButton("Show Detailed BPM Data")
        mic_detail_button.setToolTip("Show detailed BPM data for microphone recording")
        mic_detail_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 12px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        mic_detail_button.clicked.connect(self.show_mic_detailed_data)
        mic_controls_layout.addWidget(mic_detail_button)
        
        mic_chart_layout.addWidget(mic_controls)
        
        main_layout.addWidget(mic_chart_section)
        
        # Bottom compare button
        self.compare_button = QPushButton("Compare BPM")
        self.compare_button.setFixedHeight(35)
        self.compare_button.clicked.connect(self.compare_bpm)
        self.compare_button.setToolTip("Compare BPM between reference file and microphone recording")
        self.compare_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        main_layout.addWidget(self.compare_button, 0, Qt.AlignCenter)
    
    def browse_file(self):
        """
        Open file dialog to select an audio file
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_path:
            self.audio_file = file_path
            self.file_path_display.setText(file_path)
    
    def on_drag_enter(self, event):
        """
        Handle drag enter event - check if the dragged data contains files
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def on_drag_move(self, event):
        """
        Handle drag move event - allow the drag to continue
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def on_drop(self, event):
        """
        Handle drop event - get the file path from the dropped data
        """
        if event.mimeData().hasUrls():
            # Get the first dropped file URL
            file_url = event.mimeData().urls()[0]
            # Convert URL to local file path
            file_path = file_url.toLocalFile()
            
            # Check if the file is an audio file
            audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']
            if any(file_path.lower().endswith(ext) for ext in audio_extensions):
                # Set the audio file path
                self.audio_file = file_path
                self.file_path_display.setText(file_path)
                event.acceptProposedAction()
            else:
                # Show error message for non-audio files
                QMessageBox.warning(self, "Error", "Please drop an audio file (.wav, .mp3, .ogg, .flac)")
    
    def analyze_bpm(self):
        """
        Analyze BPM from the selected audio file
        """
        if not self.audio_file:
            QMessageBox.information(self, "Information", "Please select an audio file first.")
            return
        
        if self.analyzing:
            return
        
        self.analyzing = True
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.show()
        
        # Start analysis in a separate thread
        def analyze_thread():
            try:
                # Load audio file using audio_processing module
                samples, sample_rate = load_audio_file(self.audio_file)
                
                # Store audio duration
                self.ref_audio_duration = len(samples) / sample_rate
                
                # Clear previous results
                self.time_bpm_pairs = []
                
                # Get selected segment duration from dropdown
                segment_duration = float(self.segment_selector.currentText())  # seconds
                overlap_duration = segment_duration * 0.5  # 50% overlap
                
                # Process audio segments
                segments = process_audio_segment(samples, sample_rate, segment_duration, overlap_duration)
                
                # Calculate total segments
                total_segments = len(segments)
                
                # Analyze each segment
                for i, (segment_time, segment) in enumerate(segments):
                    # Analyze segment BPM
                    bpm = self.analyzer.analyze_audio_data(segment, sample_rate)
                    
                    # Add to results
                    self.time_bpm_pairs.append((segment_time, bpm))
                    
                    # Update progress bar
                    progress = int(((i + 1) / total_segments) * 100)
                    QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.QueuedConnection, Q_ARG(int, progress))
                    
                    # Small delay to allow UI updates
                    time.sleep(0.01)
                
                # Calculate average BPM
                if self.time_bpm_pairs:
                    bpm_values = [bpm for _, bpm in self.time_bpm_pairs]
                    avg_bpm = np.mean(bpm_values)
                    info = f"BPM analysis completed. Detected BPM: {avg_bpm:.1f}"
                    
                    # Emit signal to update UI in main thread
                    self.bpm_result_ready.emit(avg_bpm, info)
            except Exception as e:
                # Emit signal to show error in main thread
                self.error_occurred.emit(f"Error analyzing BPM: {str(e)}")
            finally:
                # Update flag in thread-safe manner
                self.analyzing = False
                # Reset progress bar
                QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.QueuedConnection, Q_ARG(int, 0))
        
        threading.Thread(target=analyze_thread, daemon=True).start()
    
    def update_bpm_result(self, bpm, info):
        """
        Update BPM result in main thread
        """
        # This method will be called from analyze_thread
        print(f"BPM: {bpm:.1f}")
        print(f"Info: {info}")
        
        # Create BPM chart using ChartManager
        self.chart_manager.create_bpm_chart(self.ax, self.canvas, self.fig, self.time_bpm_pairs, self.ref_audio_duration)
    
    def show_error(self, message):
        """
        Show error message in main thread
        """
        QMessageBox.critical(self, "Error", message)
    

    
    def toggle_microphone(self):
        """
        Toggle microphone recording and BPM analysis
        """
        # Check if currently playing audio - cannot record while playing
        if hasattr(self, 'mic_playing') and self.mic_playing:
            QMessageBox.information(self, "Information", "Cannot record while playing audio. Please stop playback first.")
            return
            
        if not self.mic_recorder.is_recording():
            # Start microphone recording
            self.mic_recorder.start_recording()
            self.mic_button.setStyleSheet("""
                QPushButton {
                    background-color: #ff3b30;
                    border: none;
                    border-radius: 6px;
                    padding: 5px 10px;
                    font-size: 13px;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #cc0000;
                }
                QPushButton:pressed {
                    background-color: #990000;
                }
            """)
            
            # Start processing audio in a separate thread
            def mic_process_thread():
                try:
                    while self.mic_recorder.is_recording():
                        # Get selected segment duration from dropdown
                        segment_duration = float(self.segment_selector.currentText())  # seconds
                        
                        # Process audio chunk
                        result = self.mic_recorder.process_audio_chunk(segment_duration)
                        
                        if result:
                            current_time, bpm = result
                            # Debug: print BPM data
                            print(f"BPM: {bpm:.1f} at {current_time:.2f}s")
                            print(f"Total BPM points: {len(self.mic_recorder.mic_time_bpm_pairs)}")
                            
                            # Update chart
                            self._update_mic_bpm_chart()
                        
                        # Small delay to reduce CPU usage
                        time.sleep(0.01)
                except Exception as e:
                    print(f"Error in microphone thread: {e}")
                finally:
                    # Final chart update
                    self._update_mic_bpm_chart()
            
            threading.Thread(target=mic_process_thread, daemon=True).start()
        else:
            # Stop microphone recording
            self.mic_recorder.stop_recording()
            self.mic_button.setStyleSheet("""
                QPushButton {
                    background-color: #e5e5ea;
                    border: none;
                    border-radius: 6px;
                    padding: 5px 10px;
                    font-size: 13px;
                    color: #1d1d1f;
                }
                QPushButton:hover {
                    background-color: #d2d2d7;
                }
                QPushButton:pressed {
                    background-color: #c7c7cc;
                }
            """)
            
            # Final chart update
            self._update_mic_bpm_chart()
    
    def _update_mic_bpm_chart(self):
        """
        Update the microphone BPM chart
        """
        # Get BPM data from mic recorder
        mic_time_bpm_pairs = self.mic_recorder.get_bpm_data()
        
        # Update the mic_time_bpm_pairs attribute for backward compatibility
        self.mic_time_bpm_pairs = mic_time_bpm_pairs
        
        # Use ChartManager to update the chart
        self.chart_manager.update_mic_bpm_chart(self.ax_mic, self.canvas_mic, self.fig_mic, mic_time_bpm_pairs)
    
    def toggle_ref_playback(self):
        """
        Toggle playback of the reference audio
        """
        if not self.audio_file:
            QMessageBox.information(self, "Information", "Please select an audio file first.")
            return
        
        if not self.audio_player.playing:
            # Start playback
            self.ref_play_button.setText("⏸")
            
            # Load audio if not already loaded
            if not hasattr(self.audio_player, 'audio_duration') or self.audio_player.audio_duration == 0:
                self.audio_player.load_audio(self.audio_file)
            
            # Check if we have a saved playback position
            if self.audio_player.playback_position > 0:
                # Start playback from saved position
                self.audio_player.play(start_position=self.audio_player.playback_position)
            else:
                # Start from beginning
                self.audio_player.play()
            
            # Start update timer
            self.update_timer = QTimer(self)
            self.update_timer.timeout.connect(self._update_playback_progress)
            self.update_timer.start(100)
        else:
            # Pause playback
            self.ref_play_button.setText("▶")
            self.audio_player.pause()
            
            # Stop timer
            if hasattr(self, 'update_timer'):
                self.update_timer.stop()
    
    def _update_playback_progress(self):
        """
        Update playback progress bar and time display
        """
        if not self.audio_player.playing:
            return
        
        # Get current playback position from AudioPlayer
        current_time = self.audio_player.get_current_position()
        
        # Update progress bar
        if hasattr(self, 'ref_audio_duration') and self.ref_audio_duration > 0:
            progress = min(100, (current_time / self.ref_audio_duration) * 100)
            self.ref_progress.setValue(int(progress))
            
            # Update time display using format_time from utils
            current_str = format_time(current_time)
            duration_str = format_time(self.ref_audio_duration)
            self.ref_time_label.setText(f"{current_str} / {duration_str}")
            
            # Update chart indicator using ChartManager
            self.chart_manager.update_chart_indicator(self.ax, self.canvas, self.time_bpm_pairs, current_time, self)
        
        # Check if playback has ended
        if not self.audio_player.is_playing():
            self.audio_player.playing = False
            self.ref_play_button.setText("▶")
            if hasattr(self, 'update_timer'):
                self.update_timer.stop()
            # Remove indicator when playback ends using ChartManager
            self.chart_manager.clear_chart_indicator(self.ax, self.canvas, self)
    

    
    def reset_ref_playback(self):
        """
        Reset reference audio playback to beginning
        """
        # Stop playback if running
        self.audio_player.stop()
        self.ref_play_button.setText("▶")
        
        # Stop timer
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        
        # Reset progress
        self.ref_progress.setValue(0)
        
        # Reset time display using format_time from utils
        duration_str = format_time(self.ref_audio_duration)
        self.ref_time_label.setText(f"00:00 / {duration_str}")
        
        # Remove any real-time indicators from chart using ChartManager
        self.chart_manager.clear_chart_indicator(self.ax, self.canvas, self)
        self.canvas.draw()
    
    def on_ref_progress_changed(self, value):
        """
        Handle progress bar value changes (seeking)
        """
        if not hasattr(self, 'ref_audio_duration') or self.ref_audio_duration <= 0:
            return
        
        # Convert slider value (0-100) to seconds
        new_position = (value / 100) * self.ref_audio_duration
        
        # Update playback position in AudioPlayer
        if self.audio_player.playing:
            try:
                self.audio_player.set_position(new_position)
            except Exception as e:
                print(f"Error seeking: {e}")
        
        # Update time display using format_time from utils
        current_str = format_time(new_position)
        duration_str = format_time(self.ref_audio_duration)
        self.ref_time_label.setText(f"{current_str} / {duration_str}")
        
        # Update chart indicator using ChartManager
        self.chart_manager.update_chart_indicator(self.ax, self.canvas, self.time_bpm_pairs, new_position, self)
    
    def show_ref_detailed_data(self):
        """
        Show detailed BPM data for reference audio in a table with MAC style and export functionality
        """
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            QMessageBox.information(self, "Information", "No BPM data available. Please analyze an audio file first.")
            return
        
        # Create a new window to display detailed data
        from PySide6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView, QHBoxLayout, QPushButton, QFileDialog
        from PySide6.QtGui import QColor, QFont
        import openpyxl
        from openpyxl.styles import Font, Alignment, Border, Side
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Detailed BPM Data")
        dialog.setGeometry(200, 200, 900, 600)
        dialog.setStyleSheet("background-color: #f5f5f7;")
        
        # Main layout with padding
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create table widget with MAC style
        table = QTableWidget()
        table.setRowCount(len(self.time_bpm_pairs))
        table.setColumnCount(2)
        
        # Set column headers
        table.setHorizontalHeaderLabels(["Time (seconds)", "BPM"])
        
        # Configure table appearance
        table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                gridline-color: #e5e5ea;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTableWidget::item:selected {
                background-color: #e6f2ff;
                color: #007aff;
            }
            QHeaderView::section {
                background-color: #f5f5f7;
                border: none;
                border-bottom: 1px solid #d2d2d7;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
                color: #1d1d1f;
            }
            QTableCornerButton::section {
                background-color: #f5f5f7;
                border: none;
                border-bottom: 1px solid #d2d2d7;
                border-right: 1px solid #d2d2d7;
            }
        """)
        
        # Set row height for better readability
        table.verticalHeader().setDefaultSectionSize(35)
        
        # Fill table with data
        for row, (time_seconds, bpm) in enumerate(self.time_bpm_pairs):
            # Time column
            time_item = QTableWidgetItem(f"{time_seconds:.2f}")
            time_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 0, time_item)
            
            # BPM column
            bpm_item = QTableWidgetItem(f"{bpm:.1f}")
            bpm_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 1, bpm_item)
        
        # Set column widths to fit content
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Add average BPM row
        table.insertRow(len(self.time_bpm_pairs))
        
        # Average time label (empty)
        avg_label_item = QTableWidgetItem("Average")
        avg_label_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        avg_label_item.setForeground(QColor("#007aff"))
        avg_label_item.setFont(QFont("Helvetica", 13, QFont.Bold))
        table.setItem(len(self.time_bpm_pairs), 0, avg_label_item)
        
        # Average BPM value
        avg_bpm = np.mean([bpm for _, bpm in self.time_bpm_pairs])
        avg_bpm_item = QTableWidgetItem(f"{avg_bpm:.1f}")
        avg_bpm_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        avg_bpm_item.setForeground(QColor("#007aff"))
        avg_bpm_item.setFont(QFont("Helvetica", 13, QFont.Bold))
        table.setItem(len(self.time_bpm_pairs), 1, avg_bpm_item)
        
        # Function to export data to Excel
        def export_to_excel():
            # Get save path from user
            file_path, _ = QFileDialog.getSaveFileName(
                dialog, "Export to Excel", 
                f"bpm_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", 
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Create a new workbook
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "BPM Data"
            
            # Set column headers
            headers = ["Time (seconds)", "BPM"]
            for col, header in enumerate(headers, 1):
                cell = sheet.cell(row=1, column=col)
                cell.value = header
                cell.font = Font(bold=True, size=12)
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Fill sheet with data
            for row, (time_seconds, bpm) in enumerate(self.time_bpm_pairs, 2):
                sheet.cell(row=row, column=1, value=time_seconds).alignment = Alignment(horizontal="right", vertical="center")
                sheet.cell(row=row, column=2, value=bpm).alignment = Alignment(horizontal="right", vertical="center")
            
            # Add average BPM row
            avg_row = len(self.time_bpm_pairs) + 2
            sheet.cell(row=avg_row, column=1, value="Average").font = Font(bold=True, color="007AFF")
            sheet.cell(row=avg_row, column=1).alignment = Alignment(horizontal="right", vertical="center")
            sheet.cell(row=avg_row, column=2, value=avg_bpm).font = Font(bold=True, color="007AFF")
            sheet.cell(row=avg_row, column=2).alignment = Alignment(horizontal="right", vertical="center")
            
            # Auto-adjust column widths
            for col in sheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 4, 20)
                sheet.column_dimensions[column].width = adjusted_width
            
            # Save the workbook
            workbook.save(file_path)
            
            # Show success message
            QMessageBox.information(dialog, "Success", f"Data exported successfully to {file_path}")
        
        # Create bottom layout with export button
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        # Export button with MAC style
        export_button = QPushButton("Export to Excel")
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
            QPushButton:pressed {
                background-color: #0052a3;
            }
        """)
        export_button.clicked.connect(export_to_excel)
        
        bottom_layout.addWidget(export_button)
        
        # Add widgets to main layout
        main_layout.addWidget(table)
        main_layout.addLayout(bottom_layout)
        
        dialog.exec()
    
    def show_mic_detailed_data(self):
        """
        Show detailed BPM data for microphone recording in a table with MAC style and export functionality
        """
        if not hasattr(self, 'mic_time_bpm_pairs') or not self.mic_time_bpm_pairs:
            QMessageBox.information(self, "Information", "No microphone BPM data available. Please start recording first.")
            return
        
        # Create a new window to display detailed data
        from PySide6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView, QHBoxLayout, QPushButton, QFileDialog
        from PySide6.QtGui import QColor, QFont
        import openpyxl
        from openpyxl.styles import Font, Alignment, Border, Side
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Detailed Microphone BPM Data")
        dialog.setGeometry(200, 200, 900, 600)
        dialog.setStyleSheet("background-color: #f5f5f7;")
        
        # Main layout with padding
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create table widget with MAC style
        table = QTableWidget()
        table.setRowCount(len(self.mic_time_bpm_pairs))
        table.setColumnCount(2)
        
        # Set column headers
        table.setHorizontalHeaderLabels(["Time (seconds)", "BPM"])
        
        # Configure table appearance
        table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                gridline-color: #e5e5ea;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTableWidget::item:selected {
                background-color: #e6f2ff;
                color: #007aff;
            }
            QHeaderView::section {
                background-color: #f5f5f7;
                border: none;
                border-bottom: 1px solid #d2d2d7;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
                color: #1d1d1f;
            }
            QTableCornerButton::section {
                background-color: #f5f5f7;
                border: none;
                border-bottom: 1px solid #d2d2d7;
                border-right: 1px solid #d2d2d7;
            }
        """)
        
        # Set row height for better readability
        table.verticalHeader().setDefaultSectionSize(35)
        
        # Fill table with data
        for row, (time_seconds, bpm) in enumerate(self.mic_time_bpm_pairs):
            # Time column
            time_item = QTableWidgetItem(f"{time_seconds:.2f}")
            time_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 0, time_item)
            
            # BPM column
            bpm_item = QTableWidgetItem(f"{bpm:.1f}")
            bpm_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 1, bpm_item)
        
        # Set column widths to fit content
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Add average BPM row
        table.insertRow(len(self.mic_time_bpm_pairs))
        
        # Average time label (empty)
        avg_label_item = QTableWidgetItem("Average")
        avg_label_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        avg_label_item.setForeground(QColor("#007aff"))
        avg_label_item.setFont(QFont("Helvetica", 13, QFont.Bold))
        table.setItem(len(self.mic_time_bpm_pairs), 0, avg_label_item)
        
        # Average BPM value
        avg_bpm = np.mean([bpm for _, bpm in self.mic_time_bpm_pairs])
        avg_bpm_item = QTableWidgetItem(f"{avg_bpm:.1f}")
        avg_bpm_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        avg_bpm_item.setForeground(QColor("#007aff"))
        avg_bpm_item.setFont(QFont("Helvetica", 13, QFont.Bold))
        table.setItem(len(self.mic_time_bpm_pairs), 1, avg_bpm_item)
        
        # Function to export data to Excel
        def export_to_excel():
            # Get save path from user
            file_path, _ = QFileDialog.getSaveFileName(
                dialog, "Export to Excel", 
                f"mic_bpm_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", 
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Create a new workbook
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Microphone BPM Data"
            
            # Set column headers
            headers = ["Time (seconds)", "BPM"]
            for col, header in enumerate(headers, 1):
                cell = sheet.cell(row=1, column=col)
                cell.value = header
                cell.font = Font(bold=True, size=12)
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Fill sheet with data
            for row, (time_seconds, bpm) in enumerate(self.mic_time_bpm_pairs, 2):
                sheet.cell(row=row, column=1, value=time_seconds).alignment = Alignment(horizontal="right", vertical="center")
                sheet.cell(row=row, column=2, value=bpm).alignment = Alignment(horizontal="right", vertical="center")
            
            # Add average BPM row
            avg_row = len(self.mic_time_bpm_pairs) + 2
            sheet.cell(row=avg_row, column=1, value="Average").font = Font(bold=True, color="007AFF")
            sheet.cell(row=avg_row, column=1).alignment = Alignment(horizontal="right", vertical="center")
            sheet.cell(row=avg_row, column=2, value=avg_bpm).font = Font(bold=True, color="007AFF")
            sheet.cell(row=avg_row, column=2).alignment = Alignment(horizontal="right", vertical="center")
            
            # Auto-adjust column widths
            for col in sheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 4, 20)
                sheet.column_dimensions[column].width = adjusted_width
            
            # Save the workbook
            workbook.save(file_path)
            
            # Show success message
            QMessageBox.information(dialog, "Success", f"Data exported successfully to {file_path}")
        
        # Create bottom layout with export button
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        # Export button with MAC style
        export_button = QPushButton("Export to Excel")
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
            QPushButton:pressed {
                background-color: #0052a3;
            }
        """)
        export_button.clicked.connect(export_to_excel)
        
        bottom_layout.addWidget(export_button)
        
        # Add widgets to main layout
        main_layout.addWidget(table)
        main_layout.addLayout(bottom_layout)
        
        dialog.exec()
    
    def toggle_mic_playback(self):
        """
        Toggle playback of microphone recorded audio
        """
        # Check if currently recording - cannot play while recording
        if hasattr(self, 'mic_recording') and self.mic_recording:
            QMessageBox.information(self, "Information", "Cannot play audio while recording. Please stop recording first.")
            return
            
        if not hasattr(self, 'mic_time_bpm_pairs') or not self.mic_time_bpm_pairs:
            QMessageBox.information(self, "Information", "No microphone BPM data available. Please start recording first.")
            return
        
        if not hasattr(self, 'mic_playing'):
            self.mic_playing = False
        
        # Always convert the latest microphone data to WAV for playback
        try:
            self._convert_mic_to_wav_for_playback()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preparing microphone audio for playback: {str(e)}")
            return
        
        if not self.mic_playing:
            # Start playback
            self.mic_playing = True
            self.mic_play_button.setText("⏸")
            
            # Initialize playback parameters if not already set
            if not hasattr(self, 'mic_playback_position') or self.mic_playback_position < 0:
                self.mic_playback_position = 0
            
            # Start playback
            pygame.mixer.music.load(self.temp_mic_wav_file)
            if self.mic_playback_position > 0:
                # Start playback from saved position
                pygame.mixer.music.play(start=self.mic_playback_position)
                self.mic_last_update_time = time.time() - self.mic_playback_position
            else:
                # Start playback from beginning
                pygame.mixer.music.play()
                self.mic_last_update_time = time.time()
            
            # Start timer for updating progress
            self.mic_update_timer = QTimer(self)
            self.mic_update_timer.timeout.connect(self._update_mic_playback_progress)
            self.mic_update_timer.start(100)
        else:
            # Pause playback
            self.mic_playing = False
            self.mic_play_button.setText("▶")
            
            # Save current playback position
            self.mic_playback_position = time.time() - self.mic_last_update_time
            
            # Pause the audio
            pygame.mixer.music.pause()
            
            # Stop timer
            if hasattr(self, 'mic_update_timer'):
                self.mic_update_timer.stop()
    
    def _convert_mic_to_wav_for_playback(self):
        """
        Convert microphone recorded data to WAV file for playback
        """
        # Get microphone data from MicRecorder instance
        audio_data = self.mic_recorder.get_audio_data()
        
        if len(audio_data) == 0:
            raise ValueError("No microphone data available for playback")
        
        import wave
        
        # Create temporary WAV file
        from datetime import datetime
        self.temp_mic_wav_file = f"temp_mic_playback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        # Calculate actual audio duration in seconds
        self.mic_audio_duration = len(audio_data) / self.mic_sample_rate
        
        # Normalize audio data to int16 range (-32768 to 32767)
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
        # Save as WAV file
        with wave.open(self.temp_mic_wav_file, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes per sample (int16)
            wf.setframerate(self.mic_sample_rate)
            wf.writeframes(audio_data.tobytes())
    
    def _update_mic_playback_progress(self):
        """
        Update microphone playback progress bar and time display
        """
        if not self.mic_playing:
            return
        
        # Calculate current playback position
        current_time = time.time() - self.mic_last_update_time
        self.mic_playback_position = current_time
        
        # Update progress bar
        if self.mic_audio_duration > 0:
            progress = min(100, (current_time / self.mic_audio_duration) * 100)
            self.mic_progress.setValue(int(progress))
            
            # Update time display
            current_str = format_time(current_time)
            duration_str = format_time(self.mic_audio_duration)
            self.mic_time_label.setText(f"{current_str} / {duration_str}")
        
        # Update chart indicator
        self._update_mic_chart_indicator(current_time)
        
        # Check if playback has ended (either by duration or pygame indicating it's done)
        if current_time >= self.mic_audio_duration or not pygame.mixer.music.get_busy():
            self.reset_mic_playback()
    
    def reset_mic_playback(self):
        """
        Reset microphone playback to beginning
        """
        # Stop playback if running
        self.mic_playing = False
        self.mic_play_button.setText("▶")
        
        # Stop audio playback
        pygame.mixer.music.stop()
        
        # Stop timer
        if hasattr(self, 'mic_update_timer'):
            self.mic_update_timer.stop()
        
        # Reset progress
        self.mic_progress.setValue(0)
        
        # Reset time display
        duration_str = format_time(self.mic_audio_duration)
        self.mic_time_label.setText(f"00:00 / {duration_str}")
        
        # Reset playback position
        self.mic_playback_position = 0
        
        # Remove any real-time indicators from chart
        if hasattr(self, '_mic_current_bpm_line'):
            try:
                self._mic_current_bpm_line.remove()
                if hasattr(self, '_mic_current_bpm_text'):
                    self._mic_current_bpm_text.remove()
                delattr(self, '_mic_current_bpm_line')
                delattr(self, '_mic_current_bpm_text')
            except:
                pass
        self.canvas_mic.draw()
    
    def on_mic_progress_changed(self, value):
        """
        Handle microphone progress bar value changes (seeking)
        """
        if not hasattr(self, 'mic_audio_duration') or self.mic_audio_duration <= 0:
            return
        
        # Convert slider value (0-100) to seconds
        new_position = (value / 100) * self.mic_audio_duration
        
        # Update playback position
        self.mic_playback_position = new_position
        
        # Update time display
        current_str = format_time(new_position)
        duration_str = format_time(self.mic_audio_duration)
        self.mic_time_label.setText(f"{current_str} / {duration_str}")
        
        # Update last_update_time to match new position if playing
        if hasattr(self, 'mic_playing') and self.mic_playing:
            self.mic_last_update_time = time.time() - new_position
        
        # Update chart indicator
        self._update_mic_chart_indicator(new_position)
    
    def _update_mic_chart_indicator(self, current_time):
        """
        Update the real-time BPM indicator on the microphone chart
        """
        if not hasattr(self, 'mic_time_bpm_pairs') or not self.mic_time_bpm_pairs:
            return
        
        # Find current BPM based on time
        current_bpm = None
        for time_seconds, bpm in self.mic_time_bpm_pairs:
            if time_seconds > current_time:
                break
            current_bpm = bpm
        
        if current_bpm is None:
            return
        
        # Remove previous indicator if exists
        if hasattr(self, '_mic_current_bpm_line'):
            try:
                self._mic_current_bpm_line.remove()
                if hasattr(self, '_mic_current_bpm_text'):
                    self._mic_current_bpm_text.remove()
            except:
                pass
        
        # Draw vertical line at current position
        self._mic_current_bpm_line = self.ax_mic.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
        
        # Display current BPM value
        y_min, y_max = self.ax_mic.get_ylim()
        y_pos = y_min + (y_max - y_min) * 0.9  # Position near top of chart
        self._mic_current_bpm_text = self.ax_mic.text(current_time, y_pos, f"Current BPM: {current_bpm:.1f}", 
                                                   color='red', fontsize=10, ha='center', va='top',
                                                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        
        # Redraw canvas
        self.canvas_mic.draw()
    
    def compare_bpm(self):
        """
        Compare BPM with reference value using original design with tabs
        """
        if not hasattr(self, 'time_bpm_pairs') or not self.time_bpm_pairs:
            QMessageBox.information(self, "Information", "No reference BPM data available. Please analyze an audio file first.")
            return
        
        if not hasattr(self, 'mic_time_bpm_pairs') or not self.mic_time_bpm_pairs:
            QMessageBox.information(self, "Information", "No microphone BPM data available. Please record some audio first.")
            return
        
        # Import necessary modules for the dialog
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, QPushButton
        
        # Create a new dialog for detailed comparison
        dialog = QDialog(self)
        dialog.setWindowTitle("BPM Comparison Analysis Report")
        # Set dialog size to match Advanced BPMAnalyzer main window
        dialog.setGeometry(100, 100, 900, 800)
        # Set fixed size to ensure dimensions are respected
        dialog.setFixedSize(900, 800)
        dialog.setStyleSheet("background-color: white;")
        
        # Main layout
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Calculate average BPM for both
        ref_avg = np.mean([bpm for _, bpm in self.time_bpm_pairs])
        mic_avg = np.mean([bpm for _, bpm in self.mic_time_bpm_pairs])
        
        # Calculate difference
        difference = mic_avg - ref_avg
        percentage_diff = (difference / ref_avg) * 100 if ref_avg > 0 else 0
        
        # Calculate advanced metrics
        # Align data by time
        from scipy.interpolate import interp1d
        ref_times, ref_bpms = zip(*self.time_bpm_pairs)
        mic_times, mic_bpms = zip(*self.mic_time_bpm_pairs)
        
        # Create interpolation function for reference BPM
        ref_interp = interp1d(ref_times, ref_bpms, kind='linear', fill_value='extrapolate')
        
        # Interpolate reference BPM at microphone time points
        ref_bpms_interp = ref_interp(mic_times)
        
        # Calculate differences
        differences = np.array(mic_bpms) - np.array(ref_bpms_interp)
        
        # Calculate rhythm stability (standard deviation of differences)
        rhythm_stability = np.std(differences)
        
        # Calculate timing consistency
        within_5_percent = len([d for d in differences if abs(d/ref_avg * 100) <= 5]) / len(differences) * 100
        within_10_percent = len([d for d in differences if abs(d/ref_avg * 100) <= 10]) / len(differences) * 100
        within_15_percent = len([d for d in differences if abs(d/ref_avg * 100) <= 15]) / len(differences) * 100
        
        # Evaluate performance
        speed_accuracy = "Excellent" if abs(percentage_diff) <= 2 else "Good" if abs(percentage_diff) <= 5 else "Poor"
        rhythm_stability_eval = "Excellent" if rhythm_stability <= 2 else "Good" if rhythm_stability <= 5 else "Poor"
        timing_consistency_eval = "Excellent" if within_10_percent >= 90 else "Good" if within_10_percent >= 70 else "Poor"
        expression_style_eval = "Excellent" if abs(percentage_diff) <= 5 and rhythm_stability <= 3 else "Good" if abs(percentage_diff) <= 10 else "Poor"
        
        # Create tab widget (only once)
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # --- Summary & Metrics Tab --- #
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(20, 20, 20, 20)
        
        # Top-right Export button - keep this at the top
        export_layout = QHBoxLayout()
        export_layout.setContentsMargins(0, 0, 0, 15)
        export_layout.addStretch()
        
        # Export button
        export_button = QPushButton("Export")
        export_button.setFixedSize(120, 35)
        export_button.setToolTip("Export Summary Metrics and Visual Comparison to PDF")
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                color: #1d1d1f;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        export_layout.addWidget(export_button)
        
        # Add Export button layout to summary layout
        summary_layout.addLayout(export_layout)
        
        # Control bar for model dropdown and Generate button - to be moved to AI Feedback area
        ai_control_bar = QHBoxLayout()
        ai_control_bar.setContentsMargins(0, 0, 0, 15)
        ai_control_bar.addStretch()
        
        # Model selection dropdown - matches Generate button background
        model_dropdown = QComboBox()
        model_dropdown.setFixedSize(180, 35)
        model_dropdown.addItem("deepseek-v3")
        model_dropdown.addItem("deepseek-r1")
        model_dropdown.setStyleSheet("""
            QComboBox {
                background-color: #d2d2d7;
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                font-size: 14px;
                padding: 0 15px;
                color: #1d1d1f;
            }
            QComboBox:hover {
                background-color: #c7c7cc;
                border-color: #c7c7cc;
            }
            QComboBox:focus {
                border-color: #007aff;
                outline: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QComboBox::drop-down {
                width: 25px;
                border-left: 1px solid #b8b8bd;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background-color: rgba(255, 255, 255, 0.2);
            }
            QComboBox::drop-down:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        ai_control_bar.addWidget(model_dropdown)
        
        # Add spacing between dropdown and button
        ai_control_bar.addSpacing(15)
        
        # Generate button
        generate_button = QPushButton("Generate")
        generate_button.setFixedSize(120, 35)
        generate_button.setToolTip("Generate AI summary using DeepSeek")
        generate_button.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                color: #1d1d1f;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #c7c7cc;
            }
            QPushButton:pressed {
                background-color: #b8b8bd;
            }
        """)
        ai_control_bar.addWidget(generate_button)
        
        # Create a summary text widget to display DeepSeek feedback
        from PySide6.QtWidgets import QTextEdit, QSplitter, QScrollArea
        
        # Create top section for current content
        top_summary_widget = QWidget()
        top_summary_layout = QVBoxLayout(top_summary_widget)
        
        # Title
        title_label = QLabel("BPM COMPARISON ANALYSIS REPORT")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 20px;")
        top_summary_layout.addWidget(title_label)
        
        # Create bottom section for AI feedback
        feedback_widget = QWidget()
        feedback_layout = QVBoxLayout(feedback_widget)
        
        feedback_title = QLabel("AI Feedback (DeepSeek)")
        feedback_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        feedback_layout.addWidget(feedback_title)
        
        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        self.feedback_text.setMinimumHeight(300)  # Set minimum height
        self.feedback_text.setStyleSheet("font-size: 13px; background-color: #f5f5f7;")
        self.feedback_text.setAcceptRichText(True)  # Enable HTML rendering
        feedback_layout.addWidget(self.feedback_text)
        
        # Create scroll area for top summary content to ensure it's fully visible
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(top_summary_widget)
        
        # Create a container for AI feedback and controls
        ai_container = QWidget()
        ai_layout = QVBoxLayout(ai_container)
        
        # Create horizontal layout to align feedback title with controls
        title_control_layout = QHBoxLayout()
        title_control_layout.setContentsMargins(0, 0, 0, 10)
        
        # Add feedback title on the left
        title_control_layout.addWidget(feedback_title)
        
        # Add stretch to push controls to the right
        title_control_layout.addStretch()
        
        # Add DeepSeek controls on the right, aligned with the title
        title_control_layout.addLayout(ai_control_bar)
        
        # Remove margins from ai_control_bar to avoid extra spacing
        ai_control_bar.setContentsMargins(0, 0, 0, 0)
        
        # Add the combined title+controls layout
        ai_layout.addLayout(title_control_layout)
        
        # Add AI feedback text edit
        ai_layout.addWidget(self.feedback_text)
        
        # Create splitter to divide the summary into two sections
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(scroll_area)
        splitter.addWidget(ai_container)
        
        # Set initial sizes for splitter sections
        splitter.setSizes([400, 400])
        
        # Add splitter to summary layout
        summary_layout.addWidget(splitter)
        
        # Implement the _generate_deepseek_summary function
        def _generate_deepseek_summary():
            """
            Call DeepSeek chat API using current comparison data and append the
            returned feedback to the Summary & Metrics tab. Runs in a background thread.
            """
            def build_prompt():
                # Prefer recorded microphone series if available
                if hasattr(self, 'recorded_mic_bpm_data') and self.recorded_mic_bpm_data:
                    pairs = self.recorded_mic_bpm_data
                else:
                    pairs = self.mic_time_bpm_pairs
                max_items = 60
                pairs_str = ", ".join([f"{round(t,1)}s:{round(b,1)}" for t, b in pairs[:max_items]]) if pairs else "(no data)"
                instruction = (
                    f"Reference BPM (Score): {ref_avg:.1f}\n"
                    f"Recorded BPM time-series (time:value, up to {max_items} items): {pairs_str}\n"
                    f"Recorded BPM stats — mean: {mic_avg:.1f}, difference vs reference: {difference:.1f} ({percentage_diff:.1f}%).\n"
                    "Please compare the recorded BPM against the reference BPM, provide an evaluation of the performance, and suggest improvements.\n"
                    "Respond in English, concise and structured with these sections: Overall Evaluation, Issues Observed, Actionable Improvement Suggestions."
                )
                return instruction
            
            def request_thread():
                try:
                    instruction = build_prompt()
                    
                    # Try multiple environment variable names
                    api_key = None
                    possible_env_names = ['DEEPSEEK_API_KEY', 'deepseek_api_key', 'DEEPSEEK-API-KEY', 'deepseek-api-key']
                    
                    for env_name in possible_env_names:
                        temp_key = os.environ.get(env_name)
                        print(f"Debug: Checking env var '{env_name}': {'Found' if temp_key else 'Not found'}")
                        if temp_key:
                            api_key = temp_key
                            print(f"Debug: Using API key from env var: '{env_name}'")
                            break
                    
                    # Fallback to project config.json with multiple possible keys
                    if not api_key:
                        try:
                            cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
                            print(f"Debug: Checking config.json at: {cfg_path}")
                            if os.path.exists(cfg_path):
                                with open(cfg_path, "r", encoding="utf-8") as f:
                                    cfg = json.load(f)
                                
                                possible_config_keys = ['DEEPSEEK_API_KEY', 'deepseek_api_key', 'api_key', 'api-key', 'DEEPSEEK-API-KEY', 'deepseek-api-key']
                                for config_key in possible_config_keys:
                                    temp_key = cfg.get(config_key)
                                    print(f"Debug: Checking config key '{config_key}': {'Found' if temp_key else 'Not found'}")
                                    if temp_key:
                                        api_key = temp_key
                                        print(f"Debug: Using API key from config: '{config_key}'")
                                        break
                                
                                if not api_key:
                                    print(f"Debug: No API key found in config.json. Available keys: {list(cfg.keys())}")
                            else:
                                print(f"Debug: config.json not found at: {cfg_path}")
                        except Exception as e:
                            print(f"Debug: Error reading config.json: {str(e)}")
                            api_key = None
                    
                    # Try another approach: check current directory
                    if not api_key:
                        try:
                            cfg_path = "config.json"
                            print(f"Debug: Checking config.json in current dir: {cfg_path}")
                            if os.path.exists(cfg_path):
                                with open(cfg_path, "r", encoding="utf-8") as f:
                                    cfg = json.load(f)
                                
                                possible_config_keys = ['DEEPSEEK_API_KEY', 'deepseek_api_key', 'api_key', 'api-key', 'DEEPSEEK-API-KEY', 'deepseek-api-key']
                                for config_key in possible_config_keys:
                                    temp_key = cfg.get(config_key)
                                    print(f"Debug: Checking current dir config key '{config_key}': {'Found' if temp_key else 'Not found'}")
                                    if temp_key:
                                        api_key = temp_key
                                        print(f"Debug: Using API key from current dir config: '{config_key}'")
                                        break
                        except Exception as e:
                            print(f"Debug: Error reading current dir config.json: {str(e)}")
                            api_key = None
                    
                    # Check if API key is just an empty string
                    if api_key == "":
                        print(f"Debug: API key found but is empty string")
                        api_key = None
                    
                    # Try one more approach: direct assignment for testing
                    # Uncomment the line below and replace with your actual API key for testing
                    # api_key = "your-actual-api-key-here"
                    
                    if not api_key:
                        # Use QMetaObject.invokeMethod to show message box on main thread
                        error_msg = "Missing DEEPSEEK_API_KEY. Set environment variable or add to config.json."
                        QMetaObject.invokeMethod(
                            thread_helper,
                            "show_error_message",
                            Qt.QueuedConnection,
                            Q_ARG(str, error_msg)
                        )
                        return
                    
                    # Debug: API key found, continue with request
                    print("Debug: API key found, continuing with request")
                    
                    url = "https://api.deepseek.com/v1/chat/completions"
                    
                    def _map_model(name):
                        n = (name or "").strip().lower()
                        if n in ("deepseek-v3", "deepseek_chat", "deepseek-chat", "v3"):
                            return "deepseek-chat"
                        if n in ("deepseek-r1", "r1", "deepseek-reasoner", "deepseek-reasoner"):
                            return "deepseek-reasoner"
                        return "deepseek-chat"
                    
                    payload = {
                        "model": _map_model(model_dropdown.currentText()),
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
                    except ImportError:
                        # Fallback to default SSL context if certifi is not available
                        ssl_context = ssl.create_default_context()
                    
                    with urllib.request.urlopen(req, timeout=180, context=ssl_context) as resp:
                        body = resp.read().decode('utf-8')
                    
                    result = json.loads(body)
                    msg = result.get('choices', [{}])[0].get('message', {})
                    content = msg.get('content', '')
                    reasoning = msg.get('reasoning', '')
                    if reasoning:
                        content = f"### Reasoning\n{reasoning}\n\n### Answer\n{content}" if content else f"### Reasoning\n{reasoning}"
                    if not content:
                        content = "(No content returned)"
                    
                    # Update the feedback text widget in the main thread
                    QMetaObject.invokeMethod(
                        thread_helper,
                        "update_feedback_text",
                        Qt.QueuedConnection,
                        Q_ARG(str, content)
                    )
                    
                except Exception as e:
                    # Use QMetaObject.invokeMethod to show message box on main thread
                    error_msg = f"Error calling DeepSeek API: {str(e)}"
                    QMetaObject.invokeMethod(
                        thread_helper,
                        "show_error_message",
                        Qt.QueuedConnection,
                        Q_ARG(str, error_msg)
                    )
            
            # Start the request in a background thread
            threading.Thread(target=request_thread, daemon=True).start()
        
        # Import QObject and Slot for the helper class
        from PySide6.QtCore import QObject, Slot
        
        # Create a helper object with slots that can be called from background threads
        class ThreadHelper(QObject):
            """Helper class with slots that can be invoked from background threads"""
            def __init__(self, parent_dialog, feedback_text):
                super().__init__()
                self.parent_dialog = parent_dialog
                self.feedback_text = feedback_text
            
            @Slot(str)
            def show_error_message(self, message):
                """Slot to show error message from background thread"""
                QMessageBox.critical(self.parent_dialog, "DeepSeek Error", message)
            
            @Slot(str)
            def update_feedback_text(self, content):
                """Slot to update feedback text from background thread, converting markdown to HTML"""
                # Convert markdown to HTML
                from markdown import markdown
                html_content = markdown(content)
                self.feedback_text.setHtml(html_content)
        
        # Create the helper object with reference to the feedback text widget
        thread_helper = ThreadHelper(dialog, self.feedback_text)
        
        # Connect Generate button to the function
        generate_button.clicked.connect(_generate_deepseek_summary)
        
        # Comparison Metrics section
        metrics_label = QLabel("COMPARISON METRICS:")
        metrics_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a1a1a; margin-top: 10px;")
        top_summary_layout.addWidget(metrics_label)
        
        metrics_text = f"Reference BPM (Score): {ref_avg:.1f}\n"
        metrics_text += f"Your Average BPM: {mic_avg:.1f}\n"
        metrics_text += f"BPM Difference: {difference:+.1f} ({percentage_diff:+.1f}%)\n"
        metrics_label_content = QLabel(metrics_text)
        metrics_label_content.setStyleSheet("font-size: 13px; margin-left: 15px; margin-bottom: 15px;")
        top_summary_layout.addWidget(metrics_label_content)
        
        # Detailed Evaluation section
        eval_label = QLabel("DETAILED EVALUATION:")
        eval_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a1a1a; margin-top: 10px;")
        top_summary_layout.addWidget(eval_label)
        
        eval_text = "<ul>"
        eval_text += f"<li><span style='color: {'green' if speed_accuracy in ['Excellent', 'Good'] else 'red'};'>{'✓' if speed_accuracy in ['Excellent', 'Good'] else '✗'}</span> <strong>Speed Accuracy</strong>: {speed_accuracy} - {'Your performance speed is very close to the score' if speed_accuracy == 'Excellent' else 'Your performance speed is close to the score' if speed_accuracy == 'Good' else 'Your performance speed differs significantly from the score'}</li>"
        eval_text += f"<li><span style='color: {'green' if rhythm_stability_eval in ['Excellent', 'Good'] else 'red'};'>{'✓' if rhythm_stability_eval in ['Excellent', 'Good'] else '✗'}</span> <strong>Rhythm Stability</strong>: {rhythm_stability_eval} - {'Your rhythm is consistent and stable' if rhythm_stability_eval == 'Excellent' else 'Your rhythm has minor fluctuations' if rhythm_stability_eval == 'Good' else 'Your rhythm has significant fluctuations'}</li>"
        eval_text += f"<li><span style='color: {'green' if timing_consistency_eval in ['Excellent', 'Good'] else 'red'};'>{'✓' if timing_consistency_eval in ['Excellent', 'Good'] else '✗'}</span> <strong>Timing Consistency</strong>: {timing_consistency_eval} - {'Your timing is very consistent' if timing_consistency_eval == 'Excellent' else 'Your timing is mostly consistent' if timing_consistency_eval == 'Good' else 'Your timing has noticeable inconsistencies'}</li>"
        eval_text += f"<li><span style='color: {'green' if expression_style_eval in ['Excellent', 'Good'] else 'red'};'>{'✓' if expression_style_eval in ['Excellent', 'Good'] else '✗'}</span> <strong>Expression & Style</strong>: {expression_style_eval} - {'Your performance demonstrates professional musical expression' if expression_style_eval == 'Excellent' else 'Your performance shows good musical expression' if expression_style_eval == 'Good' else 'Your performance lacks musical expression'}</li>"
        eval_text += "</ul>"
        
        eval_label_content = QLabel(eval_text)
        eval_label_content.setStyleSheet("font-size: 13px; margin-left: 15px; margin-bottom: 15px;")
        eval_label_content.setTextFormat(Qt.RichText)
        top_summary_layout.addWidget(eval_label_content)
        
        # Improvement Suggestions section
        improve_label = QLabel("IMPROVEMENT SUGGESTIONS:")
        improve_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a1a1a; margin-top: 10px;")
        top_summary_layout.addWidget(improve_label)
        
        suggestions = [
            "Practice with a metronome to improve your sense of rhythm",
            "Record yourself regularly to identify rhythmic inconsistencies",
            "Focus on maintaining a steady tempo throughout your performance",
            "Start slowly and gradually increase speed as you gain consistency"
        ]
        
        suggestions_text = "<ul>"
        for suggestion in suggestions:
            suggestions_text += f"<li>{suggestion}</li>"
        suggestions_text += "</ul>"
        
        suggestions_label = QLabel(suggestions_text)
        suggestions_label.setStyleSheet("font-size: 13px; margin-left: 15px; margin-bottom: 15px;")
        suggestions_label.setTextFormat(Qt.RichText)
        top_summary_layout.addWidget(suggestions_label)
        
        # Advanced Performance Metrics section
        advanced_label = QLabel("ADVANCED PERFORMANCE METRICS:")
        advanced_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a1a1a; margin-top: 10px;")
        top_summary_layout.addWidget(advanced_label)
        
        advanced_text = "<ul>"
        advanced_text += f"<li><strong>Rhythm Stability</strong>: Stability Score: {rhythm_stability:.2f}/10</li>"
        advanced_text += f"<li><strong>Timing Consistency</strong>:</li>"
        advanced_text += f"<ul style='margin-left: 20px;'>"
        advanced_text += f"<li>Within ±5% of reference: {within_5_percent:.1f}%</li>"
        advanced_text += f"<li>Within ±10% of reference: {within_10_percent:.1f}%</li>"
        advanced_text += f"<li>Within ±15% of reference: {within_15_percent:.1f}%</li>"
        advanced_text += "</ul></li>"
        advanced_text += f"<li><strong>Tempo Progression</strong>: Performance Trend: {'Consistent' if rhythm_stability <= 3 else 'Slight Fluctuations' if rhythm_stability <= 6 else 'Significant Variations'}</li>"
        advanced_text += "</ul>"
        
        advanced_label_content = QLabel(advanced_text)
        advanced_label_content.setStyleSheet("font-size: 13px; margin-left: 15px; margin-bottom: 15px;")
        advanced_label_content.setTextFormat(Qt.RichText)
        top_summary_layout.addWidget(advanced_label_content)
        
        # Interpretation section
        interpretation_label = QLabel("INTERPRETATION:")
        interpretation_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a1a1a; margin-top: 10px;")
        top_summary_layout.addWidget(interpretation_label)
        
        interpretation_text = "<ul>"
        if rhythm_stability <= 3:
            interpretation_text += "<li>Your rhythm is very consistent with minimal fluctuations.</li>"
        elif rhythm_stability <= 6:
            interpretation_text += "<li>Your rhythm has noticeable fluctuations that should be addressed.</li>"
        else:
            interpretation_text += "<li>Your rhythm has significant variations that require improvement.</li>"
        
        if within_10_percent >= 80:
            interpretation_text += "<li>Your timing is mostly consistent with occasional variations.</li>"
        else:
            interpretation_text += "<li>Your timing needs improvement to achieve more consistent performance.</li>"
        interpretation_text += "</ul>"
        
        interpretation_label_content = QLabel(interpretation_text)
        interpretation_label_content.setStyleSheet("font-size: 13px; margin-left: 15px; margin-bottom: 15px;")
        interpretation_label_content.setTextFormat(Qt.RichText)
        top_summary_layout.addWidget(interpretation_label_content)
        
        # Spacer to push content up
        top_summary_layout.addStretch()
        
        # Add summary tab
        tab_widget.addTab(summary_tab, "Summary & Metrics")
        
        # --- Visual Comparison Tab --- #
        visual_tab = QWidget()
        visual_layout = QVBoxLayout(visual_tab)
        visual_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create a scroll area for better navigation
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)  # Set to False so scrollbars appear when content is larger than viewport
        scroll_area.setFixedHeight(600)  # Set a fixed height to ensure scrollbars appear
        scroll_area.setStyleSheet("QScrollArea { border: none; } QScrollBar:vertical { width: 15px; } QScrollBar::handle:vertical { background-color: #c1c1c1; border-radius: 7px; }")
        
        # Create a container widget for the scroll area
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(40)  # Increase spacing between charts to 40 pixels
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # Import matplotlib components
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from bpm_visuals import plot_bpm_timeseries, plot_distributions, plot_deviation_heatmap
        
        # --- Plot 1: Real-time Microphone BPM vs Reference BPM (Time Series) --- #
        fig1 = Figure(figsize=(8, 4), dpi=100)  # Reduced size from 12x6 to 8x4
        ax1 = fig1.add_subplot(111)
        plot_bpm_timeseries(ax1, mic_times, mic_bpms, ref_avg, reference_pairs=self.time_bpm_pairs)
        
        # Unify font sizes
        ax1.set_title(ax1.get_title(), fontsize=10, fontweight='bold', pad=6)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(8)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=8)
        ax1.set_ylabel(ax1.get_ylabel(), fontsize=8)
        
        fig1.tight_layout(pad=3.0)  # Add padding around the figure
        canvas1 = FigureCanvasQTAgg(fig1)
        canvas1.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        scroll_layout.addWidget(canvas1)
        
        # --- Plot 2: BPM Distribution (Mic vs Reference) --- #
        fig2 = Figure(figsize=(8, 3), dpi=100)  # Reduced size from 12x5 to 8x3
        ax2 = fig2.add_subplot(111)
        
        # Plot violin plot for BPM distribution comparison
        violin_parts = ax2.violinplot([mic_bpms, ref_bpms], positions=[1, 2], showmeans=True, showmedians=True)
        
        # Customize violin plot
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor('#2E86AB' if i == 0 else '#A23B72')
            pc.set_alpha(0.6)
        
        if 'cbars' in violin_parts:
            violin_parts['cbars'].set_color('#333333')
        if 'cmins' in violin_parts:
            violin_parts['cmins'].set_color('#333333')
        if 'cmaxes' in violin_parts:
            violin_parts['cmaxes'].set_color('#333333')
        if 'cmeans' in violin_parts:
            violin_parts['cmeans'].set_color('#F18F01')
        if 'cmedians' in violin_parts:
            violin_parts['cmedians'].set_color('#333333')
        
        # Configure violin plot
        ax2.set_title('BPM Distribution (Mic vs Reference)', fontsize=10, fontweight='bold', pad=6)
        ax2.set_ylabel('BPM', fontsize=8)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['Mic', 'Reference'], fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Add mean and median labels
        mean_mic = float(np.mean(mic_bpms)) if len(mic_bpms) > 0 else float('nan')
        median_mic = float(np.median(mic_bpms)) if len(mic_bpms) > 0 else float('nan')
        mean_ref = float(np.mean(ref_bpms)) if len(ref_bpms) > 0 else float('nan')
        median_ref = float(np.median(ref_bpms)) if len(ref_bpms) > 0 else float('nan')
        
        if np.isfinite(mean_mic) and np.isfinite(median_mic):
            ax2.text(1, mean_mic, f'Mean: {mean_mic:.1f}', color='#F18F01', fontsize=7, ha='center', va='bottom')
            ax2.text(1, median_mic, f'Median: {median_mic:.1f}', color='#333333', fontsize=7, ha='center', va='top')
        
        if np.isfinite(mean_ref) and np.isfinite(median_ref):
            ax2.text(2, mean_ref, f'Mean: {mean_ref:.1f}', color='#F18F01', fontsize=7, ha='center', va='bottom')
            ax2.text(2, median_ref, f'Median: {median_ref:.1f}', color='#333333', fontsize=7, ha='center', va='top')
        
        # Unify tick font sizes
        for label in ax2.get_yticklabels():
            label.set_fontsize(8)
        
        fig2.tight_layout(pad=3.0)
        canvas2 = FigureCanvasQTAgg(fig2)
        canvas2.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        scroll_layout.addWidget(canvas2)
        
        # --- Plot 3: Deviation Distribution (Mic vs Reference) --- #
        fig3 = Figure(figsize=(8, 3), dpi=100)  # Reduced size from 12x5 to 8x3
        ax3 = fig3.add_subplot(111)
        
        # Calculate deviations
        deviations = np.array(mic_bpms) - np.array(ref_bpms_interp)
        
        # Create box plot of deviations
        bp = ax3.boxplot([deviations], positions=[1], patch_artist=True, widths=0.6)
        
        # Customize box plot
        for box in bp['boxes']:
            box.set_facecolor('#F18F01')
            box.set_alpha(0.6)
        
        for median in bp['medians']:
            median.set_color('#A23B72')
            median.set_linewidth(2)
        
        # Add scatter plot for individual deviations
        jitter = np.random.normal(0, 0.05, len(deviations))
        ax3.scatter(np.ones_like(deviations) + jitter, deviations, alpha=0.3, color='#2E86AB', s=10)
        
        # Add reference lines
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(np.mean(deviations), color='#A23B72', linestyle='--', linewidth=2, label=f'Mean diff: {np.mean(deviations):.2f}')
        
        # Configure deviation plot
        ax3.set_title('Deviation Distribution (Mic vs Reference)', fontsize=10, fontweight='bold', pad=6)
        ax3.set_ylabel('BPM difference', fontsize=8)
        ax3.set_xticks([1])
        ax3.set_xticklabels(['Diff'], fontsize=8)
        ax3.legend(fontsize=8, loc='upper right')
        
        # Unify tick font sizes
        for label in ax3.get_yticklabels():
            label.set_fontsize(8)
        ax3.grid(True, alpha=0.3)
        
        fig3.tight_layout(pad=3.0)
        canvas3 = FigureCanvasQTAgg(fig3)
        canvas3.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        scroll_layout.addWidget(canvas3)
        
        # --- Plot 4: Tempo Deviation Heatmap (%) --- #
        fig4 = Figure(figsize=(8, 3), dpi=100)  # Reduced size from 12x5 to 8x3
        ax4 = fig4.add_subplot(111)
        plot_deviation_heatmap(ax4, mic_times, mic_bpms, ref_bpms_interp, ref_avg, segment_count=8)
        
        # Unify font sizes for heatmap
        ax4.set_title(ax4.get_title(), fontsize=10, fontweight='bold', pad=6)
        for label in ax4.get_xticklabels() + ax4.get_yticklabels():
            label.set_fontsize(8)
        if ax4.get_xlabel():
            ax4.set_xlabel(ax4.get_xlabel(), fontsize=8)
        if ax4.get_ylabel():
            ax4.set_ylabel(ax4.get_ylabel(), fontsize=8)
        
        fig4.tight_layout(pad=3.0)
        canvas4 = FigureCanvasQTAgg(fig4)
        canvas4.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        scroll_layout.addWidget(canvas4)
        
        # --- Plot 5: Reference Heatmap --- #
        fig5 = Figure(figsize=(8, 3), dpi=100)  # Reduced size from 12x5 to 8x3
        ax5 = fig5.add_subplot(111)
        
        # Create data for reference heatmap - compare to reference average instead of itself
        # This will show deviation from the average BPM, not from itself
        ref_avg_series = np.array([ref_avg] * len(ref_bpms))
        plot_deviation_heatmap(ax5, ref_times, ref_bpms, ref_avg_series, ref_avg, segment_count=8)
        ax5.set_title('Reference Heatmap (%)', fontsize=10, fontweight='bold', pad=6)
        
        # Unify font sizes for reference heatmap
        for label in ax5.get_xticklabels() + ax5.get_yticklabels():
            label.set_fontsize(8)
        if ax5.get_xlabel():
            ax5.set_xlabel(ax5.get_xlabel(), fontsize=8)
        if ax5.get_ylabel():
            ax5.set_ylabel(ax5.get_ylabel(), fontsize=8)
        
        fig5.tight_layout(pad=3.0)
        canvas5 = FigureCanvasQTAgg(fig5)
        canvas5.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        scroll_layout.addWidget(canvas5)
        
        # --- Plot 6: Mic Heatmap --- #
        fig6 = Figure(figsize=(8, 3), dpi=100)  # Reduced size from 12x5 to 8x3
        ax6 = fig6.add_subplot(111)
        
        # Create data for mic heatmap
        mic_avg_bpm = np.mean(mic_bpms)
        mic_ref_series = np.array([mic_avg_bpm] * len(mic_bpms))
        plot_deviation_heatmap(ax6, mic_times, mic_bpms, mic_ref_series, mic_avg_bpm, segment_count=8)
        ax6.set_title('Mic Heatmap (%)', fontsize=10, fontweight='bold', pad=6)
        
        # Unify font sizes for mic heatmap
        for label in ax6.get_xticklabels() + ax6.get_yticklabels():
            label.set_fontsize(8)
        if ax6.get_xlabel():
            ax6.set_xlabel(ax6.get_xlabel(), fontsize=8)
        if ax6.get_ylabel():
            ax6.set_ylabel(ax6.get_ylabel(), fontsize=8)
        
        fig6.tight_layout(pad=3.0)
        canvas6 = FigureCanvasQTAgg(fig6)
        canvas6.setStyleSheet("background-color: white; border: 1px solid #d2d2d7; border-radius: 6px;")
        scroll_layout.addWidget(canvas6)
        
        # Add spacer to the end
        scroll_layout.addStretch()
        
        # Set up the scroll area
        scroll_area.setWidget(scroll_widget)
        visual_layout.addWidget(scroll_area)
        
        # Function to export to PDF
        def export_to_pdf():
            from PySide6.QtWidgets import QFileDialog
            import tempfile
            import os
            import re
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_pdf import PdfPages
            
            # Get save path from user
            file_path, _ = QFileDialog.getSaveFileName(
                dialog, "Export PDF", 
                f"bpm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", 
                "PDF Files (*.pdf);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Helper function to clean HTML tags
            def clean_html(html):
                # Process HTML to ensure proper line breaks for lists
                # Replace list items with bullet points and newlines
                html = html.replace('<li>', '\n• ')
                html = html.replace('</li>', '')
                
                # Remove list container tags
                html = html.replace('<ul>', '')
                html = html.replace('</ul>', '')
                html = html.replace('<ol>', '')
                html = html.replace('</ol>', '')
                
                # Remove any remaining HTML tags
                clean_text = re.sub(r'<[^>]+>', '', html)
                
                # Convert Windows-style newlines to Unix-style
                clean_text = clean_text.replace('\r\n', '\n')
                
                # Preserve spaces within lines but ensure proper line breaks
                lines = clean_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Remove leading/trailing spaces and collapse multiple spaces within the line
                    cleaned_line = ' '.join(line.strip().split())
                    if cleaned_line:
                        cleaned_lines.append(cleaned_line)
                
                # Join lines with single newlines
                clean_text = '\n'.join(cleaned_lines)
                return clean_text
            
            # Create temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save Summary Metrics text
                summary_text = f"BPM COMPARISON ANALYSIS REPORT\n\n"
                
                # Comparison Metrics
                summary_text += f"COMPARISON METRICS:\n"
                summary_text += metrics_text + "\n\n"
                
                # Detailed Evaluation
                summary_text += f"DETAILED EVALUATION:\n"
                clean_eval = clean_html(eval_text)
                # Format evaluation points
                summary_text += clean_eval + "\n\n"
                
                # Improvement Suggestions
                summary_text += f"IMPROVEMENT SUGGESTIONS:\n"
                clean_suggestions = clean_html(suggestions_text)
                # Format suggestions as bullet points
                suggestions_list = clean_suggestions.split('\n')
                for suggestion in suggestions_list:
                    if suggestion.strip():
                        summary_text += f"- {suggestion.strip()}\n"
                summary_text += "\n"
                
                # Advanced Performance Metrics
                summary_text += f"ADVANCED PERFORMANCE METRICS:\n"
                clean_advanced = clean_html(advanced_text)
                # Format advanced metrics
                advanced_lines = clean_advanced.split('\n')
                for line in advanced_lines:
                    if line.strip():
                        summary_text += f"{line.strip()}\n"
                summary_text += "\n"
                
                # Interpretation
                summary_text += f"INTERPRETATION:\n"
                clean_interpretation = clean_html(interpretation_text)
                # Format interpretation as bullet points
                interpretation_list = clean_interpretation.split('\n')
                for interpretation in interpretation_list:
                    if interpretation.strip():
                        summary_text += f"- {interpretation.strip()}\n"
                summary_text += "\n"
                
                # AI Feedback (DeepSeek)
                if hasattr(self, 'feedback_text'):
                    ai_feedback = self.feedback_text.toPlainText()
                    if ai_feedback and ai_feedback.strip() != "":
                        # Process markdown to plain text for PDF
                        import re
                        from markdown import markdown
                        from bs4 import BeautifulSoup
                        
                        # Convert markdown to HTML first
                        html_content = markdown(ai_feedback)
                        
                        # Use BeautifulSoup to extract plain text with proper formatting
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Process headings, lists, etc.
                        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                            h.string = f"\n{h.text.strip()}:\n"
                        
                        for ul in soup.find_all('ul'):
                            for li in ul.find_all('li'):
                                li.string = f"• {li.text.strip()}\n"
                        
                        for ol in soup.find_all('ol'):
                            for i, li in enumerate(ol.find_all('li')):
                                li.string = f"{i+1}. {li.text.strip()}\n"
                        
                        # Extract plain text
                        plain_text = soup.get_text()
                        
                        # Clean up extra newlines and spaces
                        plain_text = re.sub(r'\n+', '\n', plain_text.strip())
                        
                        summary_text += f"AI FEEDBACK (DEEPSEEK):\n"
                        summary_text += plain_text + "\n\n"
                
                # Save all charts
                chart_paths = []
                for i, canvas in enumerate([canvas1, canvas2, canvas3, canvas4, canvas5, canvas6]):
                    chart_path = os.path.join(temp_dir, f"chart_{i+1}.png")
                    canvas.figure.savefig(chart_path, dpi=300, bbox_inches='tight')
                    chart_paths.append(chart_path)
                
                # Create PDF using matplotlib
                with PdfPages(file_path) as pdf:
                    # Set consistent width of 11 inches for all pages
                    consistent_width = 11
                    
                    # Use legal size (11x14) for summary page to fit all content on one page
                    # and letter size (11x8.5) for charts, both with same width
                    fig_summary = Figure(figsize=(consistent_width, 14), dpi=300)  # Legal size for summary
                    ax_summary = fig_summary.add_subplot(111)
                    ax_summary.axis('off')
                    
                    # Add summary text with proper formatting
                    ax_summary.text(0.1, 0.97, "BPM COMPARISON ANALYSIS REPORT", 
                                  fontsize=16, fontweight='bold', ha='left', va='top')
                    
                    # Import textwrap for automatic line wrapping
                    import textwrap
                    
                    # Maximum characters per line before wrapping
                    max_line_length = 85
                    
                    # Process lines with automatic wrapping
                    original_lines = summary_text.strip().split('\n')
                    processed_lines = []
                    
                    for line in original_lines:
                        if not line.strip():
                            # Keep empty lines
                            processed_lines.append('')
                        elif line.strip().endswith(':') and len(line.strip()) > 10:
                            # Keep section headers as-is
                            processed_lines.append(line)
                        elif line.strip().startswith('•'):
                            # Wrap bullet points with indentation
                            bullet_content = line[1:].strip()
                            wrapped = textwrap.wrap(bullet_content, width=max_line_length-4)
                            if wrapped:
                                processed_lines.append(f'• {wrapped[0]}')
                                for wrapped_line in wrapped[1:]:
                                    processed_lines.append(f'  {wrapped_line}')
                        elif re.match(r'^\d+\.', line.strip()):
                            # Wrap numbered lists with indentation
                            parts = line.split('. ', 1)
                            if len(parts) == 2:
                                number, content = parts
                                wrapped = textwrap.wrap(content, width=max_line_length-len(number)-2)
                                if wrapped:
                                    processed_lines.append(f'{number}. {wrapped[0]}')
                                    for wrapped_line in wrapped[1:]:
                                        processed_lines.append(f'  {wrapped_line}')
                        else:
                            # Wrap regular text
                            wrapped = textwrap.wrap(line, width=max_line_length)
                            processed_lines.extend(wrapped)
                    
                    # Add processed lines to the figure with proper spacing
                    y_pos = 0.92
                    line_height = 0.023  # Optimal line height to fit content
                    
                    for line in processed_lines:
                        font_size = 11  # Slightly smaller font to fit more content
                        font_weight = 'normal'
                        
                        # Apply bold to section headers
                        if line.strip().endswith(':') and len(line.strip()) > 10:
                            font_weight = 'bold'
                            font_size = 14  # Adjusted header size
                            y_pos -= 0.005  # Slight extra space before headers
                        
                        # Determine x position based on line type
                        if line.startswith('  '):
                            x_pos = 0.15  # Indented wrapped lines
                        elif line.startswith('•') or (line.strip() and line[0].isdigit() and '.' in line.split()[0]):
                            x_pos = 0.13  # Bullet points and numbered lists
                        else:
                            x_pos = 0.1   # Regular text and headers
                        
                        # Add text with wrapping enabled
                        ax_summary.text(x_pos, y_pos, line, 
                                      fontsize=font_size, 
                                      fontweight=font_weight, 
                                      ha='left', 
                                      va='top',
                                      wrap=True)
                        y_pos -= line_height
                    
                    # Save summary page with tight layout - consistent width maintained by figure size
                    pdf.savefig(fig_summary, bbox_inches='tight')
                    
                    # Subsequent pages: Visual Comparison charts (same width, standard height)
                    for i, chart_path in enumerate(chart_paths):
                        fig_chart = Figure(figsize=(consistent_width, 8.5), dpi=300)  # Same width, standard height
                        ax_chart = fig_chart.add_subplot(111)
                        ax_chart.axis('off')
                        
                        # Add chart image with proper scaling
                        import matplotlib.image as mpimg
                        img = mpimg.imread(chart_path)
                        ax_chart.imshow(img, aspect='auto', extent=[0.1, 0.9, 0.1, 0.9])
                        
                        # Save chart page with tight layout - same width maintained by figure size
                        pdf.savefig(fig_chart, bbox_inches='tight')
                
            QMessageBox.information(dialog, "Export Successful", f"PDF exported successfully to {file_path}")
        
        # Connect Export button
        export_button.clicked.connect(export_to_pdf)
        
        # Add visual tab
        tab_widget.addTab(visual_tab, "Visual Comparison")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(20, 10, 20, 20)
        bottom_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.setFixedSize(80, 30)
        close_button.setToolTip("Close the comparison window")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #e5e5ea;
                border: none;
                border-radius: 6px;
                padding: 5px 15px;
                font-size: 13px;
                color: #1d1d1f;
            }
            QPushButton:hover {
                background-color: #d2d2d7;
            }
            QPushButton:pressed {
                background-color: #c7c7cc;
            }
        """)
        close_button.clicked.connect(dialog.close)
        bottom_layout.addWidget(close_button)
        
        main_layout.addLayout(bottom_layout)
        
        # Show dialog
        dialog.exec()
    
    def on_closing(self, event):
        """
        Handle window close event
        """
        # Stop any ongoing processes
        self.analyzing = False
        
        # Stop timer
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        
        # Stop any ongoing recording and clean up resources
        if self.mic_recorder.is_recording():
            self.mic_recorder.stop_recording()
        self.mic_recorder.cleanup()
        
        # Stop any ongoing playback and clean up resources
        self.audio_player.cleanup()
        
        # Accept the close event
        event.accept()


# Main function to run the application
if __name__ == "__main__":
    # Import QApplication here to ensure it's available
    from PySide6.QtWidgets import QApplication
    
    # Create the application instance
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = BPMGUIApp()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())
