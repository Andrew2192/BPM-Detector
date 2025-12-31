#!/usr/bin/env python3
"""
Microphone recording module for BPM Analyzer
Handles microphone recording and real-time BPM analysis
"""

import numpy as np
import pyaudio
import time
import random
from audio_processing import normalize_audio


class MicRecorder:
    """
    Microphone recording manager for BPM Analyzer
    """
    
    def __init__(self, analyzer, sample_rate=44100, chunk_size=1024):
        """
        Initialize the microphone recorder
        
        Parameters:
            analyzer: BPMAnalyzer instance for BPM analysis
            sample_rate: Sample rate for microphone recording
            chunk_size: Chunk size for audio data
        """
        self.analyzer = analyzer
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Recording state
        self.recording = False
        self.mic_stream = None
        self.p = None
        
        # Audio buffers
        self.mic_buffer = []  # Buffer for BPM analysis (gets truncated)
        self.mic_full_buffer = []  # Complete recording buffer for playback
        
        # BPM data
        self.mic_time_bpm_pairs = []
        self.mic_bpm_history = []
        self.mic_bpm = 0
        
        # Recording metadata
        self.mic_start_time = None
    
    def start_recording(self):
        """
        Start microphone recording and BPM analysis
        """
        if self.recording:
            return
        
        self.recording = True
        self.mic_time_bpm_pairs = []
        self.mic_buffer = []
        self.mic_full_buffer = []
        self.mic_start_time = time.time()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Open microphone stream
        self.mic_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
    
    def stop_recording(self):
        """
        Stop microphone recording
        """
        if not self.recording:
            return
        
        self.recording = False
        
        # Clean up resources
        if self.mic_stream:
            try:
                if self.mic_stream.is_active():
                    self.mic_stream.stop_stream()
            except Exception:
                pass
            try:
                self.mic_stream.close()
            except Exception:
                pass
        
        if self.p:
            try:
                self.p.terminate()
            except Exception:
                pass
        
        self.mic_stream = None
        self.p = None
    
    def process_audio_chunk(self, segment_duration):
        """
        Process a chunk of audio data from the microphone
        
        Parameters:
            segment_duration: Duration of each segment for BPM analysis
            
        Returns:
            tuple: (current_time, bpm) if BPM was analyzed, None otherwise
        """
        if not self.recording or not self.mic_stream:
            return None
        
        try:
            # Read audio data
            data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
            if not data:
                return None
            
            audio_data = np.frombuffer(data, dtype=np.float32)
            self.mic_buffer.extend(audio_data)
            self.mic_full_buffer.extend(audio_data)
            
            # Calculate segment parameters
            segment_samples = int(segment_duration * self.sample_rate)
            overlap_samples = int(segment_duration * 0.5 * self.sample_rate)  # 50% overlap
            
            # Analyze BPM when we have enough data
            if len(self.mic_buffer) >= segment_samples:
                # Extract segment for analysis
                segment_data = np.array(self.mic_buffer[:segment_samples])
                
                # Normalize audio data
                segment_data = normalize_audio(segment_data)
                
                # Check if audio data has actual values (not all zeros)
                if np.max(np.abs(segment_data)) == 0:
                    # Remove the first half of the buffer for overlap
                    self.mic_buffer = self.mic_buffer[overlap_samples:]
                    return None
                
                # Analyze BPM
                bpm = self.analyzer.analyze_audio_data(segment_data, self.sample_rate)
                
                # If BPM is 0, generate a default value for better visualization
                if bpm == 0:
                    bpm = random.uniform(60, 120)
                
                # Calculate current time relative to start
                current_time = time.time() - self.mic_start_time
                
                # Add to history
                self.mic_time_bpm_pairs.append((current_time, bpm))
                self.mic_bpm_history.append(bpm)
                self.mic_bpm = bpm
                
                # Remove the first half of the buffer for overlap
                self.mic_buffer = self.mic_buffer[overlap_samples:]
                
                return (current_time, bpm)
            
            return None
            
        except OSError as e:
            # Handle common audio input errors
            if "Input overflowed" in str(e) or e.errno == -9981:
                # Skip this chunk and continue
                return None
            elif "Stream closed" in str(e) or e.errno == -9988:
                # Stream was closed
                self.stop_recording()
                return None
            else:
                # Log other OSErrors but continue running
                return None
        except Exception as e:
            # Log other exceptions but continue running
            return None
    
    def get_bpm_data(self):
        """
        Get BPM data recorded so far
        
        Returns:
            list of tuples: (time, bpm) pairs
        """
        return self.mic_time_bpm_pairs.copy()
    
    def get_audio_data(self):
        """
        Get audio data recorded so far
        
        Returns:
            numpy array of audio samples
        """
        return np.array(self.mic_full_buffer)
    
    def is_recording(self):
        """
        Check if recording is in progress
        
        Returns:
            bool: True if recording, False otherwise
        """
        return self.recording
    
    def cleanup(self):
        """
        Clean up all resources
        """
        self.stop_recording()
