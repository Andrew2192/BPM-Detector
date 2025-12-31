#!/usr/bin/env python3
"""
Audio processing module for BPM Analyzer
Handles audio file loading, conversion, and processing
"""

import os
import numpy as np
from pydub import AudioSegment


def load_audio_file(file_path):
    """
    Load audio file using pydub and convert to standard format
    
    Parameters:
        file_path: Path to audio file
        
    Returns:
        tuple: (numpy array of samples, sample rate)
    """
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate to 44.1kHz for consistency
        audio = audio.set_frame_rate(44100)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Normalize to [-1, 1]
        max_val = 2 ** (audio.sample_width * 8 - 1)
        samples = samples.astype(np.float32) / max_val
        
        return samples, audio.frame_rate
        
    except Exception as e:
        raise Exception(f"Error loading audio file: {e}")


def convert_to_wav(audio_file, temp_wav_file):
    """
    Convert audio file to WAV format if needed
    
    Parameters:
        audio_file: Path to original audio file
        temp_wav_file: Path to temporary WAV file
        
    Returns:
        Path to WAV file (either original or converted)
    """
    if audio_file.lower().endswith('.wav'):
        return audio_file
    
    try:
        # Convert to WAV
        audio = AudioSegment.from_file(audio_file)
        audio.export(temp_wav_file, format="wav")
        return temp_wav_file
    except Exception as e:
        raise Exception(f"Error converting to WAV: {e}")


def process_audio_segment(audio_segment, sample_rate, segment_duration, overlap_duration):
    """
    Process audio segment for BPM analysis
    
    Parameters:
        audio_segment: numpy array of audio samples
        sample_rate: Audio sample rate
        segment_duration: Duration of each segment in seconds
        overlap_duration: Overlap duration between segments in seconds
        
    Returns:
        list of tuples: (segment_time, segment_samples)
    """
    segment_samples = int(segment_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    
    segments = []
    total_segments = max(1, int((len(audio_segment) - segment_samples) / (segment_samples - overlap_samples)) + 1)
    
    for i in range(total_segments):
        # Calculate segment start and end indices
        start_idx = i * (segment_samples - overlap_samples)
        end_idx = start_idx + segment_samples
        
        # Ensure we don't go beyond the audio
        if end_idx > len(audio_segment):
            end_idx = len(audio_segment)
            start_idx = max(0, end_idx - segment_samples)
        
        # Extract segment
        segment = audio_segment[start_idx:end_idx]
        
        # Calculate segment time in seconds
        segment_time = start_idx / sample_rate
        
        segments.append((segment_time, segment))
    
    return segments


def normalize_audio(audio_data):
    """
    Normalize audio data to [-1, 1] range
    
    Parameters:
        audio_data: numpy array of audio samples
        
    Returns:
        Normalized audio data
    """
    max_val = np.max(np.abs(audio_data))
    if max_val == 0:
        return audio_data
    return audio_data.astype(np.float32) / max_val
