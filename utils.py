#!/usr/bin/env python3
"""
Utility module for BPM Analyzer
Contains common utility functions
"""


def format_time(seconds):
    """
    Format seconds into MM:SS format
    
    Parameters:
        seconds: Time in seconds
        
    Returns:
        Formatted time string in MM:SS format
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def validate_audio_file(file_path):
    """
    Validate if the file is a supported audio file
    
    Parameters:
        file_path: Path to the file
        
    Returns:
        bool: True if valid audio file, False otherwise
    """
    import os
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']
    return any(file_path.lower().endswith(ext) for ext in audio_extensions)


def safe_remove_file(file_path):
    """
    Safely remove a file, ignoring errors if the file doesn't exist
    
    Parameters:
        file_path: Path to the file to remove
    """
    import os
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")


def calculate_overlap_samples(sample_rate, segment_duration, overlap_percentage=50):
    """
    Calculate the number of overlap samples based on segment duration and overlap percentage
    
    Parameters:
        sample_rate: Audio sample rate
        segment_duration: Duration of each segment in seconds
        overlap_percentage: Overlap percentage between segments (default: 50)
        
    Returns:
        int: Number of overlap samples
    """
    return int(segment_duration * sample_rate * (overlap_percentage / 100))


def get_audio_duration(file_path):
    """
    Get the duration of an audio file
    
    Parameters:
        file_path: Path to the audio file
        
    Returns:
        float: Duration in seconds
    """
    from pydub import AudioSegment
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # Convert from milliseconds to seconds
    except Exception as e:
        raise Exception(f"Error getting audio duration: {e}")


def bpm_to_category(bpm):
    """
    Convert BPM value to music category/genre description
    
    Parameters:
        bpm: Beats per minute
        
    Returns:
        String description of the BPM category
    """
    if bpm >= 200:
        return "Extremely Fast (Electronic Hardcore)"
    elif bpm >= 175:
        return "Very Fast (Drum & Bass, Gabber)"
    elif bpm >= 150:
        return "Fast (Trance, Hardstyle)"
    elif bpm >= 130:
        return "Moderately Fast (House, Techno)"
    elif bpm >= 110:
        return "Medium (Pop, Rock, EDM)"
    elif bpm >= 90:
        return "Moderately Slow (Hip Hop, R&B)"
    elif bpm >= 70:
        return "Slow (Ballads, Reggae)"
    else:
        return "Very Slow (Ambient, Doom Metal)"
