#!/usr/bin/env python3
"""
Audio playback module for BPM Analyzer
Handles audio playback functionality
"""

import os
import pygame
import time
from audio_processing import convert_to_wav


class AudioPlayer:
    """
    Audio playback manager for BPM Analyzer
    """
    
    def __init__(self):
        """
        Initialize the audio player
        """
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Player state
        self.playing = False
        self.playback_position = 0
        self.temp_wav_file = None
        self.audio_duration = 0.0
        self.last_update_time = 0
        
    def load_audio(self, audio_file):
        """
        Load audio file for playback
        
        Parameters:
            audio_file: Path to audio file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Create temporary WAV file for playback
            self.temp_wav_file = f"temp_{os.getpid()}.wav"
            
            # Convert to WAV if needed
            wav_file = convert_to_wav(audio_file, self.temp_wav_file)
            
            # Load audio
            pygame.mixer.music.load(wav_file)
            
            # Get audio duration
            with open(wav_file, 'rb') as f:
                f.seek(22)
                bytes_per_sample = int.from_bytes(f.read(2), 'little')
                f.seek(24)
                sample_rate = int.from_bytes(f.read(4), 'little')
                f.seek(40)
                data_size = int.from_bytes(f.read(4), 'little')
                self.audio_duration = data_size / (sample_rate * bytes_per_sample)
            
            return True
            
        except Exception as e:
            raise Exception(f"Error loading audio: {e}")
    
    def play(self, start_position=0):
        """
        Start playback from specified position
        
        Parameters:
            start_position: Start position in seconds
        """
        try:
            pygame.mixer.music.play(start=start_position)
            self.playing = True
            self.last_update_time = time.time() - start_position
        except Exception as e:
            raise Exception(f"Error playing audio: {e}")
    
    def pause(self):
        """
        Pause playback
        """
        pygame.mixer.music.pause()
        self.playing = False
    
    def stop(self):
        """
        Stop playback and reset position
        """
        pygame.mixer.music.stop()
        self.playing = False
        self.playback_position = 0
        self.last_update_time = time.time()
    
    def set_position(self, position):
        """
        Set playback position
        
        Parameters:
            position: Position in seconds
        """
        try:
            pygame.mixer.music.set_pos(position)
            self.playback_position = position
            self.last_update_time = time.time() - position
        except Exception as e:
            raise Exception(f"Error setting position: {e}")
    
    def get_current_position(self):
        """
        Get current playback position
        
        Returns:
            Current position in seconds
        """
        if self.playing:
            return time.time() - self.last_update_time
        return self.playback_position
    
    def is_playing(self):
        """
        Check if audio is playing
        
        Returns:
            bool: True if playing, False otherwise
        """
        return self.playing and pygame.mixer.music.get_busy()
    
    def cleanup(self):
        """
        Cleanup resources
        """
        # Stop playback
        self.stop()
        
        # Remove temporary WAV file
        if self.temp_wav_file and os.path.exists(self.temp_wav_file):
            try:
                os.remove(self.temp_wav_file)
            except Exception:
                pass
        
        # Quit pygame mixer
        pygame.mixer.quit()
