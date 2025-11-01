import numpy as np
from pydub import AudioSegment
from scipy import signal

class BPMAnalyzer:
    def __init__(self, frame_size=2048, hop_size=512):
        """
        Initialize the BPM analyzer with optimized parameters
        
        Parameters:
            frame_size: Size of audio frames for processing
            hop_size: Step size between consecutive frames
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        # Threshold multiplier for beat detection sensitivity
        self.beat_threshold_multiplier = 1.3  # Increased from 1.2 for better detection
        # Apply smoothing to BPM values
        self.bpm_smoothing_window = 3  # Moving average window size
        # Spectral flux threshold for better beat detection
        self.spectral_flux_threshold = 0.15
    
    def analyze_audio_data(self, audio_data, sample_rate):
        """
        Analyze audio data to detect beats and calculate BPM
        
        Parameters:
            audio_data: 1D numpy array of audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Detected BPM value
        """
        # Detect beats using improved algorithm
        beats = self._detect_beats_improved(audio_data, sample_rate)
        
        if not beats:
            return 0
            
        # Calculate BPM candidates from beat intervals
        bpm_candidates = self._calculate_bpm_candidates(beats, sample_rate)
        
        # Apply IQR to filter outliers
        filtered_bpm = self._filter_outliers_iqr(bpm_candidates)
        
        if not filtered_bpm:
            return np.median(bpm_candidates) if bpm_candidates else 0
            
        # Apply moving average smoothing
        if len(filtered_bpm) >= self.bpm_smoothing_window:
            # Calculate moving average for smoothed BPM
            smoothed_bpm = self._moving_average(filtered_bpm, self.bpm_smoothing_window)
            return np.mean(smoothed_bpm)
        
        return np.mean(filtered_bpm)
    
    def _detect_beats_improved(self, audio_data, sample_rate):
        """
        Improved beat detection using spectral flux, energy, and onset detection
        
        Parameters:
            audio_data: 1D numpy array of audio samples
            sample_rate: Audio sample rate
            
        Returns:
            List of beat timestamps in seconds
        """
        # Normalize audio data
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Calculate energy envelope (root mean square per frame)
        energy = []
        for i in range(0, len(audio_data) - self.frame_size, self.hop_size):
            frame = audio_data[i:i + self.frame_size]
            energy.append(np.sqrt(np.mean(frame**2)))
        
        # Calculate spectral flux (change in frequency domain)
        spectral_flux = self._calculate_spectral_flux(audio_data, sample_rate)
        
        # Combine energy and spectral flux for better beat detection
        # Ensure both arrays have the same length
        min_length = min(len(energy), len(spectral_flux))
        combined_onset = 0.7 * np.array(energy[:min_length]) + 0.3 * np.array(spectral_flux[:min_length])
        
        # Calculate dynamic threshold using a moving window
        window_size = 30  # ~0.3 seconds at typical sample rates
        dynamic_threshold = []
        
        for i in range(len(combined_onset)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(combined_onset), i + 1)
            local_mean = np.mean(combined_onset[start_idx:end_idx])
            local_std = np.std(combined_onset[start_idx:end_idx]) if end_idx > start_idx else 0
            
            # Adaptive threshold based on local statistics
            threshold = local_mean + self.beat_threshold_multiplier * local_std
            dynamic_threshold.append(threshold)
        
        # Find peaks that exceed the dynamic threshold
        beats = []
        for i in range(1, len(combined_onset) - 1):
            # Peak condition: higher than neighbors and above threshold
            is_peak = (combined_onset[i] > combined_onset[i-1] and 
                      combined_onset[i] > combined_onset[i+1] and 
                      combined_onset[i] > dynamic_threshold[i])
            
            if is_peak:
                # Convert frame index to time in seconds
                beat_time = i * self.hop_size / sample_rate
                beats.append(beat_time)
        
        # Remove very closely spaced beats (likely duplicates)
        refined_beats = []
        min_beat_interval = 0.05  # Minimum 50ms between beats
        
        for beat in beats:
            if not refined_beats or beat - refined_beats[-1] > min_beat_interval:
                refined_beats.append(beat)
        
        return refined_beats
    
    def _calculate_spectral_flux(self, audio_data, sample_rate):
        """
        Calculate spectral flux for onset detection
        
        Parameters:
            audio_data: 1D numpy array of audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Array of spectral flux values
        """
        flux = []
        prev_magnitude = None
        
        for i in range(0, len(audio_data) - self.frame_size, self.hop_size):
            # Extract frame and apply window function
            frame = audio_data[i:i + self.frame_size] * np.hanning(self.frame_size)
            
            # Compute FFT and get magnitude spectrum
            fft_values = np.fft.rfft(frame)
            magnitude = np.abs(fft_values)
            
            # Calculate spectral flux as the sum of squared differences
            if prev_magnitude is not None:
                # Only consider positive changes
                diff = np.maximum(0, magnitude - prev_magnitude)
                flux_value = np.sum(diff**2)
                flux.append(flux_value)
            
            prev_magnitude = magnitude
        
        # Normalize flux values
        if flux:
            max_flux = np.max(flux)
            if max_flux > 0:
                flux = np.array(flux) / max_flux
        
        return flux
    
    def _calculate_bpm_candidates(self, beats, sample_rate):
        """
        Calculate BPM candidates from beat intervals
        
        Parameters:
            beats: List of beat timestamps in seconds
            sample_rate: Audio sample rate
            
        Returns:
            List of BPM candidates
        """
        if len(beats) < 2:
            return []
        
        # Calculate beat intervals (in seconds)
        intervals = []
        for i in range(1, len(beats)):
            interval = beats[i] - beats[i-1]
            intervals.append(interval)
        
        # Convert intervals to BPM
        bpm_candidates = []
        for interval in intervals:
            if interval > 0:
                # Calculate base BPM (quarter note)
                base_bpm = 60 / interval
                
                # Also consider common beat subdivisions and multiples
                # (eighth note, half note, whole note, etc.)
                for ratio in [0.5, 1.0, 2.0]:
                    candidate = base_bpm * ratio
                    # Keep BPM in reasonable range (40-220 BPM)
                    if 40 <= candidate <= 220:
                        bpm_candidates.append(candidate)
        
        return bpm_candidates
    
    def _filter_outliers_iqr(self, values):
        """
        Filter outliers using IQR method
        
        Parameters:
            values: List of values to filter
            
        Returns:
            Filtered list of values
        """
        if not values:
            return []
            
        # Convert to numpy array
        values_array = np.array(values)
        
        # Calculate Q1, Q3, and IQR
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter out outliers
        filtered_values = values_array[(values_array >= lower_bound) & (values_array <= upper_bound)]
        
        return filtered_values.tolist()
    
    def _moving_average(self, values, window_size):
        """
        Calculate moving average for smoothing
        
        Parameters:
            values: List of values
            window_size: Size of the moving window
            
        Returns:
            List of smoothed values
        """
        if len(values) < window_size:
            return values
            
        smoothed = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def analyze_audio_file(self, file_path):
        """
        Analyze audio file and return BPM
        
        Parameters:
            file_path: Path to audio file
            
        Returns:
            BPM value
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
            
            # Analyze audio data
            bpm = self.analyze_audio_data(samples, audio.frame_rate)
            
            return bpm
            
        except Exception as e:
            print(f"Error analyzing audio file: {e}")
            return 0
    
    def analyze_audio_segment(self, audio_segment, sample_rate):
        """
        Analyze a segment of audio data
        
        Parameters:
            audio_segment: 1D numpy array of audio samples
            sample_rate: Audio sample rate
            
        Returns:
            BPM value for the segment
        """
        return self.analyze_audio_data(audio_segment, sample_rate)

    def _bpm_to_category(self, bpm):
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