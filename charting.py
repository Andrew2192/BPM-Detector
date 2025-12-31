#!/usr/bin/env python3
"""
Charting module for BPM Analyzer
Handles chart plotting and visualization functionality
"""

import numpy as np
from scipy import signal


class ChartManager:
    """
    Chart management for BPM Analyzer
    """
    
    def __init__(self):
        """
        Initialize the chart manager
        """
        pass
    
    def create_bpm_chart(self, ax, canvas, fig, time_bpm_pairs, audio_duration):
        """
        Create BPM variation chart using matplotlib
        
        Parameters:
            ax: matplotlib axes object
            canvas: matplotlib canvas object
            fig: matplotlib figure object
            time_bpm_pairs: list of tuples (time, bpm)
            audio_duration: total audio duration in seconds
        """
        if not time_bpm_pairs:
            return
            
        # Clear previous plot
        ax.clear()
        
        # Extract data
        times, bpms = zip(*time_bpm_pairs)
        times_seconds = list(times)  # Use seconds directly
        
        # Apply smoothing to BPM values for better visualization
        smoothed_bpms = self._smooth_bpm_values(bpms)
        
        # Plot smoothed BPM curve
        ax.plot(times_seconds, smoothed_bpms, 'b-', linewidth=2, alpha=0.7, label='BPM')
        
        # Plot original BPM points
        ax.scatter(times_seconds, bpms, color='r', s=30, alpha=0.5, label='Raw BPM')
        
        # Add average BPM line
        avg_bpm = np.mean(bpms)
        ax.axhline(y=avg_bpm, color='g', linestyle='--', alpha=0.7, label=f'Avg BPM: {avg_bpm:.1f}')
        
        # Configure plot
        ax.set_title("BPM Variation Over Time", pad=10)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("BPM")
        
        # Set appropriate y-axis limits
        min_bpm = max(40, np.min(bpms) - 10)
        max_bpm = min(220, np.max(bpms) + 10)
        ax.set_ylim(min_bpm, max_bpm)
        
        # Set x-axis limits to include the full audio duration or data extent
        max_time = max(times_seconds) if times_seconds else 0.0
        right_limit = audio_duration if audio_duration > 0 else (max_time + 2)
        if right_limit < 5:
            right_limit = 5
        ax.set_xlim(0, right_limit)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Ensure title is not clipped
        try:
            fig.subplots_adjust(top=0.92)
        except Exception:
            pass
        # Redraw canvas with safe margins for titles
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        canvas.draw()
    
    def update_mic_bpm_chart(self, ax_mic, canvas_mic, fig_mic, mic_time_bpm_pairs):
        """
        Update the microphone BPM chart
        
        Parameters:
            ax_mic: matplotlib axes object for microphone chart
            canvas_mic: matplotlib canvas object for microphone chart
            fig_mic: matplotlib figure object for microphone chart
            mic_time_bpm_pairs: list of tuples (time, bpm) for microphone data
        """
        if not mic_time_bpm_pairs:
            # Initialize empty chart
            ax_mic.clear()
            ax_mic.set_title("Real-time Microphone BPM")
            ax_mic.set_xlabel("Time (seconds)")
            ax_mic.set_ylabel("BPM")
            ax_mic.set_ylim(40, 220)
            ax_mic.grid(True, alpha=0.3)
            # Add a message for empty data
            ax_mic.text(0.5, 0.5, "No BPM data yet\nClick the microphone button to start recording", 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=ax_mic.transAxes, fontsize=12, color='gray')
            fig_mic.tight_layout(rect=[0, 0.12, 1, 0.92])
            canvas_mic.draw()
            return
            
        try:
            # Extract data
            times, bpms = zip(*mic_time_bpm_pairs)
            times_seconds = list(times)
            
            # Clear previous plot
            ax_mic.clear()
            
            # Apply smoothing to BPM values if we have enough data
            if len(bpms) >= 3:
                smoothed_bpms = self._smooth_bpm_values(bpms)
                # Plot smoothed BPM curve
                ax_mic.plot(times_seconds, smoothed_bpms, 'b-', linewidth=2, alpha=0.7, label='BPM')
            else:
                # For few data points, just plot the line without smoothing
                ax_mic.plot(times_seconds, bpms, 'b-', linewidth=2, alpha=0.7, label='BPM')
            
            # Plot original BPM points
            ax_mic.scatter(times_seconds, bpms, color='r', s=30, alpha=0.5, label='Raw BPM')
            
            # Add average BPM line if we have enough data
            if len(bpms) > 0:
                avg_bpm = np.mean(bpms)
                ax_mic.axhline(y=avg_bpm, color='g', linestyle='--', alpha=0.7, label=f'Avg BPM: {avg_bpm:.1f}')
            
            # Configure plot
            ax_mic.set_title("Real-time Microphone BPM")
            ax_mic.set_xlabel("Time (seconds)")
            ax_mic.set_ylabel("BPM")
            
            # Set appropriate y-axis limits
            min_bpm = max(40, np.min(bpms) - 10) if bpms else 40
            max_bpm = min(220, np.max(bpms) + 10) if bpms else 220
            ax_mic.set_ylim(min_bpm, max_bpm)
            
            # Set x-axis limits
            max_time = max(times_seconds) if times_seconds else 0
            ax_mic.set_xlim(0, max_time + 2)
            
            # Add grid and legend
            ax_mic.grid(True, alpha=0.3)
            ax_mic.legend(loc='upper right')
            
            # Update canvas without tight_layout to reduce flickering
            # Only call tight_layout occasionally
            if len(mic_time_bpm_pairs) % 10 == 0:
                fig_mic.tight_layout(rect=[0, 0.12, 1, 0.92])
            
            # Redraw canvas
            canvas_mic.draw()
        except Exception as e:
            # Log error but don't crash
            print(f"Error updating microphone chart: {e}")
    
    def update_chart_indicator(self, ax, canvas, time_bpm_pairs, current_time, parent_obj):
        """
        Update the real-time BPM indicator on the chart
        
        Parameters:
            ax: matplotlib axes object
            canvas: matplotlib canvas object
            time_bpm_pairs: list of tuples (time, bpm)
            current_time: current playback time in seconds
            parent_obj: parent object to store indicator attributes
        """
        if not time_bpm_pairs:
            return
        
        # Find current BPM based on time
        current_bpm = None
        for time_seconds, bpm in time_bpm_pairs:
            if time_seconds > current_time:
                break
            current_bpm = bpm
        
        if current_bpm is None:
            return
        
        # Remove previous indicator if exists
        if hasattr(parent_obj, '_current_bpm_line'):
            try:
                parent_obj._current_bpm_line.remove()
                if hasattr(parent_obj, '_current_bpm_text'):
                    parent_obj._current_bpm_text.remove()
            except:
                pass
        
        # Draw vertical line at current position
        parent_obj._current_bpm_line = ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
        
        # Display current BPM value
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * 0.9  # Position near top of chart
        parent_obj._current_bpm_text = ax.text(current_time, y_pos, f"Current BPM: {current_bpm:.1f}", 
                                             color='red', fontsize=10, ha='center', va='top',
                                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        
        # Redraw canvas
        canvas.draw()
    
    def _smooth_bpm_values(self, bpm_values, window_size=3):
        """
        Apply smoothing to BPM values for better visualization
        
        Parameters:
            bpm_values: list or numpy array of BPM values
            window_size: size of smoothing window
            
        Returns:
            numpy array of smoothed BPM values
        """
        if len(bpm_values) < window_size:
            return bpm_values
            
        # Use Gaussian filter for smoothing
        smoothed = signal.wiener(bpm_values, window_size)
        return smoothed
    
    def clear_chart_indicator(self, ax, canvas, parent_obj):
        """
        Clear the chart indicator
        
        Parameters:
            ax: matplotlib axes object
            canvas: matplotlib canvas object
            parent_obj: parent object to check for indicator attributes
        """
        if hasattr(parent_obj, '_current_bpm_line'):
            try:
                parent_obj._current_bpm_line.remove()
                if hasattr(parent_obj, '_current_bpm_text'):
                    parent_obj._current_bpm_text.remove()
                delattr(parent_obj, '_current_bpm_line')
                delattr(parent_obj, '_current_bpm_text')
            except:
                pass
            canvas.draw()
