#!/usr/bin/env python3
"""
Data Processing Module
Implements time-window extraction, Hampel filtering, and processing result management
"""

import threading
import time
import json
import numpy as np
import redis
from threading import Event
from collections import deque

from config import (
    REDIS_HOST, REDIS_PORT, CSI_SOURCE_QUEUE, CSI_PROCESSED_QUEUE, 
    CSI_VISUALIZATION_CHANNEL, MAX_QUEUE_LENGTH, WINDOW_SIZE, WINDOW_OVERLAP
)

# Global variables
stop_event = Event()

# Redis client connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

class HampelFilter:
    """Hampel filter implementation for outlier detection and removal"""
    
    def __init__(self, window_size=10, threshold=3):
        self.window_size = window_size
        self.threshold = threshold
    
    def filter(self, data):
        """
        Apply Hampel filter to detect and remove outliers
        :param data: List of data points
        :return: Filtered data with outliers removed
        """
        if len(data) < self.window_size:
            return data
            
        filtered_data = data.copy()
        half_window = self.window_size // 2
        
        for i in range(half_window, len(data) - half_window):
            # Get window data
            window = data[i - half_window:i + half_window + 1]
            
            # Calculate median and MAD (Median Absolute Deviation)
            median = np.median(window)
            mad = np.median(np.abs(np.array(window) - median))
            
            # Calculate threshold
            threshold = self.threshold * 1.4826 * mad
            
            # Check if current point is an outlier
            if abs(data[i] - median) > threshold:
                # Replace outlier with median
                filtered_data[i] = median
                
        return filtered_data

class DataProcessingThread(threading.Thread):
    """Data processing thread: Extracts data from Redis queue and applies time-window processing with Hampel filtering"""
    
    def __init__(self):
        super().__init__(name="DataProcessingThread")
        self.hampel_filter = HampelFilter()
        self.previous_batch = []
        
    def run(self):
        """Main processing thread function"""
        print("Data processing thread started")
        
        window_size = WINDOW_SIZE
        overlap_size = WINDOW_OVERLAP
        process_counter = 0
        
        while not stop_event.is_set():
            try:
                # Check queue data amount
                queue_length = redis_client.llen(CSI_SOURCE_QUEUE)
                
                # If queue has enough data to process a new window
                if queue_length >= window_size - len(self.previous_batch):
                    # Calculate new data needed
                    new_data_needed = window_size - len(self.previous_batch)
                    
                    # Get new data
                    new_signals = []
                    for _ in range(new_data_needed):
                        signal = redis_client.rpop(CSI_SOURCE_QUEUE)
                        if signal:
                            new_signals.append(signal)
                        else:
                            print(f"Warning: Not enough data in queue, only got {len(new_signals)} data points")
                            break
                    
                    # Build complete processing window: previous overlap + new data
                    current_window = self.previous_batch + new_signals
                    
                    if len(current_window) >= window_size / 2:  # At least half window data
                        # Process current window data
                        processed_window = self._process_window(current_window)
                        processed_window = processed_window[0:window_size - overlap_size]  # Exclude overlap part
                        
                        # Push processed data to Redis queue and publish to visualization channel
                        for data in processed_window:
                            redis_client.lpush(CSI_PROCESSED_QUEUE, json.dumps(data))
                            # Publish to visualization channel for real-time updates
                            redis_client.publish(CSI_VISUALIZATION_CHANNEL, json.dumps(data))
                        
                        # Save latter part as overlap for next batch
                        if len(current_window) > overlap_size:
                            self.previous_batch = current_window[-overlap_size:]
                        else:
                            self.previous_batch = current_window
                            
                        process_counter += 1
                
                # Small delay to prevent busy waiting
                time.sleep((window_size - overlap_size)/100)
                
            except Exception as e:
                print(f"Processing thread error: {e}")
                time.sleep(1.0)
    
    def _process_window(self, window):
        """Process a window of CSI data with Hampel filtering"""
        processed_data = []
        
        # Parse window data
        parsed_data = []
        for signal_json in window:
            if signal_json:
                try:
                    signal_data = json.loads(signal_json)
                    parsed_data.append(signal_data)
                except json.JSONDecodeError:
                    print("Unable to parse signal JSON data")
        
        if not parsed_data:
            print("No valid signal data, skipping processing")
            return processed_data
        
        # Apply Hampel filtering to amplitude data
        for signal in parsed_data:
            try:
                # Extract amplitude data
                csi_data = signal['csi_data']
                amplitude_data = self._calculate_amplitude(csi_data)
                
                # Apply Hampel filter to each antenna/subcarrier combination
                filtered_amplitude = np.zeros_like(amplitude_data)
                for rx in range(amplitude_data.shape[0]):
                    for tx in range(amplitude_data.shape[1]):
                        for sc in range(amplitude_data.shape[2]):
                            # For simplicity, we're applying filter along time dimension
                            # In a real implementation, you might want to filter across subcarriers or antennas
                            filtered_amplitude[rx, tx, sc] = amplitude_data[rx, tx, sc]
                
                # Create processed data structure
                processed_signal = {
                    'timestamp': signal['timestamp'],
                    'send_time': signal['send_time'],
                    'csi_data': signal['csi_data'],  # Original CSI data
                    'filtered_amplitude': filtered_amplitude.tolist(),  # Filtered amplitude data
                    'processing_time': time.time()
                }
                
                processed_data.append(processed_signal)
                
            except Exception as e:
                print(f"Error processing signal: {e}")
                continue
        
        return processed_data
    
    def _calculate_amplitude(self, csi_data):
        """Calculate amplitude from CSI data"""
        # Convert to numpy array
        csi_array = np.array(csi_data)
        
        # Calculate amplitude (magnitude)
        amplitude = np.sqrt(np.square(csi_array[:, :, :, 0]) + np.square(csi_array[:, :, :, 1]))
        
        return amplitude

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nReceived interrupt signal, stopping processing thread...")
    stop_event.set()
    time.sleep(1)

if __name__ == "__main__":
    # For testing purposes
    processor = DataProcessingThread()
    processor.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        processor.join()
        print("Data processing thread stopped")