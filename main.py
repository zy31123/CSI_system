#!/usr/bin/env python3
"""
Main Application Module
Coordinates all components of the CSI processing system
"""

import sys
import os
import signal
import time
import threading
import subprocess
from pathlib import Path

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PORT, REDIS_HOST, REDIS_PORT
from modules.data_reception import DataReceptionThread, signal_handler
from modules.data_processing import DataProcessingThread
from modules.neural_network import NeuralNetworkInferenceThread

class CSIMainApplication:
    """Main application class to coordinate all CSI processing components"""
    
    def __init__(self):
        self.data_reception_thread = None
        self.data_processing_thread = None
        self.nn_inference_thread = None
        self.web_server_process = None
        self.running = False
        
    def start(self):
        """Start all components of the CSI processing system"""
        print("Starting CSI processing system...")
        
        try:
            # Start data reception thread
            self.data_reception_thread = DataReceptionThread()
            self.data_reception_thread.start()
            print("Data reception thread started")
            
            # Start data processing thread
            self.data_processing_thread = DataProcessingThread()
            self.data_processing_thread.start()
            print("Data processing thread started")
            
            # Start neural network inference thread
            self.nn_inference_thread = NeuralNetworkInferenceThread()
            self.nn_inference_thread.start()
            print("Neural network inference thread started")
            
            # Start web interface server in a separate process
            self._start_web_server()
            print("Web interface server started")
            
            # Set running flag
            self.running = True
            
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print("CSI processing system is now running. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            print(f"Error starting CSI processing system: {e}")
            self.stop()
    
    def _start_web_server(self):
        """Start the web interface server as a separate process"""
        try:
            # Change to the project directory to ensure correct path resolution
            project_dir = Path(__file__).parent
            
            # Start the web server process
            self.web_server_process = subprocess.Popen(
                [sys.executable, "-m", "modules.web_interface"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=project_dir
            )
            
            # Start a thread to read and print web server output
            threading.Thread(target=self._read_web_server_output, daemon=True).start()
            
        except Exception as e:
            print(f"Error starting web server: {e}")
    
    def _read_web_server_output(self):
        """Read and print web server output"""
        if self.web_server_process and self.web_server_process.stdout:
            for line in self.web_server_process.stdout:
                print(f"[Web Server] {line.strip()}")
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals for graceful shutdown"""
        print("\nReceived interrupt signal, stopping CSI processing system...")
        self.stop()
        sys.exit(0)
    
    def stop(self):
        """Stop all components of the CSI processing system"""
        print("Stopping CSI processing system...")
        self.running = False
        
        # Stop data reception thread
        if self.data_reception_thread:
            self.data_reception_thread.stop()
            self.data_reception_thread.join(timeout=5)
            print("Data reception thread stopped")
        
        # Stop data processing thread
        if self.data_processing_thread:
            # Set stop event for data processing thread
            import modules.data_processing as dp
            dp.stop_event.set()
            self.data_processing_thread.join(timeout=5)
            print("Data processing thread stopped")
        
        # Stop neural network inference thread
        if self.nn_inference_thread:
            # Set stop event for neural network thread
            import modules.neural_network as nn
            nn.stop_event.set()
            self.nn_inference_thread.join(timeout=5)
            print("Neural network inference thread stopped")
        
        # Stop web server process
        if self.web_server_process:
            self.web_server_process.terminate()
            try:
                self.web_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.web_server_process.kill()
            print("Web interface server stopped")
        
        print("CSI processing system stopped")

def main():
    """Main entry point for the application"""
    app = CSIMainApplication()
    
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping application...")
        app.stop()
    except Exception as e:
        print(f"Unexpected error: {e}")
        app.stop()

if __name__ == "__main__":
    main()