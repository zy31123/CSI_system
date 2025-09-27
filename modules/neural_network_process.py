#!/usr/bin/env python3
"""
Neural Network Process Module
Runs the neural network inference in a separate process for better GPU resource allocation
"""

import sys
import os
import time
import signal
from threading import Event
from neural_network import classifyLinear 

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.neural_network import NeuralNetworkInferenceThread, stop_event

def signal_handler(sig, frame):
    """Handle interrupt signals for graceful shutdown"""
    print("\nReceived interrupt signal, stopping neural network inference process...")
    stop_event.set()
    time.sleep(1)
    sys.exit(0)

def main():
    """Main entry point for the neural network process"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start neural network inference thread
    nn_thread = NeuralNetworkInferenceThread()
    nn_thread.start()
    print("Neural network inference thread started in separate process")
    
    try:
        # Keep process alive
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping neural network process...")
        stop_event.set()
    
    # Wait for thread to finish
    nn_thread.join(timeout=5)
    print("Neural network inference thread stopped")

if __name__ == "__main__":
    main()