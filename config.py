#!/usr/bin/env python3
"""
Configuration Module
Centralized configuration for the CSI processing system
"""

import os

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

# Queue and channel names
CSI_SOURCE_QUEUE = 'csi_source_queue'       # Raw CSI data queue
CSI_PROCESSED_QUEUE = 'csi_processed_queue' # Processed CSI data queue
CSI_VISUALIZATION_CHANNEL = 'csi_visualization'  # Real-time visualization channel
CSI_CONTROL_CHANNEL = 'csi_control'         # Control commands channel

# System configuration
MAX_QUEUE_LENGTH = 1000     # Maximum Redis queue length
WINDOW_SIZE = 20           # Processing window size (number of packets)
WINDOW_OVERLAP = 5         # Window overlap size (number of packets)
PORT = 4145                 # TCP port for CSI data reception
PACKET_RATE = 100           # Packet processing rate (packets per second)
NETWORK_SIZE = 400
NETWORK_OVERLAP = 300