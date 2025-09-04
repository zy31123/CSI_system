#!/usr/bin/env python3
"""
Data Reception Module
Handles real-time CSI data collection and storage in Redis source queue
"""

import socket
import struct
import time
import sys
import numpy as np
import redis
import json
import os
import threading
from queue import Queue
from threading import Lock, Event
import datetime
import signal
from datetime import datetime, timezone

from config import (
    REDIS_HOST, REDIS_PORT, CSI_SOURCE_QUEUE, CSI_PROCESSED_QUEUE,
    CSI_VISUALIZATION_CHANNEL, CSI_CONTROL_CHANNEL,
    MAX_QUEUE_LENGTH, PORT
)

# Configuration parameters
NUM_SUBCARRIERS = 114
NUM_RX_ANTENNAS = 3
NUM_TX_ANTENNAS = 2
COMPLEX_ELEMENTS = NUM_SUBCARRIERS * NUM_RX_ANTENNAS * NUM_TX_ANTENNAS
TIMESTAMP_SIZE = 8  # Timestamp size (4 shorts, 8 bytes)
CSI_DATA_SIZE = TIMESTAMP_SIZE + (COMPLEX_ELEMENTS * 2 * 2)  # Timestamp + real and imaginary parts (2 bytes each)
print(f"CSI data packet size calculation: {TIMESTAMP_SIZE} + ({NUM_SUBCARRIERS} * {NUM_RX_ANTENNAS} * {NUM_TX_ANTENNAS} * 2 * 2) = {CSI_DATA_SIZE} bytes")

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Global variables
stop_event = Event()
data_lock = Lock()
packet_count = 0
packet_rate = 100
last_timestamp = 0

# Redis client connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

class DataReceptionThread(threading.Thread):
    def __init__(self):
        super().__init__(name="DataReceptionThread")
        self.socket = None
        self.running = True
        
    def run(self):
        """Main reception thread: Accept TCP connection and receive CSI data packets into Redis source queue"""
        global packet_count, last_timestamp
        
        print("Data reception thread started, waiting for client connection...")
        
        # Single client connection variables
        client_socket = None
        client_addr = None
        
        # Create TCP socket and bind to specified port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', PORT))
        self.socket.listen(5)  # Allow up to 5 connections in queue
        self.socket.settimeout(1.0)  # Set timeout to check stop event
        print(f"CSI server started, listening on TCP port {PORT}...")
        
        while not stop_event.is_set():
            try:
                # If no client connected yet, try to accept connection
                if client_socket is None:
                    try:
                        client_socket, client_addr = self.socket.accept()
                        client_socket.settimeout(0.5)  # Set client socket timeout
                        print(f"Client connected: {client_addr}")
                    except socket.timeout:
                        # Accept timeout, continue waiting
                        time.sleep(0.01)
                        continue
                
                # Client connected, receive data
                valid_count = 0
                invalid_count = 0   
                data = b""
                while client_socket and not stop_event.is_set():
                    try:
                        # Use loop to receive complete packet
                        timeout_count = 0
                        try:
                            chunk = client_socket.recv(CSI_DATA_SIZE - len(data))
                            bytes_received = 0
                            timeout_count = 0
                            data += chunk
                            if not data:  # Connection closed
                                print("Client disconnected, waiting for new connection...")
                                client_socket.close()
                                client_socket = None
                                break
                            bytes_received = len(data)

                            # Continue if received bytes don't equal expected size
                            if bytes_received != CSI_DATA_SIZE:
                                continue
                            timeout_count = 0  # Reset timeout counter
                        except socket.timeout:
                            # Receive timeout, continue trying
                            timeout_count += 1
                            time.sleep(0.1)  # Short sleep before retry
                            continue
                        
                        # Process complete packet
                        if len(data) == CSI_DATA_SIZE:
                            valid_count += 1
                            # Parse packet
                            timestamp_us = time.time()  # Microsecond timestamp
                            try:
                                # Parse data (matrix format)
                                parsed_data = self._parse_csi_data(data, timestamp_us)

                                # Efficiently push data to Redis queue
                                redis_client.lpush(CSI_SOURCE_QUEUE, json.dumps(parsed_data))
                                
                                # Limit queue length
                                redis_client.ltrim(CSI_SOURCE_QUEUE, 0, MAX_QUEUE_LENGTH - 1)
                                
                                # Minimal packet counting
                                with data_lock:
                                    packet_count += 1
                                    
                                # Print status every 100 packets
                                if packet_count % 100 == 0:
                                    print(f"Received {packet_count} packets, queue length: {redis_client.llen(CSI_SOURCE_QUEUE)}, process length: {redis_client.llen(CSI_PROCESSED_QUEUE)}")
                                data = b""
                            except Exception as e:
                                # Only log error, continue receiving
                                print(f"Data parsing error: {e}")
                                continue
                        else:
                            invalid_count += 1

                    except ConnectionError:
                        print("Client connection error, waiting for new connection...")
                        if client_socket:
                            client_socket.close()
                        client_socket = None
                        break  # Break inner loop, try to re-establish connection
                    except Exception as e:
                        print(f"Error processing client data: {e}")
                        continue
            except socket.timeout:
                # Main socket timeout, check if should stop
                continue
            except Exception as e:
                print(f"Main reception thread error: {e}")
                if not stop_event.is_set():
                    time.sleep(1)  # Avoid rapid error loop
    
    def _parse_csi_data(self, data, timestamp_us):
        """Efficiently parse CSI data packet, including timestamp in matrix format"""
        try:
            # Update data size to match added 4 shorts (8 bytes) timestamp
            expected_size = CSI_DATA_SIZE
            if len(data) != expected_size:
                raise ValueError(f"Packet size error: {len(data)} bytes, expected: {expected_size} bytes")
            
            # Extract send timestamp from first 8 bytes of packet (microseconds since Jan 1, 2025)
            # Read 4 unsigned short values and combine into 64-bit timestamp
            time_bytes = data[:8]
            # Use '<H' to parse as unsigned short
            timestamp_parts = [struct.unpack('<H', time_bytes[i:i+2])[0] for i in range(0, 8, 2)]
            
            # Combine timestamp using unsigned integers for bitwise operations
            send_timestamp = ((timestamp_parts[0] & 0xFFFF) << 48) | \
                            ((timestamp_parts[1] & 0xFFFF) << 32) | \
                            ((timestamp_parts[2] & 0xFFFF) << 16) | \
                            (timestamp_parts[3] & 0xFFFF)

            send_timestamp = send_timestamp / 1_000_000.0

            # Calculate number of complex values
            num_complex = NUM_SUBCARRIERS * NUM_RX_ANTENNAS * NUM_TX_ANTENNAS
            
            # Efficiently parse all real and imaginary parts, note data offset by 8 bytes (4 shorts for timestamp)
            # Use list comprehension to improve parsing efficiency
            offset = 8  # 8 bytes for timestamp
            real_parts = [struct.unpack('<h', data[offset + i*2:offset + i*2+2])[0] for i in range(num_complex)]
            imag_parts = [struct.unpack('<h', data[offset + num_complex*2 + i*2:offset + num_complex*2 + i*2+2])[0] for i in range(num_complex)]
            
            # Create 3D array - this part must be preserved
            csi_data = np.zeros((NUM_RX_ANTENNAS, NUM_TX_ANTENNAS, NUM_SUBCARRIERS, 2), dtype=np.int16)
            
            # Efficiently populate array
            for rx in range(NUM_RX_ANTENNAS):
                for tx in range(NUM_TX_ANTENNAS):
                    for sc in range(NUM_SUBCARRIERS):
                        idx = rx * (NUM_TX_ANTENNAS * NUM_SUBCARRIERS) + tx * NUM_SUBCARRIERS + sc
                        csi_data[rx, tx, sc, 0] = real_parts[idx]  # Real part
                        csi_data[rx, tx, sc, 1] = imag_parts[idx]  # Imaginary part
            
            # Return result, including send timestamp
            return {
                'timestamp': timestamp_us,              # Receive timestamp (microseconds)
                'send_time': send_timestamp,            # Send timestamp (microseconds, since Jan 1, 2025)
                'csi_data': csi_data.tolist()           # Convert to list for JSON serialization
            }
            
        except Exception as e:
            # Simplified error handling, keep main thread running
            print(f"Data parsing error: {str(e)}")
            raise
    
    def stop(self):
        """Stop the reception thread"""
        self.running = False
        stop_event.set()
        
        if self.socket:
            try:
                self.socket.close()
                print("Closed main TCP server socket")
            except Exception as e:
                print(f"Error closing socket: {e}")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nReceived interrupt signal, stopping server...")
    stop_event.set()
    time.sleep(1)
    print("Forcefully exiting program...")
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)