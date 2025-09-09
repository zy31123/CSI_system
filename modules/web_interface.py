#!/usr/bin/env python3
"""
Web Interface Module
Provides web-based visualization of CSI data from all processing stages
"""

import os
import sys
import json
import time
import threading
import numpy as np
from collections import deque
from threading import Lock

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import redis

# Add parent directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REDIS_HOST, REDIS_PORT, CSI_VISUALIZATION_CHANNEL, 
    CSI_CONTROL_CHANNEL,
    CSI_SOURCE_QUEUE, CSI_PROCESSED_QUEUE
)

# Create Flask app and SocketIO instance
app = Flask(__name__, static_folder='../static', template_folder='../templates')
app.config['SECRET_KEY'] = 'csi-visualization-secret-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Redis client connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Ensure templates and static files directories exist
os.makedirs('../templates', exist_ok=True)
os.makedirs('../static', exist_ok=True)

# Configuration parameters
SUBCARRIER_RANGE = 114  # Number of subcarriers
RX_ANTENNAS = 3  # Number of receive antennas
TX_ANTENNAS = 2  # Number of transmit antennas
PORT = 8080

# Store classification history
classification_history = []
MAX_HISTORY_SIZE = 100  # Maximum history records to save

# Status monitoring
class SystemStatus:
    def __init__(self, history_size=100):
        self.lock = Lock()
        self.history_size = history_size
        self.packet_rate = 0
        self.queue_lengths = {
            'source': 0,
            'processed': 0
        }
        self.processed_count = 0
        self.processing_rate = 0
        self.last_update = time.time()
        
        # History data
        self.packet_rate_history = deque(maxlen=history_size)
        self.queue_lengths_history = {
            'source': deque(maxlen=history_size),
            'processed': deque(maxlen=history_size)
        }
        self.processing_rate_history = deque(maxlen=history_size)
        
    def update(self, packet_rate=None, queue_lengths=None, processed_count=None):
        with self.lock:
            if packet_rate is not None:
                self.packet_rate = packet_rate
                self.packet_rate_history.append((time.time(), packet_rate))
                
            if queue_lengths is not None:
                self.queue_lengths.update(queue_lengths)
                for key, value in queue_lengths.items():
                    if key in self.queue_lengths_history:
                        self.queue_lengths_history[key].append((time.time(), value))
                
            if processed_count is not None:
                if hasattr(self, 'last_processed_count'):
                    time_diff = time.time() - self.last_update
                    if time_diff > 0:
                        self.processing_rate = (processed_count - self.last_processed_count) / time_diff
                        self.processing_rate_history.append((time.time(), self.processing_rate))
                
                self.last_processed_count = processed_count
                self.processed_count = processed_count
                
            self.last_update = time.time()
    
    def get_stats(self):
        with self.lock:
            return {
                'packet_rate': self.packet_rate,
                'queue_lengths': self.queue_lengths,
                'processed_count': self.processed_count,
                'processing_rate': self.processing_rate,
                'last_update': self.last_update
            }
    
    def get_history(self):
        with self.lock:
            return {
                'packet_rate': list(self.packet_rate_history),
                'queue_lengths': {k: list(v) for k, v in self.queue_lengths_history.items()},
                'processing_rate': list(self.processing_rate_history),
            }

# Create system status instance
system_status = SystemStatus()

# Routes
@app.route('/')
def index():
    return render_template('index.html', 
                          subcarrier_range=SUBCARRIER_RANGE,
                          rx_antennas=RX_ANTENNAS,
                          tx_antennas=TX_ANTENNAS)

@app.route('/api/status')
def get_status():
    stats = system_status.get_stats()
    # Add queue lengths
    stats['queue_lengths'] = {
        'source': redis_client.llen(CSI_SOURCE_QUEUE),
        'processed': redis_client.llen(CSI_PROCESSED_QUEUE)
    }
    return jsonify(stats)

@app.route('/api/history')
def get_history():
    return jsonify(system_status.get_history())

@app.route('/api/classification_history')
def get_classification_history():
    return jsonify({
        'history': classification_history,
        'count': len(classification_history)
    })

@app.route('/api/signal_summary')
def get_signal_summary():
    try:
        # Get recent CSI data
        recent_data = []
        for i in range(10):  # Get last 10 items
            data = redis_client.lindex(CSI_PROCESSED_QUEUE, i)
            if data:
                recent_data.append(json.loads(data))
        
        if recent_data:
            # Calculate summary statistics
            timestamps = [d['timestamp'] for d in recent_data]
            latest_time = max(timestamps)
            earliest_time = min(timestamps)
            time_span = latest_time - earliest_time
            
            return jsonify({
                'status': 'success',
                'count': len(recent_data),
                'latest_timestamp': latest_time,
                'time_span_ms': time_span,
                'sample_rate': len(recent_data) * 1000 / time_span if time_span > 0 else 0
            })
        else:
            return jsonify({
                'status': 'empty',
                'message': 'No data in queue'
            })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# SocketIO events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('config', {
        'subcarrier_range': SUBCARRIER_RANGE,
        'rx_antennas': RX_ANTENNAS,
        'tx_antennas': TX_ANTENNAS
    })
    
    # Send current system status
    socketio.emit('status_update', system_status.get_stats())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('control')
def handle_control(data):
    print(f"Received control command: {data}")
    redis_client.publish(CSI_CONTROL_CHANNEL, json.dumps(data))

# Redis CSI data listener thread
def redis_csi_listener():
    pubsub = redis_client.pubsub()
    # Subscribe to visualization channel
    pubsub.subscribe(CSI_VISUALIZATION_CHANNEL)
    print(f"Subscribed to CSI visualization channel: {CSI_VISUALIZATION_CHANNEL}")
    last_emit_time = 0
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                # Send received message to all connected clients via SocketIO
                data = message['data'].decode()
                parsed_data = json.loads(data)
                
                # Check if it's a classification result message
                if parsed_data.get('type') == 'classification_result':
                    # Get current classification result
                    current_result = {
                        'timestamp': parsed_data.get('timestamp'),
                        'formatted_time': parsed_data.get('formatted_time', time.strftime("%Y-%m-%d %H:%M:%S")),
                        'classification': parsed_data.get('classification'),
                        'confidence': parsed_data.get('confidence'),
                        'message': parsed_data.get('message', f"Classification result: {parsed_data.get('classification')}")
                    }
                    
                    # Check if it's a duplicate record (avoid duplicate display)
                    is_duplicate = False
                    if classification_history:
                        last_record = classification_history[-1]
                        # Check if classification result and confidence are the same
                        if (last_record.get('classification') == current_result.get('classification') and
                            last_record.get('confidence') == current_result.get('confidence')):
                            is_duplicate = True
                            print(f"Skipping duplicate classification result: {current_result.get('classification')}")
                    
                    # Save and send if not duplicate
                    if not is_duplicate:
                        # Save to history
                        if len(classification_history) >= MAX_HISTORY_SIZE:
                            classification_history.pop(0)  # Remove oldest record
                        
                        classification_history.append(current_result)
                        
                        # Send to frontend
                        socketio.emit('classification_result', json.dumps(parsed_data))
                        # print(f"Forwarded classification result: {parsed_data.get('classification')}, confidence: {parsed_data.get('confidence')}")
                    
                    continue
                
                # Add timestamp if missing
                if 'timestamp' not in parsed_data:
                    parsed_data['timestamp'] = int(time.time() * 1000)
                    
                # Optimize sending logic: Ensure sending frequency isn't too high causing browser lag
                current_time = time.time()
                if current_time - last_emit_time >= 0.01:  # Max 50ms per send
                    socketio.emit('csi_data', json.dumps(parsed_data))
                    last_emit_time = current_time

            except Exception as e:
                print(f"Error processing CSI message: {e}")
                import traceback
                traceback.print_exc()

# System status monitoring thread
def status_monitor():
    print("System status monitoring thread started")
    
    while True:
        try:
            # Get queue lengths
            source_length = redis_client.llen(CSI_SOURCE_QUEUE)
            processed_length = redis_client.llen(CSI_PROCESSED_QUEUE)
            
            queue_lengths = {
                'source': source_length,
                'processed': processed_length
            }
            
            # Calculate packet rate (simple estimation)
            current_time = time.time()
            if not hasattr(status_monitor, 'last_source_count'):
                status_monitor.last_source_count = source_length
                status_monitor.last_time = current_time
            
            time_diff = current_time - status_monitor.last_time
            if time_diff > 0:
                packet_rate = (source_length - status_monitor.last_source_count) / time_diff
                # Ensure packet_rate is not negative
                packet_rate = max(0, packet_rate)
            else:
                packet_rate = 0
                
            status_monitor.last_source_count = source_length
            status_monitor.last_time = current_time
            
            # Update system status with all metrics
            system_status.update(
                packet_rate=packet_rate,
                queue_lengths=queue_lengths,
                processed_count=processed_length
            )
            
            # Send status update every second
            stats = system_status.get_stats()
            # print(f"Sending status update: {stats}")  # 调试信息
            socketio.emit('status_update', stats)
            
            time.sleep(1.0)
        except Exception as e:
            print(f"Status monitoring thread error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)

# Main function
def main():
    # Start Redis listener thread
    redis_thread = threading.Thread(target=redis_csi_listener, name="RedisCSIListener")
    redis_thread.daemon = True
    redis_thread.start()
    
    # Start system status monitoring thread
    status_thread = threading.Thread(target=status_monitor, name="StatusMonitor")
    status_thread.daemon = True
    status_thread.start()
    
    # Start SocketIO server
    print(f"CSI visualization server starting at http://localhost:{PORT}")
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()