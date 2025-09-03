#!/usr/bin/env python3
"""
Neural Network Inference Module
Processes time-window data and generates results using a neural network model
"""

import threading
import time
import json
import numpy as np
import redis
from threading import Event
import pickle
import os

from config import (
    REDIS_HOST, REDIS_PORT, CSI_PROCESSED_QUEUE, 
    CSI_VISUALIZATION_CHANNEL, MAX_QUEUE_LENGTH, WINDOW_SIZE
)

# Global variables
stop_event = Event()

# Redis client connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

class NeuralNetworkInferenceThread(threading.Thread):
    """Neural network inference thread: Processes time-window data and generates results"""
    
    def __init__(self):
        super().__init__(name="NeuralNetworkInferenceThread")
        self.model = None
        self.buffer = []
        self.load_model()
        
    def load_model(self):
        """Load neural network model (placeholder implementation)"""
        # In a real implementation, you would load your trained model here
        # For example, using TensorFlow/Keras:
        # self.model = tf.keras.models.load_model('path/to/model.h5')
        
        # For this implementation, we'll create a mock model that generates random results
        print("Loading neural network model...")
        # Check if model file exists
        if os.path.exists("models/nn_model.pkl"):
            try:
                with open("models/nn_model.pkl", "rb") as f:
                    self.model = pickle.load(f)
                print("Neural network model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print("Model file not found, using mock model")
            self.model = None
    
    def run(self):
        """Main neural network inference thread function"""
        print("Neural network inference thread started")
        
        while not stop_event.is_set():
            try:
                # Check if there's data in the processed queue
                queue_length = redis_client.llen(CSI_PROCESSED_QUEUE)
                
                # Process data in batches
                batch_size = 10
                if queue_length >= batch_size:
                    # Get a batch of processed data
                    batch_data = []
                    for _ in range(batch_size):
                        data = redis_client.rpop(CSI_PROCESSED_QUEUE)
                        if data:
                            batch_data.append(data)
                    
                    # Process the batch
                    if batch_data:
                        results = self._process_batch(batch_data)
                        
                        # Publish results to visualization channel for real-time updates
                        for result in results:
                            # Format result for visualization
                            visualization_result = {
                                'type': 'classification_result',
                                'timestamp': result['timestamp'],
                                'classification': self._format_classification(result['prediction']),
                                'confidence': result['confidence'],
                                'processing_time': result['processing_time']
                            }
                            redis_client.publish(CSI_VISUALIZATION_CHANNEL, json.dumps(visualization_result))
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Neural network inference thread error: {e}")
                time.sleep(1.0)
    
    def _process_batch(self, batch_data):
        """Process a batch of CSI data using the neural network"""
        results = []
        
        # Parse batch data
        parsed_data = []
        for data_json in batch_data:
            if data_json:
                try:
                    data = json.loads(data_json)
                    parsed_data.append(data)
                except json.JSONDecodeError:
                    print("Unable to parse data JSON")
        
        if not parsed_data:
            print("No valid data, skipping processing")
            return results
        
        # Prepare data for neural network input
        # In a real implementation, you would preprocess the data according to your model's requirements
        nn_input = self._prepare_input(parsed_data)
        
        # Process with neural network
        if self.model:
            # Actual model inference
            # predictions = self.model.predict(nn_input)
            # For now, we'll use a mock prediction
            predictions = self._mock_predict(nn_input)
        else:
            # Mock prediction when no model is loaded
            predictions = self._mock_predict(nn_input)
        
        # Format results
        for i, prediction in enumerate(predictions):
            result = {
                'timestamp': parsed_data[i]['timestamp'] if i < len(parsed_data) else time.time(),
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'confidence': float(np.max(prediction)) if hasattr(prediction, '__getitem__') else 0.0,
                'processing_time': time.time()
            }
            results.append(result)
        
        return results
    
    def _format_classification(self, prediction):
        """Format prediction result as a classification label"""
        # For this example, we'll use simple labels
        # In a real implementation, you would map the prediction to meaningful labels
        if isinstance(prediction, list):
            prediction = np.array(prediction)
        
        # Find the index of the highest probability
        if hasattr(prediction, 'argmax'):
            class_idx = prediction.argmax()
        else:
            class_idx = 0
            
        # Map index to label
        labels = ['Class A', 'Class B']  # Replace with your actual labels
        return labels[class_idx] if class_idx < len(labels) else f'Class {class_idx}'
    
    def _prepare_input(self, parsed_data):
        """Prepare data for neural network input"""
        # Extract CSI data and convert to appropriate format for neural network
        # This is a simplified example - in practice, you would need to implement
        # the specific preprocessing required by your model
        
        # For this example, we'll just extract amplitude data and flatten it
        input_data = []
        for data in parsed_data:
            try:
                # Get amplitude data (assuming it's already been processed)
                if 'filtered_amplitude' in data:
                    amplitude_data = np.array(data['filtered_amplitude'])
                else:
                    # Calculate amplitude from raw CSI data
                    csi_array = np.array(data['csi_data'])
                    amplitude_data = np.sqrt(np.square(csi_array[:, :, :, 0]) + np.square(csi_array[:, :, :, 1]))
                
                # Flatten the data for neural network input
                flattened = amplitude_data.flatten()
                input_data.append(flattened)
            except Exception as e:
                # print(f"Error preparing input data: {e}")
                # Add zero array as fallback
                input_data.append(np.zeros(114 * 3 * 2))  # Adjust size as needed
        
        # Convert to numpy array
        if input_data:
            nn_input = np.array(input_data)
        else:
            # Return empty array with correct shape if no data
            nn_input = np.array([]).reshape(0, 114 * 3 * 2)  # Adjust shape as needed
        
        return nn_input
    
    def _mock_predict(self, input_data):
        """Mock prediction function for testing"""
        # Generate random predictions (replace with actual model inference)
        if input_data.size > 0:
            # Generate random predictions with 3 classes for example
            predictions = np.random.rand(input_data.shape[0], 2)
            # Normalize to sum to 1 for each sample (like softmax)
            predictions = predictions / predictions.sum(axis=1, keepdims=True)
        else:
            predictions = np.array([])
        
        return predictions

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nReceived interrupt signal, stopping neural network inference thread...")
    stop_event.set()
    time.sleep(1)

if __name__ == "__main__":
    # For testing purposes
    nn_thread = NeuralNetworkInferenceThread()
    nn_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        nn_thread.join()
        print("Neural network inference thread stopped")