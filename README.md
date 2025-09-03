# CSI Processing System

This system processes CSI (Channel State Information) data through multiple stages:
1. Data reception from TCP clients
2. Data processing with time-window extraction and Hampel filtering
3. Neural network inference for pattern recognition
4. Web-based visualization of all processing stages

## Project Structure

- `main.py`: Main application coordinating all components
- `config.py`: Centralized configuration
- `requirements.txt`: Python dependencies
- `modules/`: Contains all processing modules
  - `data_reception.py`: Handles real-time CSI data collection
  - `data_processing.py`: Implements time-window extraction and Hampel filtering
  - `neural_network.py`: Neural network inference engine
  - `web_interface.py`: Web-based visualization interface
- `static/`: Static files for web interface (CSS, JS)
- `templates/`: HTML templates for web interface
- `models/`: Directory for neural network models (to be created when needed)
- `data/`: Directory for data storage (created automatically)

## Installation

1. Install Redis server:
   - On Ubuntu/Debian: `sudo apt-get install redis-server`
   - On macOS: `brew install redis`
   - On Windows: Download from https://github.com/microsoftarchive/redis/releases

2. Start Redis server:
   - On Linux/macOS: `redis-server`
   - On Windows: Run `redis-server.exe`

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the CSI processing system:
   ```bash
   python main.py
   ```

2. Connect to the web interface at http://localhost:8080

3. Send CSI data to the system via TCP on port 4145

## Configuration

All system configuration is centralized in `config.py`. You can modify:
- Redis connection parameters
- Queue names and sizes
- Processing window sizes
- Network ports

Environment variables can also be used to override configuration:
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)