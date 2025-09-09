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
from modules.csi_pre_processing1 import CSIProcessor
from scipy.signal import detrend, medfilt2d

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
                        # processed_signal = {
                        #     'timestamp': processed_window[0]['timestamp'],
                        #     'send_time': processed_window[0]['send_time'],
                        #     # 'csi_data': signal['csi_data'],  # Original CSI data
                        #     'amplitude_data': amp_median.tolist(),  # Filtered amplitude data
                        #     'phase_data': phase_median.tolist(),  # Filtered phase data
                        # }

                        # redis_client.publish(CSI_VISUALIZATION_CHANNEL, json.dumps(processed_signal))
                        redis_client.publish(CSI_VISUALIZATION_CHANNEL, json.dumps(processed_window[-1]))

                        # Push processed data to Redis queue and publish to visualization channel
                        for data in processed_window:
                            redis_client.lpush(CSI_PROCESSED_QUEUE, json.dumps(data))
                            # redis_client.publish(CSI_VISUALIZATION_CHANNEL, json.dumps(data))
                            # redis_client.publish(CSI_VISUALIZATION_CHANNEL, json.dumps(data))
                        
                        # Save latter part as overlap for next batch
                        if len(current_window) > overlap_size:
                            self.previous_batch = current_window[-overlap_size:]
                        else:
                            self.previous_batch = current_window
                            
                        process_counter += 1
                
                # Small delay to prevent busy waiting
                # time.sleep((window_size - overlap_size)/100)
                
            except Exception as e:
                print(f"Processing thread error: {e}")
                time.sleep(1.0)
    
    def _process_window(self, window):
        """Process a window of CSI data with Hampel filtering"""
        processed_data = []
        
        # Parse window data
        parsed_data = []
        csi_maxtrix = []
        for signal_json in window:
            if signal_json:
                try:
                    signal_data = json.loads(signal_json)
                    csi_maxtrix.append(signal_data['csi_data'])
                    parsed_data.append(signal_data)
                except json.JSONDecodeError:
                    print("Unable to parse signal JSON data")
        
        if not parsed_data:
            print("No valid signal data, skipping processing")
            return processed_data
        
        csi_maxtrix = np.array(csi_maxtrix)  # Shape: (num_signals, num_rx, num_tx, num_subcarriers, 2)
        # processor = CSIProcessor(csi_maxtrix[:,:,:,:56], 0.9, 10, 3, 15)

        # amplitude_data_, phase_data_ = processor.do_process()


        amplitude_data_,phase_data_,subcarriers_data_ = self.process_csi_data(csi_maxtrix) # Shape: (num_signals, num_rx, num_tx, num_subcarriers, 2)
        # amplitude_data_ = np.sqrt(np.square(csi_maxtrix[:, :, :, :, 0]) + np.square(csi_maxtrix[:, :, :, :, 1]))
        # amp_median = np.median(amplitude_data_, axis=0, keepdims=False)
        # phase_data_ = np.arctan2(csi_maxtrix[:, :, :, :, 1], csi_maxtrix[:, :, :, :, 0])
        # phase_median = np.median(phase_data_, axis=0, keepdims=False)

        # Apply Hampel filtering to amplitude data
        for i,signal in enumerate(parsed_data):
            try:
                # Extract amplitude data
                amplitude_data = amplitude_data_[i]
                phase_data = phase_data_[i]

                # Create processed data structure
                processed_signal = {
                    'timestamp': signal['timestamp'],
                    'send_time': signal['send_time'],
                    # 'csi_data': signal['csi_data'],  # Original CSI data
                    'amplitude_data': amplitude_data.tolist(),  # Filtered amplitude data
                    'phase_data': phase_data.tolist(),  # Filtered phase data
                    'subcarriers_data': subcarriers_data_[i].tolist(),  # Filtered subcarriers data
                }
                
                processed_data.append(processed_signal)
                
            except Exception as e:
                print(f"Error processing signal: {e}")
                continue

        return processed_data

    def process_csi_data(self,csi_data):
        """
        使用环形差分（Circular Differential）重构相位
        - 幅度：Hampel滤波
        - 相位：每个发射天线下，三个接收天线构成闭环差分
        - 重构所有天线的相位（基于 r0 为虚拟参考）
        - 输出保持原始结构，不返回 phase_diff
        """
        # processed_csi_data = np.copy(csi_data)
        N, R, T, M, _ = csi_data.shape
        assert R == 3, "必须是3个接收天线"
        subcarriers_to_process = 56

        # 存储最终复数数据
        # final_complex = np.zeros((N, R, T, subcarriers_to_process), dtype=complex)
        amplitude_data = np.zeros((N, R, T, subcarriers_to_process))
        phase_data = np.zeros((N, R, T, subcarriers_to_process))
        # 得到CSI的频谱数据
        subcarriers_data = np.zeros((N, R, T, subcarriers_to_process))

        # ======== Step 1: 提取并预处理每个天线的原始相位 ========
        # 将csi_data转换为复数形式，形状为(N, R, T, M)
        csi_complex = csi_data[..., 0] + 1j * csi_data[..., 1]
        
        # 计算幅度，形状为(N, R, T, M)
        amps = np.abs(csi_complex)
        
        # 参考子载波索引
        ref_subcarrier_idx = 28
        
        # 向量化归一化：使用广播机制 (N, R, T, M) / (N, R, T, 1)
        ref_csi = amps[:, :, :, ref_subcarrier_idx:ref_subcarrier_idx+1]
        ref_csi = np.where(ref_csi == 0, 1e-6, ref_csi)
        amps = amps / (ref_csi + 1e-6)
        
        # 提取前56个子载波
        amps = amps[:, :, :, :subcarriers_to_process]
        subcarriers_data = amps.copy()
        
        # 计算相位，形状为(N, R, T, M)
        phases = np.angle(csi_complex)
        phases = phases[:, :, :, :subcarriers_to_process]

        # ======== Step 2: 构造环形差分（闭环）========
        # Δ01 = φ0 - φ1, Δ12 = φ1 - φ2, Δ20 = φ2 - φ0
        # 使用向量化操作计算所有发射天线的环形差分
        # 直接在原始维度上操作 (N, R, T, M)
        phi_r0_recon = phases[:, 0, :, :] - phases[:, 1, :, :]  # (N, T, M)
        phi_r1_recon = phases[:, 1, :, :] - phases[:, 2, :, :]  # (N, T, M)
        phi_r2_recon = phases[:, 2, :, :] - phases[:, 0, :, :]  # (N, T, M)

        # ======== Step 3: 重构相位 ========
        # 重构所有天线的相位（基于 r0 为虚拟参考）
        # 这里我们直接使用差分相位作为重构相位
        
        # 重新排列差分相位为 (R, N, T, M) 格式
        reconstructed_phases = np.stack([phi_r0_recon, phi_r1_recon, phi_r2_recon], axis=1)  # (N, R, T, M)

        # ======== Step 4: 幅度滤波 ========
        # 向量化Hampel滤波处理所有发射天线
        # 重塑amps为 (N*R*T, M) 以适应滤波器
        amps_reshaped = amps.reshape(N, -1)  # (N*R*T, M)
        amps_filtered = self.vectorized_hampel_filter(amps_reshaped, window_size=11, n_sigmas=0.5)
        # 恢复形状 (N, R, T, M)
        amps_filtered = amps_filtered.reshape(N, R, T, subcarriers_to_process)
        amplitude_data = amps_filtered
        
        # 向量化Hampel滤波处理所有重构相位
        # 重塑reconstructed_phases为 (N*R*T, M) 以适应滤波器
        phases_reshaped = reconstructed_phases.reshape(N, -1)  # (N*R*T, M)
        phases_filtered = self.vectorized_hampel_filter(phases_reshaped, window_size=11, n_sigmas=0.5)
        # 恢复形状 (N, R, T, M)
        phases_filtered = phases_filtered.reshape(N, R, T, subcarriers_to_process)
        phase_data = phases_filtered

        # ======== Step 6: 写回 processed_csi_data ========
        # for rx in range(R):
        #     for tx in range(T):
        #         processed_csi_data[:, rx, tx, :subcarriers_to_process, 0] = np.real(final_complex[:, rx, tx, :])
        #         processed_csi_data[:, rx, tx, :subcarriers_to_process, 1] = np.imag(final_complex[:, rx, tx, :])

        return amplitude_data, phase_data, subcarriers_data

    def vectorized_hampel_filter(self, data, window_size=11, n_sigmas=0.6):
        """
        向量化的Hampel滤波器实现，用于提高处理速度
        
        参数:
        data: 输入数据数组 (数据包数, 子载波数)
        window_size: 滑动窗口大小
        n_sigmas: 判定为异常值的标准差倍数
        
        返回:
        filtered_data: 过滤后的数据
        """
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
        
        # 使用scipy的medfilt2d进行更高效的中值滤波
        # 先对数据进行中值滤波
        median_filtered = medfilt2d(data, kernel_size=(window_size, 1))
        
        # 计算MAD (Median Absolute Deviation)
        mad = np.median(np.abs(data - median_filtered), axis=0)
        
        # 计算阈值
        thresholds = n_sigmas * 1.4826 * mad
        
        # 检查每个点是否为异常值
        diff = np.abs(data - median_filtered)
        outlier_mask = diff > thresholds
        
        # 创建过滤后的数据副本
        filtered_data = np.copy(data)
        
        # 替换异常值为中值
        filtered_data[outlier_mask] = median_filtered[outlier_mask]
        
        return filtered_data

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