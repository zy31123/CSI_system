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
from scipy.signal import detrend

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

        # 天线索引
        r0, r1, r2 = 0, 1, 2

        for tx in range(T):
            # ======== Step 1: 提取并预处理每个天线的原始相位 ========
            phases = np.zeros((R, N, subcarriers_to_process))
            amps = np.zeros((R, N, subcarriers_to_process))

            for idx in range(R):
                csi_slice = csi_data[:, idx, tx, :subcarriers_to_process, :]
                csi_complex = csi_slice[..., 0] + 1j * csi_slice[..., 1]
                amp = np.abs(csi_complex)

                ref_subcarrier_idx = 28  # 选择第29个子载波作为参考
                ref_csi = amp[:, ref_subcarrier_idx:ref_subcarrier_idx+1]  # (N, R*T, 1)
                ref_csi = np.where(ref_csi == 0, 1e-6, ref_csi)
                
                amp = amp / (ref_csi + 1e-6)  # (N, R*T, K)
                subcarriers_data[:, idx, tx, :] = amp

                # ref_subcarrier_idx = 28
                # # 添加安全检查，避免除以零或极小值
                # ref_amp = amp[:, [ref_subcarrier_idx]]
                # # 设置一个最小阈值，避免除以过小的值
                # ref_amp = np.where(ref_amp < 1e-10, 1e-10, ref_amp)
                # amp = amp / ref_amp

                # amp = amp / amp[:, [28]]
                # amp = amp / amp[:, [28]]  # 归一化，除以参考子载波
                # for i in range(amp.shape[1]):
                #     amp[:,i] = amp[:,i] / amp[:,28]

                pha = np.angle(csi_complex)
                # pha = np.unwrap(pha, axis=0)
                # pha_detrended = detrend(pha_unwrapped, axis=0, type='linear')
                amps[idx] = amp
                phases[idx] = pha

            # ======== Step 2: 构造环形差分（闭环）========
            # Δ01 = φ0 - φ1, Δ12 = φ1 - φ2, Δ20 = φ2 - φ0
            phi_r0_recon = phases[r0] - phases[r1]  # (N, M)
            phi_r1_recon = phases[r1] - phases[r2]
            phi_r2_recon = phases[r2] - phases[r0]

            # 可选：对差分相位再次解缠（提高连续性）
            # phi_r0_recon = np.unwrap(phi_r0_recon, axis=0)
            # phi_r1_recon = np.unwrap(phi_r1_recon, axis=0)
            # phi_r2_recon = np.unwrap(phi_r2_recon, axis=0)

            # # # # # 去线性趋势
            # phi_r0_recon = detrend(phi_r0_recon, axis=0, type='linear')
            # phi_r1_recon = detrend(phi_r1_recon, axis=0, type='linear')
            # phi_r2_recon = detrend(phi_r2_recon, axis=0, type='linear')

            # phi_r0_recon = np.unwrap(phi_r0_recon, axis=0)
            # phi_r1_recon = np.unwrap(phi_r1_recon, axis=0)
            # phi_r2_recon = np.unwrap(phi_r2_recon, axis=0)

            reconstructed_phases = [phi_r0_recon, phi_r1_recon, phi_r2_recon]

            # ======== Step 5: 幅度滤波 + 重建复数 CSI ========
            for rx in range(R):
                # 滤波幅度
                amp_raw = amps[rx]

                amp_filtered = self.vectorized_hampel_filter(amp_raw, window_size=11, n_sigmas=0.5)
                amplitude_data[:, rx, tx, :] = amp_filtered
                # amp_filtered = amp_raw
                phase_filtered = self.vectorized_hampel_filter(reconstructed_phases[rx], window_size=11, n_sigmas=0.5)
                phase_data[:, rx, tx, :] = phase_filtered
                # phase_filtered = reconstructed_phases[rx]
                # 重建复数
                # final_complex[:, rx, tx, :] = amp_filtered * np.exp(1j * phase_filtered)

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
        
        half_window = window_size // 2
        n_packets, n_subcarriers = data.shape
        filtered_data = np.copy(data)
        
        # 创建一个数组来存储每个点的中位数和MAD
        all_medians = np.zeros_like(data)
        all_thresholds = np.zeros_like(data)
        
        # 预计算每个点的中位数和MAD
        for i in range(n_packets):
            # 确定窗口范围
            start_idx = max(0, i - half_window)
            end_idx = min(n_packets, i + half_window + 1)
            
            # 获取窗口内的数据 (窗口大小, 子载波数)
            window_data = data[start_idx:end_idx, :]
            
            # 计算中位数和MAD (Median Absolute Deviation)
            median = np.median(window_data, axis=0)  # (子载波数,)
            mad = np.median(np.abs(window_data - median), axis=0)  # (子载波数,)
            
            # 存储中位数和阈值
            all_medians[i, :] = median
            all_thresholds[i, :] = n_sigmas * 1.4826 * mad  # (子载波数,)
        
        # 检查每个点是否为异常值
        diff = np.abs(data - all_medians)  # (数据包数, 子载波数)
        outlier_mask = diff > all_thresholds  # (数据包数, 子载波数)
        
        # 替换异常值为中位数
        filtered_data[outlier_mask] = all_medians[outlier_mask]
        
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