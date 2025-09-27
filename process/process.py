from scipy.signal import detrend
import numpy as np
import math
from scipy.signal import medfilt2d
import struct

NUM_SUBCARRIERS = 114
NUM_RX_ANTENNAS = 3
NUM_TX_ANTENNAS = 2
CSI_DATA_SIZE = 2736 + 8  # Updated size to include 4 shorts
n_sigmas = 3

class CSIProcessor:
    def __init__(self):
        pass

    def _parse_csi_data(self, data):
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
            return csi_data, send_timestamp
            
        except Exception as e:
            # Simplified error handling, keep main thread running
            print(f"Data parsing error: {str(e)}")
            raise

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
        amps = amps / ref_csi
        
        # 提取前56个子载波
        amps = amps[:, :, :, :subcarriers_to_process]
        # subcarriers_data = amps.copy()
        
        # 计算相位，形状为(N, R, T, M)
        phases = np.angle(csi_complex)
        phases = phases[:, :, :, :subcarriers_to_process]
        # phases = np.unwrap(phases,axis=-1)

        # ======== Step 2: 构造环形差分（闭环）========
        # Δ01 = φ0 - φ1, Δ12 = φ1 - φ2, Δ20 = φ2 - φ0
        # 使用向量化操作计算所有发射天线的环形差分
        # 直接在原始维度上操作 (N, R, T, M)
        phi_r0_recon = phases[:, 0, :, :] - phases[:, 1, :, :]  # (N, T, M)
        phi_r1_recon = phases[:, 1, :, :] - phases[:, 2, :, :]  # (N, T, M)
        phi_r2_recon = phases[:, 2, :, :] - phases[:, 0, :, :]  # (N, T, M)

        # phi_r0_recon = np.angle(np.exp(1j * phi_r0_recon))
        # phi_r1_recon = np.angle(np.exp(1j * phi_r1_recon))
        # phi_r2_recon = np.angle(np.exp(1j * phi_r2_recon))

        # ======== Step 3: 重构相位 ========
        # 重构所有天线的相位（基于 r0 为虚拟参考）
        # 这里我们直接使用差分相位作为重构相位
        
        # 重新排列差分相位为 (N, R, T, M) 格式
        reconstructed_phases = np.stack([phi_r0_recon, phi_r1_recon, phi_r2_recon], axis=1)  # (N, R, T, M)
        reconstructed_phases = (reconstructed_phases + np.pi) % (2 * np.pi) - np.pi
        # reconstructed_phases = self.process_multiantenna_phase(reconstructed_phases, smooth_window=9)
        # 对差分相位进行解缠绕
        # reconstructed_phases = np.unwrap(reconstructed_phases, axis=-1)

        # ======== Step 4: 幅度滤波 ========
        # 向量化Hampel滤波处理所有发射天线
        # 重塑amps为 (N*R*T, M) 以适应滤波器
        amps_reshaped = amps.reshape(N, -1)  # (N*R*T, M)
        amps_filtered = self.vectorized_hampel_filter(amps_reshaped, window_size=11, n_sigmas=n_sigmas)
        # 恢复形状 (N, R, T, M)
        amps_filtered = amps_filtered.reshape(N, R, T, subcarriers_to_process)
        amplitude_data = amps_filtered
        
        # 向量化Hampel滤波处理所有重构相位
        # 重塑reconstructed_phases为 (N*R*T, M) 以适应滤波器
        phases_reshaped = reconstructed_phases.reshape(N, -1)  # (N*R*T, M)
        phases_filtered = self.vectorized_hampel_filter(phases_reshaped, window_size=11, n_sigmas=n_sigmas)
        # phases_filtered = np.unwrap(phases_filtered,axis=-1)
        # 恢复形状 (N, R, T, M)
        phase_data = phases_filtered.reshape(N, R, T, subcarriers_to_process)
        phase_data = (phase_data) % (np.pi)
        # phase_data = (phase_data + np.pi) % (2 * np.pi) - np.pi
        # phase_data = phases_filtered

        return amplitude_data, phase_data

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
