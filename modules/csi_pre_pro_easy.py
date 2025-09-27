
NUM_RX_ANTENNAS = 3
NUM_TX_ANTENNAS = 2
NUM_SUBCARRIERS = 114

import numpy as np

class CSIProcessor:
    """CSI信号处理器类，提供CSI数据的预处理、特征提取和降维功能"""

    def __init__(self, raw_data, hampel_window_size, hampel_threshold):
        """初始化CSI处理器"""
        # 可配置参数
        self.hampel_window_size = hampel_window_size       # Hampel滤波窗口大小
        self.hampel_threshold = hampel_threshold        # Hampel滤波阈值
        self.raw_data = raw_data        # 原始CSI数据
        
        # 存储处理后的数据
        self.preprocessed_amplitude = None # 预处理后的幅度数据
        self.preprocessed_phase = None # 预处理后的相位差数据

    def preprocess_csi_show(self, mode="ref"):
        """
        对CSI信号数据进行预处理：
        1. 幅度: 归一化 + 标准化
        - mode="ref"   : 用单个参考子载波做归一化 (传统Atheros常用)
        - mode="smooth": 用参考子载波序列平滑后做归一化 (去掉快速波动)
        - mode="global": 用全局子载波中值做归一化 (更鲁棒，但可能抹平动作)
        2. 相位: 计算相位差 (相对tx=0)，消除CFO/STO
        
        返回:
            amplitude_data: (time, rx, tx, sc) 标准化后的幅度
            phase_data: (time, rx, tx-1, sc) 相位差
        """

        # 获取数据形状
        time_windows, rx_num, tx_num, sc_num, _ = self.raw_data.shape

        amplitude_data = np.zeros((time_windows, rx_num, tx_num, sc_num))
        phase_data = np.zeros((time_windows, rx_num, tx_num, sc_num))

        # === 步骤1: 提取复数形式的CSI ===
        complex_csi = self.raw_data[..., 0] + 1j * self.raw_data[..., 1]

        # === 幅度归一化 ===
        if mode == "ref":
            # 用单个参考子载波
            ref_idx = sc_num // 2
            ref_series = np.abs(complex_csi[:, :, :, ref_idx])  # (time, rx, tx)
            for rx in range(rx_num):
                for tx in range(tx_num):
                    for sc in range(sc_num):
                        series = np.abs(complex_csi[:, rx, tx, sc])
                        relative = series / (ref_series[:, rx, tx] + 1e-10)
                        amplitude_data[:, rx, tx, sc] = relative

        # === 步骤2: 相位差计算 (接收天线环形差分) ===
        for t in range(time_windows):
            for tx in range(tx_num):
                for rx in range(rx_num):
                    # 计算环形差分：当前天线与下一个天线的相位差
                    # 对于最后一个天线，与第一个天线形成环形差分
                    next_rx = (rx + 1) % rx_num
                    phase_current = np.unwrap(np.angle(complex_csi[t, rx, tx, :]), axis=0)      # 当前天线相位
                    phase_next = np.unwrap(np.angle(complex_csi[t, next_rx, tx, :]), axis=0)    # 下一个天线相位

                    # 计算相位差，并规范到[-π, π]区间
                    phase_diff = np.angle(np.exp(1j * (phase_current - phase_next)))
                    
                    # 存储相位差数据 - 需要调整输出数组的形状
                    phase_data[t, rx, tx, :] = phase_diff

        self.preprocessed_amplitude = amplitude_data
        self.preprocessed_phase = phase_data

        return amplitude_data, phase_data
    
    def process_multiantenna_phase(self, smooth_window=10):
        """
        Atheros 相位校准 (高效版)
        - 向量化线性拟合
        - 时间平滑 CFO/STO 参数
        """
        phase_data = self.preprocessed_phase  # (T, n_rx, n_tx_diff, n_sc)
        T, n_rx, n_tx_diff, n_sc = phase_data.shape

        # 去掉边缘子载波
        margin = 0
        valid_idx = np.arange(margin, n_sc - margin)

        X = np.vstack([valid_idx, np.ones_like(valid_idx)]).T  # (n_sc_valid, 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        X_pinv = XtX_inv @ X.T  # (2, n_sc_valid)

        calibrated_phase = np.zeros_like(phase_data)

        for rx in range(n_rx):
            for tx in range(n_tx_diff):
                # 相位矩阵: (T, n_sc)
                antenna_phase = phase_data[:, rx, tx, :]
                unwrapped = np.unwrap(antenna_phase[:, valid_idx], axis=1)  # 沿子载波方向解缠绕

                # === 向量化线性拟合 ===
                # coeffs: (T, 2) → 每帧的 [a, b]
                coeffs = (X_pinv @ unwrapped.T).T  # (T, 2)

                a, b = coeffs[:, 0], coeffs[:, 1]

                # === 时间平滑 ===
                if smooth_window > 1:
                    kernel = np.ones(smooth_window) / smooth_window
                    a = np.convolve(a, kernel, mode="same")
                    b = np.convolve(b, kernel, mode="same")

                # === 去除线性项 ===
                correction = np.outer(a, valid_idx) + b[:, None]  # (T, n_sc_valid)
                corrected = unwrapped - correction
                antenna_phase[:, valid_idx] = np.angle(np.exp(1j * corrected))

                calibrated_phase[:, rx, tx, :] = antenna_phase

        return calibrated_phase
    
    def hampel_filter_phase(self, phase_data, window_size=10):
        """专为相位数据设计的Hampel滤波器"""
        # 先解卷绕，避免±π边界跳变
        unwrapped = np.unwrap(phase_data, axis=0)
        # 应用常规Hampel滤波
        filtered_unwrapped = self.vectorized_hampel_filter(unwrapped, window_size)
        # 重新折回[-π,π]范围
        filtered_unwrapped = np.angle(np.exp(1j * filtered_unwrapped))
        return filtered_unwrapped
    
    
    def vectorized_hampel_filter(self, data, window_size=11):
        """
        向量化的Hampel滤波器实现，用于提高处理速度
        
        参数:
        data: 输入数据数组 (数据包数, 子载波数)
        window_size: 滑动窗口大小
        
        返回:
        filtered_data: 过滤后的数据
        """
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
        n_sigmas = self.hampel_threshold
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

    def do_process(self):
        # Step 1. 幅度 & 相位预处理
        amplitude_data, phase_data = self.preprocess_csi_show()
        T, rx_num, tx_num, sc_num = amplitude_data.shape

        # 1. reshape 为 [T, rx*tx*sc]
        reshaped_data = amplitude_data.reshape(T, -1)

        # 2. 对每一列应用 Hampel 滤波
        # vectorized_hampel_filter 需要能处理 [T, n] 输入
        filtered_reshaped = self.vectorized_hampel_filter(reshaped_data)

        # 3. reshape 回原来的四维
        filtered_amplitude = filtered_reshaped.reshape(T, rx_num, tx_num, sc_num)

        # Step 5. 相位处理 & 滤波
        phase_corrected_data = self.process_multiantenna_phase()
        # filtered_data_phase = self.clean_phase(phase_corrected_data, amp_data=filtered_amplitude)
        phase_reshaped = phase_corrected_data.reshape(T, -1)
        filtered_spectrum_phase = self.hampel_filter_phase(phase_reshaped)
        filtered_data_phase = filtered_spectrum_phase.reshape(T, rx_num, tx_num, sc_num)

        return filtered_amplitude, filtered_data_phase
    

