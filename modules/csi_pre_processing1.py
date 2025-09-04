#!/usr/bin/env python3
"""
CSI信号预处理和分类模块
提供CSI信号归一化、特征提取和分类功能

功能包括：
1. CSI信号预处理 - 幅度归一化和相位差计算
2. 异常值检测与滤波 - Hampel滤波器
3. 降维处理 - PCA分析与可视化
4. 特征选择 - 基于方差的子载波选择
5. 数据分段 - 用于动作分类的时间片段提取

日期: 2025-08-30
"""

NUM_RX_ANTENNAS = 3
NUM_TX_ANTENNAS = 2
NUM_SUBCARRIERS = 114

import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import namedtuple
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter
import os

class CSIProcessor:
    """CSI信号处理器类，提供CSI数据的预处理、特征提取和降维功能"""

    def __init__(self, raw_data, pca_variance_threshold, hampel_window_size, hampel_threshold, top_n_subcarriers):
        """初始化CSI处理器"""
        # 可配置参数
        self.pca_variance_threshold = pca_variance_threshold  # PCA保留方差比例
        self.hampel_window_size = hampel_window_size       # Hampel滤波窗口大小
        self.hampel_threshold = hampel_threshold        # Hampel滤波阈值
        self.top_n_subcarriers = top_n_subcarriers        # 选择的子载波数量
        self.segment_length = 100          # 分段长度
        self.raw_data = raw_data        # 原始CSI数据
        
        # 存储处理结果
        self.selected_subcarriers = None   # 选择的子载波索引
        self.pca_model = None              # 训练好的PCA模型
        
        # 存储处理后的数据
        self.preprocessed_amplitude = None # 预处理后的幅度数据
        self.preprocessed_phase = None # 预处理后的相位差数据
        
        # PCA结果
        self.PCAResult = namedtuple('PCAResult', ['transformed_data', 'explained_variance_ratio', 
                                                'n_components', 'original_shape'])


    def preprocess_csi(self, mode="ref"):
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
        print(f"开始CSI数据预处理... (mode={mode})")

        # 获取数据形状
        time_windows, rx_num, tx_num, sc_num, _ = self.raw_data.shape

        amplitude_data = np.zeros((time_windows, rx_num, tx_num, sc_num))
        phase_data = np.zeros((time_windows, rx_num, tx_num - 1, sc_num))

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
                        amplitude_data[:, rx, tx, sc] = (relative - np.mean(relative)) / (np.std(relative) + 1e-10)

        elif mode == "smooth":
            # 平滑参考子载波 (所有天线平均 + 滑动窗口)
            ref_idx = sc_num // 2
            ref_series = np.abs(complex_csi[:, :, :, ref_idx]).mean(axis=(1, 2))  # (time,)
            smooth_w = 51
            kernel = np.ones(smooth_w) / smooth_w
            ref_smooth = np.convolve(ref_series, kernel, mode='same') + 1e-10
            for rx in range(rx_num):
                for tx in range(tx_num):
                    for sc in range(sc_num):
                        series = np.abs(complex_csi[:, rx, tx, sc])
                        relative = series / ref_smooth
                        amplitude_data[:, rx, tx, sc] = (relative - np.mean(relative)) / (np.std(relative) + 1e-10)

        elif mode == "global":
            # 用全局子载波中值做归一化
            global_ref = np.median(np.abs(complex_csi), axis=(1, 2, 3))  # (time,)
            for rx in range(rx_num):
                for tx in range(tx_num):
                    for sc in range(sc_num):
                        series = np.abs(complex_csi[:, rx, tx, sc])
                        relative = series / (global_ref + 1e-10)
                        amplitude_data[:, rx, tx, sc] = (relative - np.mean(relative)) / (np.std(relative) + 1e-10)

        else:
            raise ValueError(f"未知的 mode={mode}，请选择 'ref' / 'smooth' / 'global'")

        # === 步骤2: 相位差计算 (相对tx=0) ===
        for t in range(time_windows):
            for rx in range(rx_num):
                phase_ref = np.angle(complex_csi[t, rx, 0, :])  # 发射天线0
                for tx in range(1, tx_num):
                    phase_current = np.angle(complex_csi[t, rx, tx, :])
                    phase_diff = np.angle(np.exp(1j * (phase_current - phase_ref)))
                    phase_data[t, rx, tx - 1, :] = phase_diff

        self.preprocessed_amplitude = amplitude_data
        self.preprocessed_phase = phase_data

        print(f"预处理完成 - 幅度数据形状: {amplitude_data.shape}")
        print(f"预处理完成 - 相位数据形状: {phase_data.shape}")
        return amplitude_data, phase_data

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
        print(f"开始CSI数据预处理... (mode={mode})")

        # 获取数据形状
        time_windows, rx_num, tx_num, sc_num, _ = self.raw_data.shape

        amplitude_data = np.zeros((time_windows, rx_num, tx_num, sc_num))
        phase_data = np.zeros((time_windows, rx_num, tx_num - 1, sc_num))

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
                        # amplitude_data[:, rx, tx, sc] = (relative - np.mean(relative)) / (np.std(relative) + 1e-10)


        elif mode == "smooth":
            # 平滑参考子载波 (所有天线平均 + 滑动窗口)
            ref_idx = sc_num // 2
            ref_series = np.abs(complex_csi[:, :, :, ref_idx]).mean(axis=(1, 2))  # (time,)
            smooth_w = 51
            kernel = np.ones(smooth_w) / smooth_w
            ref_smooth = np.convolve(ref_series, kernel, mode='same') + 1e-10
            for rx in range(rx_num):
                for tx in range(tx_num):
                    for sc in range(sc_num):
                        series = np.abs(complex_csi[:, rx, tx, sc])
                        relative = series / ref_smooth
                        amplitude_data[:, rx, tx, sc] = (relative - np.mean(relative)) / (np.std(relative) + 1e-10)

        elif mode == "global":
            # 用全局子载波中值做归一化
            global_ref = np.median(np.abs(complex_csi), axis=(1, 2, 3))  # (time,)
            for rx in range(rx_num):
                for tx in range(tx_num):
                    for sc in range(sc_num):
                        series = np.abs(complex_csi[:, rx, tx, sc])
                        relative = series / (global_ref + 1e-10)
                        amplitude_data[:, rx, tx, sc] = (relative - np.mean(relative)) / (np.std(relative) + 1e-10)

        else:
            raise ValueError(f"未知的 mode={mode}，请选择 'ref' / 'smooth' / 'global'")

        # === 步骤2: 相位差计算 (相对tx=0) ===
        for t in range(time_windows):
            for rx in range(rx_num):
                phase_ref = np.angle(complex_csi[t, rx, 0, :])  # 发射天线0
                for tx in range(1, tx_num):
                    phase_current = np.angle(complex_csi[t, rx, tx, :])
                    phase_diff = np.angle(np.exp(1j * (phase_current - phase_ref)))
                    phase_data[t, rx, tx - 1, :] = phase_diff

        self.preprocessed_amplitude = amplitude_data
        self.preprocessed_phase = phase_data

        print(f"预处理完成 - 幅度数据形状: {amplitude_data.shape}")
        print(f"预处理完成 - 相位数据形状: {phase_data.shape}")
        return amplitude_data, phase_data
    
    def hampel_filter_phase(self, phase_data, window_size=10, n_sigmas=2.0):
        """专为相位数据设计的Hampel滤波器"""
        # 先解卷绕，避免±π边界跳变
        unwrapped = np.unwrap(phase_data, axis=1)
        # 应用常规Hampel滤波
        filtered_unwrapped = self.vectorized_hampel_filter(unwrapped, window_size, n_sigmas)
        # 重新折回[-π,π]范围
        return np.angle(np.exp(1j * filtered_unwrapped))
    
    def hampel_filter_fast(self, x, window_size=7, n_sigmas=3):
        """
        高效向量化的 Hampel 滤波器
        
        参数:
            x: 输入数据 (1D numpy array)
            window_size: 滑动窗口大小 (奇数)
            n_sigmas: 判定异常的标准差倍数
        
        返回:
            y: 滤波后的数据 (1D numpy array)
        """
        n = len(x)
        if window_size % 2 == 0:
            window_size += 1
        k = window_size // 2

        # 生成滑动窗口矩阵 (n - window_size + 1, window_size)
        windows = sliding_window_view(x, window_size)

        # 每个窗口的中位数
        medians = np.median(windows, axis=1)

        # MAD (median absolute deviation)
        mad = np.median(np.abs(windows - medians[:, None]), axis=1)
        sigma = 1.4826 * mad

        # 扩展中位数和sigma到与x同长度
        medians_full = np.pad(medians, (k, k), mode="edge")
        sigma_full = np.pad(sigma, (k, k), mode="edge")

        # 找出异常点
        diff = np.abs(x - medians_full)
        outliers = diff > n_sigmas * sigma_full

        # 替换异常点为窗口中位数
        y = x.copy()
        y[outliers] = medians_full[outliers]

        return y
    
    def hampel_filter_csi(self, data, window_size=7, n_sigmas=0.6):
        """
        对 CSI 数据批量应用 Hampel 滤波
        data: (T, rx, tx, sc) 或 (T, sc)
        """
        shape = data.shape
        T, *rest = shape
        reshaped = data.reshape(T, -1)  # (T, features)
        
        filtered = np.zeros_like(reshaped)
        for i in range(reshaped.shape[1]):
            filtered[:, i] = self.hampel_filter_fast(reshaped[:, i], window_size, n_sigmas)

        return filtered.reshape(shape)
    
    def select_best_subcarriers(self, filtered_data, n_select=15):
        """选择最佳子载波，基于信噪比和信号稳定性"""
        T, rx_num, tx_num, sc_num = filtered_data.shape
        
        # 计算每个子载波的质量指标
        quality_scores = np.zeros(sc_num)
        
        for sc in range(sc_num):
            # 提取所有天线对的该子载波数据
            sc_data = filtered_data[:, :, :, sc]
            
            # 1. 计算时域平滑度（使用一阶差分）
            diff = np.abs(np.diff(sc_data, axis=0))
            smoothness = 1.0 / (np.mean(diff) + 1e-10)  # 平滑度越高，diff越小
            
            # 2. 计算信噪比估计（信号能量/噪声能量）
            # 使用低通滤波作为信号估计，高通滤波作为噪声估计
            from scipy import signal
            signal_power = 0
            noise_power = 0
            
            for rx in range(rx_num):
                for tx in range(tx_num):
                    time_series = sc_data[:, rx, tx]
                    # 低通滤波 - 信号估计
                    b, a = signal.butter(3, 0.1)
                    signal_est = signal.filtfilt(b, a, time_series)
                    # 高通滤波 - 噪声估计
                    b, a = signal.butter(3, 0.5, 'highpass')
                    noise_est = signal.filtfilt(b, a, time_series)
                    
                    signal_power += np.mean(signal_est**2)
                    noise_power += np.mean(noise_est**2)
            
            snr = signal_power / (noise_power + 1e-10)
            
            # 3. 综合评分
            quality_scores[sc] = smoothness * snr
        
        # 选择评分最高的子载波
        selected_indices = np.argsort(quality_scores)[-n_select:]
        return np.sort(selected_indices)
    
    def vectorized_hampel_filter(self, data, window_size=11, n_sigmas=3):
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
    
    def hampel_filter(self, restored_data, window_size=None, threshold=None):
        """
        应用Hampel滤波器剔除峰值和异常值
            
        处理过程:
            1. 使用长度为window_size的滑动窗口
            2. 计算窗口内的数据中值
            3. 计算每个点与窗口中值的偏差
            4. 计算中位数绝对偏差(MAD)并转换为等效标准差
            5. 如果偏差超过阈值*标准差，则用窗口中值替换该点
        
        返回:
            滤波后的数据
        """
        data = restored_data
        window_size = self.hampel_window_size if window_size is None else window_size
        threshold = self.hampel_threshold if threshold is None else threshold

        n = len(data)
        filtered_data = data.copy()
        replaced_count = 0
        
        for i in range(n):
            # 确定窗口边界 (保证窗口居中于当前点)
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            
            # 提取窗口数据
            window_data = data[start:end]
            
            # 计算窗口内的中位数
            median = np.median(window_data)
            
            # 计算窗口内每个点与中位数的绝对偏差
            abs_deviations = np.abs(window_data - median)
            
            # 计算中位数绝对偏差 (MAD)
            mad = np.median(abs_deviations)
            
            # 使用常数因子将MAD转换为等效高斯标准差
            # 对于正态分布，MAD与标准差的关系为: sigma ≈ 1.4826 * MAD
            sigma_est = 1.4826 * mad
            
            # 防止除零错误 (如果所有值都一样，MAD会为0)
            if sigma_est < 1e-10:
                # 备用方案：使用标准方差，并确保至少有个很小的正值
                sigma_est = max(np.std(window_data), 1e-10)
                
            # 检测异常值：如果当前点与中位数的差值超过阈值*标准差，判定为异常值
            if abs(data[i] - median) > threshold * sigma_est:
                filtered_data[i] = median  # 用窗口中位数替换异常值
                replaced_count += 1
        
        # # 打印替换信息
        # if replaced_count > 0:
        #     replacement_percent = replaced_count / n * 100
        #     print(f"Hampel滤波: 替换了{replaced_count}个异常值 ({replacement_percent:.2f}%)")
            
        return filtered_data

    def apply_pca(self, filtered_data):
        """
        对输入数据应用PCA降维
        
        返回:
            PCAResult对象，包含变换后的数据、解释方差比等信息
        """
        amp_data = filtered_data
        original_shape = amp_data.shape
        data = amp_data.reshape(original_shape[0], -1)  # [时间窗口, 特征数]
        variance_threshold = self.pca_variance_threshold
            
        print(f"应用PCA降维，保留{variance_threshold*100:.1f}%的方差...")
        
        # 应用PCA
        pca = PCA(n_components=variance_threshold)
        transformed_data = pca.fit_transform(data)

        # 存储PCA模型
        self.pca_model = pca
        
        # 创建结果对象
        result = self.PCAResult(
            transformed_data=transformed_data,
            explained_variance_ratio=np.cumsum(pca.explained_variance_ratio_),
            n_components=pca.n_components_,
            original_shape=original_shape
        )
        
        print(f"PCA降维完成，从{data.shape[1]}维降至{result.n_components}维")
        print(f"累计解释方差: {result.explained_variance_ratio[-1]*100:.2f}%")
        
        return result

    def detailed_phase_calibration(self, phi_matrix, subcarrier_indices):
        """
        详细的相位校准实现
        """
        # 获取数据维度
        T, n_subcarriers = phi_matrix.shape
        
        # 初始化校准后的相位矩阵
        calibrated_phase = np.zeros_like(phi_matrix)
        
        # 存储校准参数用于分析
        a_values = np.zeros(T)  # 斜率参数
        b_values = np.zeros(T)  # 截距参数
        
        # 对每个时间点进行处理
        for t in range(T):
            # 获取当前时间点的所有子载波相位
            current_phase = phi_matrix[t, :]
            unwrapped_phase = np.unwrap(current_phase)
            # print(f"\n--- 时间点 {t} 的相位校准 ---")
            # print(f"原始相位: {current_phase[:5]}...")  # 显示前5个值

            # 更健壮的方法：使用RANSAC拟合，避免异常值影响
            from sklearn.linear_model import RANSACRegressor
            from sklearn.linear_model import LinearRegression
            
            X = subcarrier_indices.reshape(-1, 1)
            ransac = RANSACRegressor(LinearRegression())
            ransac.fit(X, unwrapped_phase)
            
            # 获取线性参数
            a = ransac.estimator_.coef_[0]  # 斜率
            b = ransac.estimator_.intercept_  # 截距
            
            # 应用校正
            corrected = unwrapped_phase - (a * subcarrier_indices + b)
            calibrated_phase[t, :] = np.angle(np.exp(1j * corrected))
            # # 步骤1: 计算斜率参数a（消除STO）
            # # 使用第一个和最后一个子载波的相位和索引
            # first_idx = 0
            # last_idx = -1
            
            # phase_first = unwrapped_phase[first_idx]
            # phase_last = unwrapped_phase[last_idx]
            # index_first = subcarrier_indices[first_idx]
            # index_last = subcarrier_indices[last_idx]
            
            # a = (phase_last - phase_first) / (index_last - index_first)
            # a_values[t] = a

            # # 步骤2: 计算截距参数b（消除CFO）
            # b = np.mean(unwrapped_phase)
            # b_values[t] = b
            # # print(f"相位平均值 b = {b:.3f}")
            
            # # 步骤3: 应用线性校准
            # for k in range(n_subcarriers):
            #     calibrated_phase[t, k] = unwrapped_phase[k] - a * subcarrier_indices[k] - b
            # # 可选：将校准后的相位折回[-π, π]范围
            # calibrated_phase[t, :] = np.angle(np.exp(1j * calibrated_phase[t, :]))
            # # print(f"校准后的相位: {calibrated_phase[t, :5]}...")

            # # 最小二乘拟合一阶多项式：unwrapped ~= a * idx + b
            # coeffs = np.polyfit(subcarrier_indices, unwrapped_phase, 1)
            # a = coeffs[0]; b = coeffs[1]
            # a_values[t] = a; b_values[t] = b
            # corrected = unwrapped_phase - (a * subcarrier_indices + b)
            # calibrated_phase[t, :] = np.angle(np.exp(1j * corrected))
            # print(f"使用的子载波索引: {index_first} 和 {index_last}")
            # print(f"对应相位值: {phase_first:.3f} 和 {phase_last:.3f}")
            # print(f"计算得到的斜率 a = {a:.6f}")

        return calibrated_phase, a_values, b_values

    def process_multiantenna_phase(self, smooth_window=9):
        """
        Atheros 相位校准 (高效版)
        - 向量化线性拟合
        - 时间平滑 CFO/STO 参数
        """
        phase_data = self.preprocessed_phase  # (T, n_rx, n_tx_diff, n_sc)
        T, n_rx, n_tx_diff, n_sc = phase_data.shape

        # 去掉边缘子载波
        margin = n_sc // 6
        valid_idx = np.arange(margin, n_sc - margin)

        X = np.vstack([valid_idx, np.ones_like(valid_idx)]).T  # (n_sc_valid, 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        X_pinv = XtX_inv @ X.T  # (2, n_sc_valid)

        calibrated_phase = np.zeros_like(phase_data)

        print(f"开始 Atheros 相位校准: 有效子载波 {len(valid_idx)} 个")

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
    
    def clean_phase(self, phase_data, amp_data=None,
                low_amp_threshold=0.05, window_size=11, n_sigmas=3.0,
                time_smooth=True):
        """
        改进版相位滤波
        """
        phase_data = np.asarray(phase_data)
        T, n_rx, n_tx, n_sc = phase_data.shape
        filtered = np.zeros_like(phase_data)

        for rx in range(n_rx):
            for tx in range(n_tx):
                phi = phase_data[:, rx, tx, :]

                # (1) 幅度屏蔽
                if amp_data is not None:
                    amp = amp_data[:, rx, tx, :]
                    mask = amp < low_amp_threshold
                    phi = phi.copy()
                    phi[mask] = np.nan

                    # 插值 NaN
                    for sc in range(n_sc):
                        if np.isnan(phi[:, sc]).any():
                            idx = np.arange(T)
                            good = ~np.isnan(phi[:, sc])
                            if good.any():
                                phi[:, sc] = np.interp(idx, idx[good], phi[good, sc])
                            else:
                                phi[:, sc] = 0.0

                # (2) 解缠绕（沿子载波）
                unwrapped = np.unwrap(phi, axis=1)

                # (3) Hampel 去掉孤立点
                unwrapped[:, :] = self.hampel_filter_phase(
                    unwrapped[:, :], window_size=window_size, n_sigmas=n_sigmas)

                # (4) 可选：时间方向平滑
                if time_smooth:
                    unwrapped = savgol_filter(unwrapped, 11, 2, axis=0)

                # (5) wrap 回 [-π, π]
                filtered[:, rx, tx, :] = np.angle(np.exp(1j * unwrapped))

        return filtered
    # 对处理后的幅度数据提取更好的特征
    # def extract_statistical_features(amplitude_data):
    #     # 提取时域统计特征
    #     mean = np.mean(amplitude_data, axis=0)
    #     std = np.std(amplitude_data, axis=0)
    #     skew = stats.skew(amplitude_data, axis=0)
    #     kurt = stats.kurtosis(amplitude_data, axis=0)
        
    #     # 提取频域特征
    #     fft_features = np.abs(np.fft.fft(amplitude_data, axis=0))[:20]
        
    #     # 组合特征
    #     return np.concatenate([mean, std, skew, kurt, fft_features.flatten()])
    def do_process(self):
        # Step 1. 幅度 & 相位预处理
        amplitude_data, phase_data = self.preprocess_csi_show()
        T, rx_num, tx_num, sc_num = amplitude_data.shape
        print(f"输入数据形状: {amplitude_data.shape}")
            
            
        filtered_amplitude = np.zeros_like(amplitude_data)
        # filtered_amplitude = self.hampel_filter_timeseries(amplitude_data)
        # 对每个天线对的子载波序列应用Hampel滤波器
        for rx in range(rx_num):
            for tx in range(tx_num):
                # print(f"应用Hampel滤波: 接收天线={rx}, 发送天线={tx}")
                
                # 两种滤波方式：
                # 1. 对每个子载波的时间序列进行滤波 (检测时间轴上的异常)
                # for sc in range(sc_num):
                #     # 提取特定子载波在所有时间点的数据
                #     time_series = pca_restored[:, rx, tx, sc]
                #     # 应用Hampel滤波 - 时间序列滤波
                #     filtered_series = self.hampel_filter(time_series, window_size, threshold)
                #     # 存储滤波后的数据
                #     filtered_data[:, rx, tx, sc] = filtered_series
                spectrum = amplitude_data[:, rx, tx, :]
                filtered_spectrum = self.vectorized_hampel_filter(spectrum)
                # 存储滤波后的数据
                filtered_amplitude[:, rx, tx, :] = filtered_spectrum
        
        # Step 5. 相位处理 & 滤波
        print("进行相位校正 + Hampel 滤波...")
        phase_corrected_data = self.process_multiantenna_phase()
        # filtered_data_phase = self.clean_phase(phase_corrected_data, amp_data=filtered_amplitude)
        filtered_data_phase = np.zeros_like(phase_corrected_data)

        for rx in range(phase_corrected_data.shape[1]):
            for tx in range(phase_corrected_data.shape[2]):
                # print(f"应用Hampel滤波: 接收天线={rx}, 发送天线={tx}")
                
                # 两种滤波方式：
                # 1. 对每个子载波的时间序列进行滤波 (检测时间轴上的异常)
                # for sc in range(sc_num):
                #     # 提取特定子载波在所有时间点的数据
                #     time_series = pca_restored[:, rx, tx, sc]
                #     # 应用Hampel滤波 - 时间序列滤波
                #     filtered_series = self.hampel_filter(time_series, window_size, threshold)
                #     # 存储滤波后的数据
                #     filtered_data[:, rx, tx, sc] = filtered_series
                spectrum_phase = phase_corrected_data[:, rx, tx, :]
                filtered_spectrum_phase = self.hampel_filter_phase(spectrum_phase)
                filtered_data_phase[:, rx, tx, :] = filtered_spectrum_phase


        # 数据分段
        # segmented_features = self.segment_data(selected_features)

        return filtered_amplitude, filtered_data_phase
    def do_extract_features(self):
        """
        从CSI数据中提取特征
        
        参数:
            amplitude_data: 幅度数据，形状为 [时间窗口, 天线数, 发射天线数, 子载波数]
            phase_diff_data: 相位差数据，形状为 [时间窗口, 天线数-1, 发射天线数, 子载波数]
            top_n_subcarriers: 选择的子载波数量，若为None则使用默认值
            window_size: Hampel滤波窗口大小，若为None则使用默认值
            threshold: Hampel滤波阈值，若为None则使用默认值
            
        返回:
            tuple: (特征数据, 选择的子载波索引)
        """
        # Step 1. 幅度 & 相位预处理
        amplitude_data, phase_data = self.preprocess_csi()
        T, rx_num, tx_num, sc_num = amplitude_data.shape
        print(f"输入数据形状: {amplitude_data.shape}")

        
        top_n_subcarriers = self.top_n_subcarriers
            
            
        print("开始特征提取...")

        # 步骤3: 对重构数据应用Hampel滤波
        print("步骤3: 应用Hampel滤波剔除异常值")
        filtered_amplitude = np.zeros_like(amplitude_data)
        # filtered_amplitude = self.hampel_filter_timeseries(amplitude_data)
        # 对每个天线对的子载波序列应用Hampel滤波器
        for rx in range(rx_num):
            for tx in range(tx_num):
                print(f"应用Hampel滤波: 接收天线={rx}, 发送天线={tx}")
                
                # 两种滤波方式：
                # 1. 对每个子载波的时间序列进行滤波 (检测时间轴上的异常)
                # for sc in range(sc_num):
                #     # 提取特定子载波在所有时间点的数据
                #     time_series = pca_restored[:, rx, tx, sc]
                #     # 应用Hampel滤波 - 时间序列滤波
                #     filtered_series = self.hampel_filter(time_series, window_size, threshold)
                #     # 存储滤波后的数据
                #     filtered_data[:, rx, tx, sc] = filtered_series
                spectrum = amplitude_data[:, rx, tx, :]
                filtered_spectrum = self.vectorized_hampel_filter(spectrum)
                # 存储滤波后的数据
                filtered_amplitude[:, rx, tx, :] = filtered_spectrum

        # Step 3. 子载波选择（基于方差）
        print("选择方差最大的子载波...")
        subcarrier_variances = np.var(filtered_amplitude, axis=0)  # (rx, tx, sc)
        subcarrier_variances = np.mean(subcarrier_variances, axis=(0,1))  # 平均到所有天线
        selected_indices_am = np.argsort(subcarrier_variances)[-self.top_n_subcarriers:]
        selected_indices_am = np.sort(selected_indices_am)
        self.selected_subcarriers = selected_indices_am
        selected_amplitude = filtered_amplitude[:, :, :, selected_indices_am]
        print(f"选择的子载波索引: {selected_indices_am}")

        # Step 4. PCA降维（直接使用transformed_data，不做inverse）
        print("应用 PCA 提取主成分特征...")
        reshaped_amp = selected_amplitude.reshape(T, -1)  # (样本数, 特征数)
        pca = PCA(n_components=self.pca_variance_threshold)
        pca_features = pca.fit_transform(reshaped_amp)
        self.pca_model = pca
        print(f"PCA降维: 从 {reshaped_amp.shape[1]} → {pca_features.shape[1]} 维")
        print(f"累计解释方差: {np.cumsum(pca.explained_variance_ratio_)[-1]*100:.2f}%")
        
        # Step 5. 相位处理 & 滤波
        print("进行相位校正 + Hampel 滤波...")
        phase_corrected_data = self.process_multiantenna_phase()
        # filtered_data_phase = self.clean_phase(phase_corrected_data, amp_data=filtered_amplitude)
        filtered_data_phase = np.zeros_like(phase_corrected_data)

        for rx in range(phase_corrected_data.shape[1]):
            for tx in range(phase_corrected_data.shape[2]):
                print(f"应用Hampel滤波: 接收天线={rx}, 发送天线={tx}")
                
                # 两种滤波方式：
                # 1. 对每个子载波的时间序列进行滤波 (检测时间轴上的异常)
                # for sc in range(sc_num):
                #     # 提取特定子载波在所有时间点的数据
                #     time_series = pca_restored[:, rx, tx, sc]
                #     # 应用Hampel滤波 - 时间序列滤波
                #     filtered_series = self.hampel_filter(time_series, window_size, threshold)
                #     # 存储滤波后的数据
                #     filtered_data[:, rx, tx, sc] = filtered_series
                spectrum_phase = phase_corrected_data[:, rx, tx, :]
                filtered_spectrum_phase = self.hampel_filter_phase(spectrum_phase)
                filtered_data_phase[:, rx, tx, :] = filtered_spectrum_phase
        # 筛选子载波
        # 步骤4: 选择方差最大的子载波
        print("步骤4: 选择方差最大的子载波")
        subcarrier_variances_phase = np.var(filtered_data_phase, axis=(0, 1, 2))
        # 获取方差最大的子载波索引
        selected_indices_ph = np.argsort(subcarrier_variances_phase)[-top_n_subcarriers:]
        selected_indices_ph = np.sort(selected_indices_ph)  # 按索引排序
        # 保存选择的子载波索引
        self.selected_subcarriers_phase = selected_indices_ph

        # 筛选子载波
        selected_features_phase = filtered_data_phase[:, :, :, selected_indices_ph]

        # 数据分段
        # segmented_features = self.segment_data(selected_features)

        return selected_amplitude, selected_features_phase, selected_indices_am, selected_indices_ph
    
def read_csi_data(filepath, num_rx=3, num_tx=2, num_sc=114):
    """
    从.dat文件中读取CSI数据
    
    参数:
        filepath: CSI数据文件路径
        num_rx: 接收天线数量
        num_tx: 发送天线数量
        num_sc: 子载波数量
    
    返回:
        numpy.ndarray: 形状为 [数据包数, 接收天线数, 发送天线数, 子载波数, 2] 的CSI数据
    """
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件 {filepath} 不存在")
    
    # 读取二进制数据
    print(f"正在读取文件: {filepath}")
    csi_data = np.fromfile(filepath, dtype=np.int16)
    
    # 计算每个数据包的大小
    # 每个复数包含实部和虚部，每个部分是int16类型(2字节)
    complex_per_packet = num_rx * num_tx * num_sc
    values_per_packet = complex_per_packet * 2  # 实部+虚部
    
    # 检查数据大小是否正确
    if len(csi_data) % values_per_packet != 0:
        raise ValueError("数据大小不匹配，文件可能已损坏或参数不正确")
    
    # 计算数据包数量
    num_packets = len(csi_data) // values_per_packet
    print(f"读取到 {num_packets} 个数据包")
    
    # 重塑数据为 [数据包数, 接收天线数, 发送天线数, 子载波数, 2]
    csi_data = csi_data.reshape(num_packets, num_rx, num_tx, num_sc, 2)
    print(f"CSI数据形状: {csi_data.shape}")
    
    return csi_data

def plot_csi_data(csi_before, csi_after, csi_after_phase, rx_ant=0, tx_ant=0, subcarriers=0):
    """绘制处理前后的CSI幅度和相位图"""
    # 检查输入数据的维度
    print(f"处理前数据形状: {csi_before.shape}")
    
    # 判断csi_after是否为元组(可能是extract_features函数返回的元组)
    if isinstance(csi_after, tuple):
        print("处理后数据是元组，提取第一个元素作为特征数据")
        csi_after = csi_after[0]  # 提取元组中的第一个元素(特征矩阵)
    
    print(f"处理后数据形状: {csi_after.shape}")
    
    # 根据数据维度调整处理方式
    has_complex_dim = len(csi_before.shape) == 5  # 检查是否有复数维度
    
    # 处理前数据 (原始CSI数据，5维)
    if has_complex_dim:
        # 处理5维数据 [time, rx, tx, sc, complex]
        csi_subset_before = csi_before[:, rx_ant, tx_ant, subcarriers, :]  # (数据包数, 2)
        csi_complex_before = csi_subset_before[:, 0] + 1j * csi_subset_before[:, 1]
        amplitude_before = np.abs(csi_complex_before)
        phase_before = np.angle(csi_complex_before)
    else:
        # 如果已经是处理过的数据（例如特征数据），可能只有4维 [time, rx, tx, sc]
        amplitude_before = csi_before[:, rx_ant, tx_ant, subcarriers]
        # 相位信息可能不可用，创建零数组
        phase_before = np.zeros_like(amplitude_before)
    
    avg_amplitude_before = amplitude_before  # (数据包数,)
    avg_phase_before = phase_before  # (数据包数,)
    
    # 处理后数据 (可能是特征数据，4维)
    if len(csi_after.shape) == 5:  # 如果仍然是5维
        csi_subset_after = csi_after[:, rx_ant, tx_ant, subcarriers, :]
        csi_complex_after = csi_subset_after[:, 0] + 1j * csi_subset_after[:, 1]
        amplitude_after = np.abs(csi_complex_after)
        phase_after = np.angle(csi_complex_after)
    else:  # 如果是4维
        # 直接提取振幅（特征数据）
        amplitude_after = csi_after[:, rx_ant, tx_ant, subcarriers]
    
    # 处理后数据 (可能是特征数据，4维)
    if len(csi_after_phase.shape) == 5:  # 如果仍然是5维
        csi_subset_after = csi_after_phase[:, rx_ant, tx_ant, subcarriers, :]
        csi_complex_after = csi_subset_after[:, 0] + 1j * csi_subset_after[:, 1]
        amplitude_after = np.abs(csi_complex_after)
        phase_after = np.angle(csi_complex_after)
    else:  # 如果是4维
        # 直接提取振幅（特征数据）
        phase_after = csi_after_phase[:, rx_ant, tx_ant, subcarriers]
    
    avg_amplitude_after = amplitude_after  # (数据包数,)
    avg_phase_after = phase_after  # (数据包数,)


    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    
    # 绘制处理前幅度图
    ax1.plot(avg_amplitude_before)
    ax1.set_title(f'Average Amplitude (Before Processing) - Antenna Pair ({rx_ant}, {tx_ant})')
    ax1.set_xlabel('Packet Index')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # 绘制处理后幅度图
    ax2.plot(avg_amplitude_after)
    ax2.set_title(f'Average Amplitude (After Processing) - Antenna Pair ({rx_ant}, {tx_ant})')
    ax2.set_xlabel('Packet Index')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    # 绘制处理前相位图
    ax3.plot(avg_phase_before)
    ax3.set_title(f'Average Phase (Before Processing) - Antenna Pair ({rx_ant}, {tx_ant})')
    ax3.set_xlabel('Packet Index')
    ax3.set_ylabel('Phase (radians)')
    ax3.grid(True)
    
    # 绘制处理后相位图
    ax4.plot(avg_phase_after)
    ax4.set_title(f'Average Phase (After Processing) - Antenna Pair ({rx_ant}, {tx_ant})')
    ax4.set_xlabel('Packet Index')
    ax4.set_ylabel('Phase (radians)')
    ax4.grid(True)
    
    plt.tight_layout()
    fig.savefig(f"csi_plot_{time.time()}.png")  # 保存为图像文件
    # plt.show()
    plt.close(fig)

# 测试代码
def test_csi_processor():
    """测试CSI处理器的功能"""
    print("=== CSI处理器测试 ===")

    raw_data = read_csi_data('csi_data_20250902_120150.dat', NUM_RX_ANTENNAS, NUM_TX_ANTENNAS, NUM_SUBCARRIERS)
    # for i in range(10):
    #     # 创建CSI处理器
    #     processor = CSIProcessor(raw_data[i*400:(i+1)*400,:,:,:56], 0.9, 10, 0.6, 15)

    #     selected_features, selected_features_phase = processor.do_extract_features()

        
    #     # selected_features, selected_features_phase = processor.preprocess_csi()

    #     # 使用正确的参数调用plot_csi_data
    #     plot_csi_data(raw_data[i*400:(i+1)*400,:,:,:56], selected_features, selected_features_phase, rx_ant=0, tx_ant=0, subcarriers=3)
    # 创建CSI处理器
    processor = CSIProcessor(raw_data[:,:,:,:56], 0.9, 11, 3, 15)

    selected_features, selected_features_phase = processor.do_process()
    # selected_features, selected_features_phase = processor.do_extract_features()

    # 使用正确的参数调用plot_csi_data
    plot_csi_data(raw_data[:,:,:,:56], selected_features, selected_features_phase, rx_ant=0, tx_ant=0, subcarriers=1)
    
    processor = CSIProcessor(raw_data[600:800,:,:,:56], 0.9, 11, 3, 15)
    selected_features, selected_features_phase = processor.do_process()
    plot_csi_data(raw_data[0:,:,:,:56], selected_features, selected_features_phase, rx_ant=0, tx_ant=0, subcarriers=1)

    print("\n测试完成!")
    

if __name__ == "__main__":
    test_csi_processor()
