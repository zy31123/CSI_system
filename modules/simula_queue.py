#!/usr/bin/env python3
"""
Simulation Queue Module
Implements simulated real-time CSI data collection by reading .dat files 
and pushing data sequentially to Redis CSI_SOURCE_QUEUE
"""

import os
import time
import json
import redis
import numpy as np
from threading import Thread, Event
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    REDIS_HOST, REDIS_PORT, CSI_PROCESSED_QUEUE, 
    CSI_VISUALIZATION_CHANNEL, MAX_QUEUE_LENGTH, WINDOW_SIZE, NETWORK_SIZE, NETWORK_OVERLAP, CSI_SOURCE_QUEUE
)
import struct

NUM_SUBCARRIERS = 114
NUM_RX_ANTENNAS = 3
NUM_TX_ANTENNAS = 2
CSI_DATA_SIZE = 2736 + 8  # Updated size to include 4 shorts

def parse_csi_data_from_file(dat_file_path):
    """
    从.dat文件中解析CSI数据
    
    Args:
        dat_file_path: .dat文件路径
        packet_count: 数据包数量
    
    Returns:
        numpy数组，形状为(样本数, 接收天线数, 发送天线数, 子载波数, 2)
    """
    try:
        # 读取二进制数据
        with open(dat_file_path, 'rb') as f:
            data = f.read()
        
        # 计算每个数据包的大小
        num_complex = NUM_SUBCARRIERS * NUM_RX_ANTENNAS * NUM_TX_ANTENNAS
        packet_size = num_complex * 2 * 2 + 8  # 2(实部+虚部) * 2(每个short为2字节) + 8(时间戳)
        packet_count = len(data) // packet_size

        # 验证文件大小
        if len(data) != packet_size * packet_count:
            print(f"警告: 文件大小不匹配。期望 {packet_size * packet_count} 字节, 实际 {len(data)} 字节")
            # 调整packet_count以适应实际数据
            packet_count = len(data) // packet_size
            print(f"调整packet_count为 {packet_count}")
        
        # 初始化结果数组
        csi_data = np.zeros((packet_count, NUM_RX_ANTENNAS, NUM_TX_ANTENNAS, NUM_SUBCARRIERS, 2), dtype=np.int16)
        send_time = np.zeros(packet_count, dtype=np.uint64)

        results = []
        # 解析每个数据包
        for i in range(packet_count):
            if i * packet_size + packet_size > len(data):
                print(f"警告: 数据不足，只处理 {i} 个数据包")
                csi_data = csi_data[:i]
                break
                
            packet_data = data[i * packet_size:(i + 1) * packet_size]

            time_bytes = packet_data[:8]
            # 使用 '<H' 解析为无符号短整型
            timestamp_parts = struct.unpack('<4H', packet_data[:8])
            send_timestamp = (timestamp_parts[0] << 48) | (timestamp_parts[1] << 32) | (timestamp_parts[2] << 16) | timestamp_parts[3]
            send_timestamp = send_timestamp / 1_000_000.0  # 转换为秒
            send_time[i] = send_timestamp

            # 解析实部和虚部
            for rx in range(NUM_RX_ANTENNAS):
                for tx in range(NUM_TX_ANTENNAS):
                    for sc in range(NUM_SUBCARRIERS):
                        # 计算索引
                        index = rx * NUM_TX_ANTENNAS * NUM_SUBCARRIERS + tx * NUM_SUBCARRIERS + sc
                        
                        # 解析实部 (前半部分)
                        real_offset = index * 2
                        real = struct.unpack('<h', packet_data[real_offset+8:real_offset+2+8])[0]
                        
                        # 解析虚部 (后半部分)
                        imag_offset = (num_complex + index) * 2
                        imag = struct.unpack('<h', packet_data[imag_offset+8:imag_offset+2+8])[0]
                        
                        # 存储数据
                        csi_data[i, rx, tx, sc, 0] = real
                        csi_data[i, rx, tx, sc, 1] = imag

            result = {
                'timestamp': time.time(),              # Receive timestamp (microseconds)
                'send_time': send_timestamp,            # Send timestamp (microseconds, since Jan 1, 2025)
                'csi_data': csi_data[i].tolist()           # Convert to list for JSON serialization
            }
            results.append(result)

        return results

    except Exception as e:
        print(f"解析文件 {dat_file_path} 时出错: {e}")
        return None


class SimulationQueueThread(Thread):
    """
    模拟数据采集线程：依次读取data/zy文件夹下的dat数据，
    并将每个数据依次加入到Redis的CSI_SOURCE_QUEUE队列中
    """
    
    def __init__(self, data_dir="data/zy", interval=0.01):
        """
        初始化模拟队列线程
        
        参数:
            data_dir: dat文件所在目录
            interval: 每个数据包发送间隔（秒）
        """
        super().__init__(name="SimulationQueueThread")
        self.data_dir = data_dir
        self.interval = interval  # 数据包发送间隔
        self.stop_event = Event()
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        
    def run(self):
        """
        主线程函数：循环读取.dat文件并将数据推送到Redis队列
        """
        print(f"模拟数据采集线程启动，数据目录: {self.data_dir}")
        
        while not self.stop_event.is_set():
            try:
                # 获取目录下所有.dat文件
                dat_files = [f for f in os.listdir(self.data_dir) 
                            if f.lower().endswith('.dat')]
                
                if not dat_files:
                    print(f"警告: 在 {self.data_dir} 目录下未找到.dat文件")
                    break
                
                print(f"找到 {len(dat_files)} 个.dat文件: {dat_files}")
                
                # 依次处理每个.dat文件
                for dat_file in dat_files:
                    if self.stop_event.is_set():
                        break
                        
                    file_path = os.path.join(self.data_dir, dat_file)
                    print(f"正在处理文件: {file_path}")
                    
                    try:
                        # 读取.dat文件中的CSI数据
                        csi_data_list = parse_csi_data_from_file(file_path)

                            # 尝试加载同名JSON文件
                        json_data_list = []
                        json_file_path = file_path.replace('.dat', '.json')
                        if os.path.exists(json_file_path):
                            try:
                                with open(json_file_path, 'r', encoding='utf-8') as f:
                                    json_data_list = json.load(f)
                                    print(f"加载JSON文件: {json_file_path}, 包含 {len(json_data_list)} 个数据项")
                            except Exception as e:
                                print(f"无法加载JSON文件 {json_file_path}: {e}")
                                json_data_list = []
                        else:
                            print(f"未找到同名JSON文件: {json_file_path}")
                        
                        if csi_data_list is None:
                            print(f"解析文件 {file_path} 失败")
                            continue
                            
                        num_packets = len(csi_data_list)
                        
                        print(f"文件 {dat_file} 包含 {num_packets} 个数据包")
                        start = time.strftime("%H:%M:%S", time.localtime(json_data_list['start_time']))
                        end = time.strftime("%H:%M:%S", time.localtime(json_data_list['end_time']))
                        
                        # 依次将每个数据包发送到Redis队列
                        for i in range(num_packets):
                            if self.stop_event.is_set():
                                break
                           
                            # 发送数据到Redis队列
                            self.redis_client.lpush(CSI_SOURCE_QUEUE, json.dumps(csi_data_list[i]))
                            data_time = time.strftime("%H:%M:%S", time.localtime(csi_data_list[i]['send_time']))
                            if i % 100 == 0:
                                print(f"已发送数据包 {i+1}/{num_packets}，发送时间: {data_time}, 动作区间: {start} - {end}")

                            # 限制队列长度以避免内存溢出
                            self.redis_client.ltrim(CSI_SOURCE_QUEUE, 0, MAX_QUEUE_LENGTH - 1)
                            
                            # 休眠指定时间以模拟实时数据采集
                            time.sleep(self.interval)
                            
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print("所有.dat文件处理完成，等待下一轮...")
                # 等待一段时间后重新开始
                # self.stop_event.wait(5.0)  # 等待5秒或停止信号
                
            except Exception as e:
                print(f"模拟队列线程错误: {e}")
                time.sleep(1.0)
    


# 全局变量
simulation_thread = None


def start_simulation():
    """
    启动模拟数据采集线程
    """
    global simulation_thread
    if simulation_thread is None or not simulation_thread.is_alive():
        simulation_thread = SimulationQueueThread()
        simulation_thread.start()
        print("模拟数据采集线程已启动")
    else:
        print("模拟数据采集线程已在运行")


if __name__ == "__main__":
    # 直接运行测试
    print("启动模拟数据采集测试...")
    test_thread = SimulationQueueThread()
    test_thread.start()
    
    try:
        # 运行一段时间后停止
        time.sleep(30)  # 运行30秒
    except KeyboardInterrupt:
        print("\n接收到中断信号...")
    finally:
        test_thread.stop()
        test_thread.join()
        print("模拟数据采集测试结束")