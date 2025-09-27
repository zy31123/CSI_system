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
import torch 
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sympy import resultant
from model import CombinedFeatureNetwork
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    REDIS_HOST, REDIS_PORT, CSI_PROCESSED_QUEUE, 
    CSI_VISUALIZATION_CHANNEL, MAX_QUEUE_LENGTH, WINDOW_SIZE, NETWORK_SIZE, NETWORK_OVERLAP
)

# Global variables
stop_event = Event()

# Redis client connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
CLASS_NUMS = 3  # 分类数量
import os
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_para', f'{CLASS_NUMS}_cnn_feature_extractor.pth')
amplitude_model_path = os.path.join("models", "amplitude_model1.pth")
phase_model_path = os.path.join("models", "phase_model1.pth")
classify_model_path = os.path.join("models", "fewshot_amplitude_phase_neural_model2.pth")


class ConvBlock(nn.Module):
    """单个卷积块的结构：Conv2D + BatchNorm（批量标准化层） + ReLU (Relu 激活函数) + Pooling"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层能过够保留重要得特征，对于图像分类任务表现更好

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # 对通道维度做归一化并引入可学习的仿射变换（稳定训练、允许更大学习率）。
        x = F.relu(x)  # 引入非线性、稀疏激活。
        x = self.pool(x)  # 下采样（降分辨率）、减少计算与内存、同时扩大感受野并具备少量平移不变性
        return x

class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器模型"""

    def __init__(self, input_shape, num_classes=4):
        super(CNNFeatureExtractor, self).__init__()

        # 计算卷积后的特征图尺寸
        self.input_shape = input_shape
        self.num_classes = num_classes

        # 四个卷积块
        self.conv1 = ConvBlock(input_shape[0], 16)  # 输入通道数, 输出通道数
        self.conv2 = ConvBlock(16, 16)
        self.conv3 = ConvBlock(16, 16)
        self.conv4 = ConvBlock(16, 16)

        # 计算全连接层输入尺寸
        self._calculate_fc_input_size()

        # 全连接层
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _calculate_fc_input_size(self):
        """计算全连接层的输入尺寸"""
        # 使用公式计算特征图尺寸
        h, w = self.input_shape[1], self.input_shape[2]
        print(f"输入形状: {self.input_shape}")
        print(f"初始特征图尺寸: {h} × {w}")
        
        # 经过4次2x2池化，每次池化尺寸减半
        for i in range(4):
            h = h // 2
            w = w // 2
            
        h_out, w_out = h, w
        self.fc_input_size = 16 * h_out * w_out  # 16是最后一个卷积层的输出通道数

    def forward(self, x):
        # 卷积块
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 展平
        x1 = x.view(x.size(0), -1)
        # print(f"CNN特征提取器输出维度: {x1.shape}")

        # 全连接层
        x = self.fc(x1)

        return x,x1


class DualCNNFeatureExtractor(nn.Module):
    """双分支CNN特征提取器，用于处理振幅和相位差信息"""

    def __init__(self, input_shape=(6, 400, 56), num_classes=2):
        super(DualCNNFeatureExtractor, self).__init__()

        # 两个独立但结构相同的CNN特征提取器
        self.amp_extractor = CNNFeatureExtractor(input_shape, num_classes)
        self.phd_extractor = CNNFeatureExtractor(input_shape, num_classes)

    def forward(self, amp_input, phd_input):
        # # 合并特征（可以根据需要调整合并策略）
        # combined_features = (amp_features + phd_features) / 2
        _,amp_features = self.amp_extractor(amp_input)
        _,phd_features = self.phd_extractor(phd_input)

        # # 合并特征（可以根据需要调整合并策略）
        # combined_features = (amp_features + phd_features) / 2
        # 确保返回PyTorch张量而不是NumPy数组
        feats = torch.cat([amp_features, phd_features], dim=1)

        return feats

class DualCNN(nn.Module):
    """双分支CNN特征提取器，用于处理振幅和相位差信息"""

    def __init__(self, input_shape=(6, 400, 56), num_classes=2, model_dir='tar_models', path=model_path, class_name='svm'):
        super(DualCNN, self).__init__()
        # 两个独立但结构相同的CNN特征提取器
        self.extract = DualCNNFeatureExtractor(input_shape, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extract = self.extract.to(self.device)
        
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    state = ckpt['model_state_dict']
                elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    state = ckpt['state_dict']
                else:
                    state = ckpt
                self.extract.load_state_dict(state)
                print(f"已加载预训练模型权重: {path}")
            except Exception as e:
                print(f"加载预训练权重失败，使用随机初始化模型: {e}")
                # Don't exit here, continue with random initialization
        else:
            print(f"未找到权重文件 {path}，使用随机初始化模型")

        self.extract.eval()

        # Use proper path joining for model files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            self.lr = joblib.load(os.path.join(base_dir, 'tar_models', 'lr_model.joblib'))
            self.svm = joblib.load(os.path.join(base_dir, 'tar_models', 'svm_model.joblib'))
            self.knn = joblib.load(os.path.join(base_dir, 'tar_models', 'knn_model.joblib'))
            # 加载标准化器
            self.scaler = joblib.load(os.path.join(base_dir, 'tar_models', 'scaler.joblib'))
        except Exception as e:
            print(f"加载分类器模型失败: {e}")
            self.lr = None
            self.svm = None
            self.knn = None
            self.scaler = None

        # 分类器名称
        self.classname = class_name

    def forward(self, amp_input, phd_input):
        # 分别处理振幅和相位差信息
        feats = self.extract(amp_input, phd_input)
        # 确保feats是PyTorch张量
        if not isinstance(feats, torch.Tensor):
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats).float()
            else:
                # 如果是列表或其他类型，先转为numpy再转为tensor
                feats = torch.from_numpy(np.asarray(feats)).float()
        
        # 确保设备一致性
        feats = feats.to(self.device)
        
        # 转换为numpy用于sklearn分类器
        feats_np = feats.detach().cpu().numpy()
        if feats_np.ndim == 1:
            feats_np = feats_np.reshape(1, -1)
            
        # 使用标准化器处理特征（如果存在）
        if self.scaler is not None:
            feats_np = self.scaler.transform(feats_np)
            
        # 使用分类器进行预测
        y1 = self.lr.predict_proba(feats_np)
        y2 = self.svm.predict_proba(feats_np)
        y3 = self.knn.predict_proba(feats_np)
        result = {'lr': y1, 'svm': y2, 'knn': y3}
        # print(f"分类器预测结果: {y1}, {y2}, {y3}")

        return result[self.classname]


class classifyLinear(nn.Module):
    """双分支CNN特征提取器，用于处理振幅和相位差信息"""

    def __init__(self, input_shape=(6, 400, 56), num_classes=2):
        super(classifyLinear, self).__init__()

        # 两个独立但结构相同的CNN特征提取器
        self.linear = nn.Linear(1200, 3)

    def forward(self, amp_input):
        # 分别处理振幅和相位差信息
        result = self.linear(amp_input)

        # # 合并特征（可以根据需要调整合并策略）
        # combined_features = (amp_features + phd_features) / 2

        return result
    
# class classifyLinear(nn.Module):    
#     """双分支CNN特征提取器，用于处理振幅和相位差信息"""    
#     def __init__(self, input_shape=(6, 400, 56), num_classes=3):        
#         super(classifyLinear, self).__init__()        
#         # 两个独立但结构相同的CNN特征提取器       
#         # # 分类层        
#         self.classifier = nn.Sequential(            
#             nn.Linear(1200,256),            
#             nn.ReLU(inplace=True),            
#             nn.Dropout(0.5),            
#             nn.Linear(256, 64),            
#             nn.ReLU(inplace=True),            
#             nn.Dropout(0.3),            
#             nn.Linear(64, num_classes)
#         )    
        
#     def forward(self, amp_input):        
#         # 分别处理振幅和相位差信息
#         result= self.classifier(amp_input)
#         # # 合并特征（可以根据需要调整合并策略）
#         # combined_features = (amp_features + phd_features) / 2
#         return result


class DualCNN1(nn.Module):
    """双分支CNN特征提取器，用于处理振幅和相位差信息"""

    def __init__(self, input_shape=(6, 400, 56), num_classes=3, model_dir='tar_models', path=model_path, class_name='svm'):
        super(DualCNN1, self).__init__()
        # 两个独立但结构相同的CNN特征提取器
        self.extract = DualCNNFeatureExtractor(input_shape, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extract = self.extract.to(self.device)
        
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    state = ckpt['model_state_dict']
                elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    state = ckpt['state_dict']
                else:
                    state = ckpt
                self.extract.load_state_dict(state)
                print(f"已加载预训练模型权重: {path}")
            except Exception as e:
                print(f"加载预训练权重失败，使用随机初始化模型: {e}")
                # Don't exit here, continue with random initialization
        else:
            print(f"未找到权重文件 {path}，使用随机初始化模型")

        self.extract.eval()

        self.classifer = classifyLinear()
        self.classifer = self.classifer.to(self.device)
        path2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tar_models/classify.pth')
        if os.path.exists(path2):
            self.classifer = torch.load(path2, map_location=self.device,weights_only=False)
            print(f"已加载分类器模型: {path2}")
        self.classifer.eval()

    def forward(self, amp_input):
        # 分别处理振幅和相位差信息
        feats = self.extract.amp_extractor(amp_input)[1]
        feats = feats.view(feats.size(0), -1)
        result = self.classifer(feats)
        result = F.softmax(result)
        # print(result)
        return result

class NeuralNetworkInferenceThread(threading.Thread):
    """Neural network inference thread: Processes time-window data and generates results"""
    
    def __init__(self):
        super().__init__(name="NeuralNetworkInferenceThread")
        # self.model = DualCNN(num_classes=3,class_name='knn')  # Adjust num_classes as needed
        # self.model = DualCNN1(num_classes=3)  # Adjust num_classes as needed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CombinedFeatureNetwork(amplitude_model_path, phase_model_path).to(device)
        self.model.eval()
        if os.path.exists(classify_model_path):
            self.model.load_state_dict(torch.load(classify_model_path, map_location=device), strict=False)
            print(f"成功加载分类器模型: {classify_model_path}")
        else:
            print(f"找不到分类器模型: {classify_model_path}")
        self.buffer = []
        self.previous_batch = []
    
    def run(self):
        """Main neural network inference thread function"""
        print("Neural network inference thread started")
        
        while not stop_event.is_set():
            try:
                # Check if there's data in the processed queue
                queue_length = redis_client.llen(CSI_PROCESSED_QUEUE)
                
                # Process data in batches
                batch_size = NETWORK_SIZE

                if queue_length >= batch_size - len(self.previous_batch):
                    # Get a batch of processed data
                    batch_data = self.previous_batch.copy()
                    for _ in range(batch_size - len(self.previous_batch)):
                        data = redis_client.rpop(CSI_PROCESSED_QUEUE)
                        if data:
                            batch_data.append(data)
                    self.previous_batch = batch_data[-NETWORK_OVERLAP:]
                    # Process the batch
                    if batch_data:
                        result = self._process_batch(batch_data)
                        classification = self._format_classification(result['prediction'])
                        # 打印当前的时间（时分秒），分类结果和置信度
                        data_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result['send_time']))
                        if classification == '跌倒':
                            print(f"\033[91m时间: {data_time}, 分类结果: {classification}, 置信度: {result['confidence']}\033[0m")  # 红色
                        elif classification == '行走':
                            print(f"\033[92m时间: {data_time}, 分类结果: {classification}, 置信度: {result['confidence']}\033[0m")  # 绿色
                        else:
                            print(f"时间: {data_time}, 分类结果: {classification}, 置信度: {result['confidence']}")  # 默认颜色

                        # Publish results to visualization channel for real-time updates
                        # for result in results:
                            # Format result for visualization
                        visualization_result = {
                            'type': 'classification_result',
                            'send_time': result['send_time'],
                            'classification': classification,
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
        amp_data, phase_data = self._prepare_input(parsed_data)
        
        # Process with neural network
        if self.model:
            # Actual model inference
            predictions = self._model_predict(amp_data, phase_data)
        else:
            # Mock prediction when no model is loaded
            predictions = self._mock_predict(amp_data, phase_data)
        
        # Format results
        for i, prediction in enumerate(predictions):
            result = {
                'send_time': parsed_data[-1]['send_time'] if i < len(parsed_data) else time.time(),
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'confidence': float(np.max(prediction)) if hasattr(prediction, '__getitem__') else 0.0,
                'processing_time': time.time()
            }
            results.append(result)
        
        return results[-1]
    
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
            # Handle empty or invalid predictions
        # Map index to label
        labels = ['静止', '行走', '跌倒']  # Replace with your actual labels
        return labels[class_idx] if class_idx < len(labels) else f'Class {class_idx}'
    
    def _prepare_input(self, parsed_data):
        """准备神经网络输入数据，格式为 (batch, 6, 400, 56)"""
        amp_data_list = []
        phase_data_list = []
        
        for data in parsed_data:
            # 初始化输入数据数组
            if 'amplitude_data' in data and 'phase_data' in data:
                amplitude_data = np.array(data['amplitude_data'])  # Shape: (3, 2, 56)
                phase_data = np.array(data['phase_data'])          # Shape: (3, 2, 56)
                amp_data_list.append(amplitude_data)
                phase_data_list.append(phase_data)


        amp_data_list = np.array(amp_data_list)  # Shape: (400, 3, 2, 56)
        phase_data_list = np.array(phase_data_list)  # Shape: (400, 3, 2, 56)

        # Reshape to (batch, 6, 400, 56)
        # 6 = 3 (rx antennas) * 2 (amplitude and phase)
        # 56 = number of subcarriers
        # 400 = time steps (WINDOW_SIZE)
        # 注意：输入形状应该是 (batch, channels, height, width) 即 (batch, 6, 400, 56)
        # 正确的重塑和转置方式：
        # 1. 将 (400, 3, 2, 56) 重塑为 (400, 6, 56)
        # 2. 转置为 (6, 400, 56)
        # 3. 增加批次维度得到 (1, 6, 400, 56)
        n_samples = amp_data_list.shape[0]  # 400
        n_rx = amp_data_list.shape[1]       # 3
        n_tx = amp_data_list.shape[2]       # 2
        n_subcarriers = amp_data_list.shape[3]  # 56
        
        # 重塑：将rx和tx维度合并为通道维度
        amp_reshaped = amp_data_list.reshape(n_samples, n_rx * n_tx, n_subcarriers)  # (400, 6, 56)
        phase_reshaped = phase_data_list.reshape(n_samples, n_rx * n_tx, n_subcarriers)  # (400, 6, 56)
        
        # 转置：将时间维度和通道维度交换，以符合PyTorch的(batch, channels, height, width)格式
        amp_transposed = amp_reshaped.transpose(1, 0, 2)  # (6, 400, 56)
        phase_transposed = phase_reshaped.transpose(1, 0, 2)  # (6, 400, 56)
        
        # 增加批次维度
        amp_final = np.expand_dims(amp_transposed, axis=0)  # (1, 6, 400, 56)
        phase_final = np.expand_dims(phase_transposed, axis=0)  # (1, 6, 400, 56)

        return amp_final, phase_final

    def _model_predict(self, amp_data, phase_data):
        """使用独立特征提取网络进行模型推理"""
        if amp_data.size == 0 or phase_data.size == 0:
            return np.array([])

        # Check if models are loaded
        # Since DualCNN1 doesn't have the lr attribute, we need to check differently
        if not self.model:
            print("模型未加载，返回空预测结果")
            return np.array([])

        # 将numpy数组转换为PyTorch张量
        # input_data应该具有形状 (batch_size, 6, 400, 56)
        
        # 根据模型类型决定输入参数
        with torch.no_grad():  # 禁用梯度计算以进行推理
            predictions = self.model(amp_data, phase_data)
        
        predictions = predictions.cpu().numpy()  # 转回numpy数组
        print(f"模型预测结果: {predictions}")
        # 如果predictions是numpy数组，直接返回
        if isinstance(predictions, np.ndarray):
            return predictions
        
        # 如果predictions是PyTorch张量，转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            return predictions.numpy()
            
        # 其他情况，尝试转换为numpy数组
        return np.asarray(predictions)

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