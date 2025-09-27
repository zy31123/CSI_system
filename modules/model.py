import torch 
import torch.nn as nn
import torch.nn.functional as F
import os


class FeatureExtractor(nn.Module):
    """特征提取网络"""
    def __init__(self, input_channels=6):
        super(FeatureExtractor, self).__init__()
        # 输入尺寸: input_channels x 400 x 56 (通道 x 高度 x 宽度)
        self.features = nn.Sequential(
            # 第一层卷积 - 使用较小的卷积核
            nn.Conv2d(input_channels, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 输出: 16 x 400 x 56
            
            # 第一层池化 - 快速降维
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            # 输出: 16 x 100 x 28
            
            # 第二层卷积
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 输出: 32 x 100 x 28
            
            # 第二层池化
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            # 输出: 32 x 25 x 14
            
            # 第三层卷积
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输出: 64 x 25 x 14
            
            # 第三层池化
            nn.MaxPool2d(kernel_size=(5, 7), stride=(5, 7)),
            # 输出: 64 x 5 x 2
        )
        
    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    """分类网络"""
    def __init__(self, num_classes=3):
        super(Classifier, self).__init__()
        # 输入特征维度: 64 x 5 x 2 = 640
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平为 640
            nn.Dropout(0.3),
            nn.Linear(640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class AmplitudeNetwork(nn.Module):
    """幅度数据分类网络"""
    def __init__(self, num_classes=3):
        super(AmplitudeNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channels=6)  # 3个接收天线 * 2个发送天线 = 6个通道
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """提取特征向量用于少样本学习"""
        features = self.feature_extractor(x)
        # 展平特征但不通过分类器
        flattened_features = torch.flatten(features, 1)
        return flattened_features


class PhaseNetwork(nn.Module):
    """相位数据分类网络"""
    def __init__(self, num_classes=3):
        super(PhaseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channels=6)  # 3个接收天线 * 2个发送天线 = 6个通道
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """提取特征向量用于少样本学习"""
        features = self.feature_extractor(x)
        # 展平特征但不通过分类器
        flattened_features = torch.flatten(features, 1)
        return flattened_features


class NetWork(nn.Module):
    """完整的网络结构：特征提取 + 分类"""
    def __init__(self, num_classes=3):
        super(NetWork, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channels=6)  # 3个接收天线 * 2个发送天线 = 6个通道
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features
    

class CombinedFeatureNetwork(nn.Module):
    """
    结合幅度和相位特征的神经网络分类器，优化用于少样本学习
    """
    def __init__(self, amplitude_model_path, phase_model_path, amplitude_feature_dim=640, phase_feature_dim=640, num_classes=3, hidden_dim=512):
        super(CombinedFeatureNetwork, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amplitude_model = AmplitudeNetwork(num_classes=num_classes)
        if os.path.exists(amplitude_model_path):
            self.amplitude_model.load_state_dict(torch.load(amplitude_model_path, map_location=device))
            print(f"成功加载幅度模型: {amplitude_model_path}")
        else:
            print(f"找不到幅度模型: {amplitude_model_path}")
            return
        
        self.phase_model = PhaseNetwork(num_classes=num_classes)
        if os.path.exists(phase_model_path):
            self.phase_model.load_state_dict(torch.load(phase_model_path, map_location=device))
            print(f"成功加载相位模型: {phase_model_path}")
        else:
            print(f"找不到相位模型: {phase_model_path}")
            return


        # 特征融合层 - 分别处理幅度和相位特征
        self.amplitude_feature_layer = nn.Sequential(
            nn.Linear(amplitude_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.Dropout(0.2),
        )
        
        self.phase_feature_layer = nn.Sequential(
            nn.Linear(phase_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.Dropout(0.2),
        )
        
        # 注意力机制 - 帮助模型关注重要特征
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=8, batch_first=True)
        
        # 融合幅度和相位特征
        combined_dim = hidden_dim * 2  # 幅度特征维度 + 相位特征维度
        
        # 分类层 - 减少参数量，避免少样本过拟合
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, amplitude, phase):
        amplitude = torch.from_numpy(amplitude).float()
        phase = torch.from_numpy(phase).float()
        # 提取幅度和相位特征
        amplitude_features = self.amplitude_model.extract_features(amplitude)  # [batch_size, amplitude_feature_dim]
        phase_features = self.phase_model.extract_features(phase)  # [batch

        # 处理幅度特征
        amp_out = self.amplitude_feature_layer(amplitude_features)
        
        # 处理相位特征
        phase_out = self.phase_feature_layer(phase_features)
        
        # 拼接特征
        combined_features = torch.cat((amp_out, phase_out), dim=1)
        
        # 注意力机制处理
        # 重塑为适合注意力机制的格式
        combined_features_expanded = combined_features.unsqueeze(1)  # [batch_size, 1, combined_dim]
        attended_features, _ = self.attention(
            combined_features_expanded, combined_features_expanded, combined_features_expanded
        )
        attended_features = attended_features.squeeze(1)  # [batch_size, combined_dim]
        
        # 最终分类
        output = self.classifier(attended_features)
        
        return output


# 辅助函数：计算网络输出维度
def calculate_output_size():
    """计算网络各层输出尺寸"""
    import torch
    model = NetWork()
    dummy_input = torch.randn(1, 12, 400, 56)  # 修改为新的输入尺寸
    
    print("输入尺寸:", dummy_input.shape)
    
    # 计算特征提取器输出
    features = model.feature_extractor(dummy_input)
    print("特征提取器输出尺寸:", features.shape)
    
    # 计算最终输出
    output = model(dummy_input)
    print("网络最终输出尺寸:", output.shape)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")


if __name__ == "__main__":
    calculate_output_size()
