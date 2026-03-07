import torch
import torch.nn as nn
import torchvision.models as models

def concatenate_attention_features(attention_outputs, max_tokens_per_modality=4096):
    """
    Concatenates attention features from multiple modalities and reshapes into 4-channel image format.
    Each modality becomes one channel.
    
    Args:
        attention_outputs: list of 4 tensors, each [B, T, D]
        max_tokens_per_modality: standardize temporal dimension to this value
    
    Returns:
        [B, 4, H, W] tensor where 4 = num_modalities
    """
    modality_channels = []
    for features in attention_outputs:
        B, T, D = features.shape
        
        # Standardize temporal dimension across all modalities
        if T > max_tokens_per_modality:
            # Downsample
            indices = torch.linspace(0, T-1, max_tokens_per_modality, dtype=torch.long, device=features.device)
            sampled_features = features[:, indices, :]  # [B, max_tokens, D]
        elif T < max_tokens_per_modality:
            # Upsample with padding
            padding = max_tokens_per_modality - T
            pad_tensor = torch.zeros(B, padding, D, device=features.device, dtype=features.dtype)
            sampled_features = torch.cat([features, pad_tensor], dim=1)
        else:
            sampled_features = features
        
        # Flatten each modality: [B, T, D] -> [B, T*D]
        flattened = sampled_features.view(B, -1)
        modality_channels.append(flattened)
    
    # Stack modalities as separate channels: [B, 4, features_per_modality]
    stacked = torch.stack(modality_channels, dim=1)  # [B, 4, T*D]
    
    B, num_channels, total_features = stacked.shape
    
    # Reshape into spatial dimensions: [B, 4, H, W]
    H = int(total_features ** 0.5)
    W = total_features // H
    
    # Pad or trim to fit exactly H * W
    needed_features = H * W
    if needed_features > total_features:
        # Pad with zeros
        padding = needed_features - total_features
        padding_tensor = torch.zeros(B, num_channels, padding, device=stacked.device, dtype=stacked.dtype)
        stacked = torch.cat([stacked, padding_tensor], dim=2)
    else:
        # Trim excess
        stacked = stacked[:, :, :needed_features]
    
    reshaped = stacked.view(B, num_channels, H, W)  # [B, 4, H, W]
    
    return reshaped

class ResNet(nn.Module):
    def __init__(self, max_tokens_per_modality, num_classes=2):
        super().__init__()
        self.max_tokens_per_modality = max_tokens_per_modality

        # Load ResNet18 and modify first conv layer to accept 4 channels
        resnet = models.resnet18(pretrained=False)
        
        # Replace first conv layer: 3 channels -> 4 channels
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            4,  # 4 input channels (one per modality)
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize the new conv layer
        # Average the original RGB weights and replicate for 4th channel
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv1.weight
            self.conv1.weight[:, 3] = original_conv1.weight.mean(dim=1)
        
        # Use rest of ResNet (excluding first conv and final fc layer)
        self.backbone = nn.Sequential(
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        # Add dropout after backbone features
        self.feature_dropout = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18 outputs 512 features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, attention_features):
        reshaped_features = concatenate_attention_features(
            attention_features, 
            max_tokens_per_modality=self.max_tokens_per_modality
        )  # [B, 4, H, W]
        
        batch_size = reshaped_features.size(0)

        # Pass through modified first conv layer
        x = self.conv1(reshaped_features)
        
        # Pass through rest of ResNet backbone
        features = self.backbone(x)  # [B, 512, 1, 1]
        features = features.view(batch_size, -1)  # [B, 512]
        
        # Apply dropout to backbone features
        features = self.feature_dropout(features)
        
        output = self.classifier(features)
        return output