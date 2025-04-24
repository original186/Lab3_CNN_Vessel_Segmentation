import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(256, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Output
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.conv_out(dec1))

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResNet34UNet, self).__init__()
        
        # Load pre-trained ResNet34
        resnet = models.resnet34(pretrained=True)
        
        # Modify first conv layer for grayscale input
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Encoder (use up to layer4)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # [B, 64, H/2, W/2]
        self.encoder2 = resnet.layer1  # [B, 64, H/2, W/2]
        self.encoder3 = resnet.layer2  # [B, 128, H/4, W/4]
        self.encoder4 = resnet.layer3  # [B, 256, H/8, W/8]
        self.encoder5 = resnet.layer4  # [B, 512, H/16, W/16]
        
        # Decoder
        self.up1 = self._up_block(512, 256)  # [B, 256, H/8, W/8]
        self.conv1 = self._conv_block(512, 256)
        
        self.up2 = self._up_block(256, 128)  # [B, 128, H/4, W/4]
        self.conv2 = self._conv_block(256, 128)
        
        self.up3 = self._up_block(128, 64)   # [B, 64, H/2, W/2]
        self.conv3 = self._conv_block(128, 64)
        
        self.up4 = self._up_block(64, 64)    # [B, 64, H, W]
        self.conv4 = self._conv_block(128, 64)

        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        
        # Output layer
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)       # [B, 64, H/2, W/2]
        e2 = self.encoder2(e1)      # [B, 64, H/2, W/2]
        e3 = self.encoder3(e2)      # [B, 128, H/4, W/4]
        e4 = self.encoder4(e3)      # [B, 256, H/8, W/8]
        e5 = self.encoder5(e4)      # [B, 512, H/16, W/16]
        
        # Decoder with skip connections
        d1 = self.up1(e5)           # [B, 256, H/8, W/8]
        d1 = torch.cat([d1, e4], dim=1)  # [B, 512, H/8, W/8]
        d1 = self.conv1(d1)         # [B, 256, H/8, W/8]
        
        d2 = self.up2(d1)           # [B, 128, H/4, W/4]
        d2 = torch.cat([d2, e3], dim=1)  # [B, 256, H/4, W/4]
        d2 = self.conv2(d2)         # [B, 128, H/4, W/4]
        
        d3 = self.up3(d2)           # [B, 64, H/2, W/2]
        d3 = torch.cat([d3, e2], dim=1)  # [B, 128, H/2, W/2]
        d3 = self.conv3(d3)         # [B, 64, H/2, W/2]
        
        d4 = self.up4(d3)           # [B, 64, H, W]

        d4 = self.final_upsample(d4)   # [B, 64, 512, 512]
     
        
        # Final output
        return self.sigmoid(self.conv_out(d4))

def get_model(model_name, in_channels=1, out_channels=1):
    models = {
        'unet': UNet,
        'resnet34unet': ResNet34UNet
    }
    return models[model_name.lower()](in_channels, out_channels)
