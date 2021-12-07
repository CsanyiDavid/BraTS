import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.layers = nn.Sequential(
            nn.Conv3d(self.input_channels, self.output_channels, kernel_size=(3,3,3), padding='same'),
            nn.ReLU(),
            nn.Conv3d(self.output_channels, self.output_channels, (3,3,3), padding='same'),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down0 = ConvBlock(4, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool3d((2,2,2)),
             ConvBlock(32, 64)
        )     
        self.down2 = nn.Sequential(
            nn.MaxPool3d((2,2,2)),
            ConvBlock(64, 128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d((2,2,2)),
             ConvBlock(128, 256)
        ) 
        self.down4 = nn.Sequential(
            nn.MaxPool3d((2,2,2)),
            ConvBlock(256, 512)
        )
        self.up1 = ConvBlock(512, 256)
        self.up2 = ConvBlock(256, 128)
        self.up3 = ConvBlock(128, 64)
        self.up4 = ConvBlock(64, 32)
        self.upc1 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, (2,2,2), stride=2),
            nn.ReLU(),
        )
        self.upc2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2,2,2), stride=2),
            nn.ReLU(),
        )
        self.upc3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, (2,2,2), stride=2),
            nn.ReLU(),
        )
        self.upc4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, (2,2,2), stride=2),
            nn.ReLU(),
        )
        self.final = nn.Conv3d(32, 1, (1,1,1))
        
    def forward(self, x):
        
        #Contracting path
        #print('input shape: ', x.shape)
        #print(x.shape)
        x = F.avg_pool3d(x, (2,2,2))
        #print(x.shape)
        
        d0 = self.down0(x)
        #print('d0 shape: ', d0.shape)
        d1 = self.down1(d0)
        #print('d1 shape: ', d1.shape)
        d2 = self.down2(d1)
        #print('d2 shape: ', d2.shape)
        d3 = self.down3(d2)
        #print('d3 shape: ', d3.shape)
        d4 = self.down4(d3)
        #print('d4 shape: ', d4.shape)
        
        #Expansive path
        x = self.upc1(d4)
        x = torch.cat((x, d3), dim=1)
        del d3
        #print('cat1 shape: ', x.shape)
        x = self.up1(x)
        #print('u1 shape: ', x.shape)
        
        x = self.upc2(x)
        x = torch.cat((x, d2), dim=1)
        del d2
        #print('cat2 shape: ', x.shape)
        x = self.up2(x)
        #print('u2 shape: ', x.shape)
       
        x = self.upc3(x)
        x = torch.cat((x, d1), dim=1)
        del d1
        #print('cat3 shape: ', x.shape)
        x = self.up3(x)
        #print('u3 shape: ', x.shape)
        
        x = self.upc4(x)
        x = torch.cat((x, d0), dim=1)
        del d0
        #print('cat4 shape: ', x.shape)
        x = self.up4(x)
        #print('u4 shape: ', x.shape)
        
        x = self.final(x)
        x = torch.sigmoid(x)
        #print('ret shape: ', x.shape)
        
        #print(x.shape)
        x = F.interpolate(x, scale_factor=2)
        #print(x.shape)
        return x
