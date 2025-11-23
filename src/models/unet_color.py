import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNetColor(nn.Module):
    """
    U-Net for L->ab. Output ab scaled to about [-110, 110] using tanh.
    """
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(b, b*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(b*2, b*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(b*4, b*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bot  = DoubleConv(b*8, b*16)

        self.up4  = nn.ConvTranspose2d(b*16, b*8, 2, 2)
        self.dec4 = DoubleConv(b*16, b*8)
        self.up3  = nn.ConvTranspose2d(b*8, b*4, 2, 2)
        self.dec3 = DoubleConv(b*8, b*4)
        self.up2  = nn.ConvTranspose2d(b*4, b*2, 2, 2)
        self.dec2 = DoubleConv(b*4, b*2)
        self.up1  = nn.ConvTranspose2d(b*2, b, 2, 2)
        self.dec1 = DoubleConv(b*2, b)

        self.out  = nn.Conv2d(b, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bot(self.pool4(e4))

        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)

        ab = self.out(d1).tanh() * 110.0
        return ab