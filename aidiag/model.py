import torch
import torch.nn as nn


class DicomUNet(nn.Module):
    """UNet - based 3D convolutional segmentation neural net.
            Parameters
            ----------
            scale : {int} variable "horizontal" scale of network. The network depth is constant, the number of
                filters in each convolutional layer is proportional to the "scale" value.
                This value should be increased if there is a lot of data, and reduced in case of overfitting or to
                reduce the resource intensity of the network.
            """

    def __init__(self, scale: int):
        super().__init__()

        self.enc_conv0 = nn.Sequential(
            nn.Conv3d(1, scale,
                      kernel_size=(1, 2, 2),
                      padding=(0, 0, 0),
                      stride=(1, 2, 2),
                      padding_mode='replicate'),
            nn.BatchNorm3d(scale),
            nn.ReLU(inplace=True),
        )  # h*256*256

        self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)  # h/2*128*128

        self.enc_conv1 = nn.Sequential(
            nn.Conv3d(scale, scale * 2,
                      kernel_size=(3, 2, 2),
                      padding=(1, 0, 0),
                      stride=(1, 2, 2),
                      padding_mode='replicate',
                      bias=False),
            # h/2*64*64
            nn.BatchNorm3d(scale * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(scale * 2, scale * 2,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            # h/2*64*64
            nn.BatchNorm3d(scale * 2),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # h/4*32*32

        self.enc_conv2 = nn.Sequential(
            nn.Conv3d(scale * 2, scale * 3,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            # h/4*32*32
            nn.BatchNorm3d(scale * 3),
            nn.ReLU(inplace=True),
            nn.Conv3d(scale * 3, scale * 3,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            # h/4*32*32
            nn.BatchNorm3d(scale * 3),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # h/8*16*16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(scale * 3, scale * 5,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            nn.BatchNorm3d(scale * 5),
            nn.ReLU(inplace=True),
            nn.Conv3d(scale * 5, scale * 16,
                      kernel_size=4,
                      padding=1,
                      stride=2,
                      padding_mode='replicate'),
            nn.BatchNorm3d(scale * 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(scale * 16, scale * 8,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            nn.BatchNorm3d(scale * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(scale * 8, scale * 3,
                               kernel_size=4,
                               padding=1,
                               stride=2,
                               bias=False
                               ),
            nn.BatchNorm3d(scale * 3),
            nn.ReLU(inplace=True),
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose3d(scale * 3, scale * 5, kernel_size=2, stride=2, bias=False)  # h/4*32*32
        self.dec_conv0 = nn.Sequential(
            nn.Conv3d(scale * 8, scale * 4,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            nn.BatchNorm3d(scale * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(scale * 4, scale * 4,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            nn.BatchNorm3d(scale * 4),
            nn.ReLU(inplace=True),
        )

        self.upsample1 = nn.ConvTranspose3d(scale * 4, scale * 2, kernel_size=2, stride=2, bias=False)  # h/2*64*64

        self.dec_conv1 = nn.Sequential(
            nn.Conv3d(scale * 4, scale * 3,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            nn.BatchNorm3d(scale * 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(scale * 3, scale * 2,
                               kernel_size=(3, 2, 2),
                               padding=(1, 0, 0),
                               stride=(1, 2, 2),
                               bias=False
                               ),
            # h/2*128*128
            nn.BatchNorm3d(scale * 2),
            nn.ReLU(inplace=True),
        )

        self.upsample2 = nn.ConvTranspose3d(scale * 2, scale, kernel_size=2, stride=2, bias=False)  # h*256*256

        self.dec_conv2 = nn.Sequential(
            nn.Conv3d(scale * 2, scale * 2,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      padding_mode='replicate',
                      bias=False),
            nn.BatchNorm3d(scale * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(scale * 2, 1,
                               kernel_size=(3, 2, 2),
                               padding=(1, 0, 0),
                               stride=(1, 2, 2)
                               ),
        )

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)  # h*256*256
        #         print('e0:' + str(e0.shape))
        e1 = self.enc_conv1(self.pool0(e0))  # h/2*64*64
        #         print('e1:' + str(e1.shape))
        e2 = self.enc_conv2(self.pool1(e1))  # h/4*32*32
        #         print('e2:' + str(e2.shape))

        # bottleneck
        x = self.bottleneck_conv(self.pool2(e2))  # h/8*16*16
        #         print('b:' + str(x.shape))

        # decoder
        x = self.dec_conv0(torch.cat((self.upsample0(x), e2), dim=1))
        #         print('d0:' + str(x.shape))
        x = self.dec_conv1(torch.cat((self.upsample1(x), e1), dim=1))
        #         print('d1:' + str(x.shape))
        x = self.dec_conv2(torch.cat((self.upsample2(x), e0), dim=1))
        #         print('d2:' + str(x.shape))
        return x
