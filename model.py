import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self , channels_img , features_d):
        super(Discriminator , self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img , features_d , kernel_size=4 , stride = 2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d , features_d*2 , 4 , 2 , 1) ,
            self._block(features_d*2 , features_d*4 , 4 , 2 , 1) ,
            self._block(features_d*4, features_d*8 , 4 , 2 , 1) ,
            nn.Conv2d(features_d*8 , 1 , kernel_size=4 , stride=2 , padding=0),
            nn.Sigmoid(),
        )
    
    def _block (self , in_channels , out_channels , kernel_size , stride , padding):
        return nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size ,stride , padding , bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self , x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self , z_dim , channels_img , features_g):
        super(Generator , self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim , features_g*16 , 4 , 1 , 0),
            self._block(features_g*16 ,features_g*8, 4 , 1 , 0),
            self._block(features_g*8 ,features_g*4, 4 , 1 , 0),
            self._block(features_g*4 ,features_g*2, 4 , 1 , 0),
            nn.ConvTranspose2d(features_g*2 , channels_img , kernel_size=4 , stride=2 , padding=1),
            nn.Tanh(),
        )
    
    def _block(self , in_channels , out_channels , kernel_size , stride , padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels , out_channels , kernel_size , stride , padding , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self , x):
        return self.gen(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

