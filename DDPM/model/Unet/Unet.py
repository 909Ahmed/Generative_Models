from .UnetLayers import *
from .TimeEmbedding import TimeEmbedding
from tensorflow.keras.models import Model

class Unet(Model):
    
    def __init__(self):
        super(Unet, self).__init__()
        self.downs = 3
        self.mids = 2
        self.ups = 3
        self.down_channels = [32, 64, 128, 256]
        self.mids_channels = [256, 256]
        self.up_channels = [256, 128, 64, 32]
        self.in_conv = Conv2D(3, activation=gelu, kernel_size = 3, padding='same')
        self.norm = BatchNormalization()
        self.act = gelu
        self.out_conv = Conv2D(1, activation='linear', kernel_size = 3, padding='same') # is this correct?
        self.time_embedder = TimeEmbedding(128)
        self.downs = [DownBlock(i) for i in self.down_channels]
        self.mids = [MidBlock(i) for i in self.mids_channels]
        self.ups = [UpBlock(i) for i in self.up_channels]
        
        
    def call(self, imgs, times):

        skip = []
        time_emd = self.time_embedder(times)

        x = self.in_conv(imgs)
        x = self.act(x)

        for i in range(len(self.down_channels)):
            x, y = self.downs[i](x, time_emd)
            skip.append(y)

        for i in range(len(self.mids_channels)):
            x = self.mids[i](x, time_emd)
        
        skip = skip[::-1]
        for i in range(len(self.up_channels)):
            x = self.ups[i](x, skip[i], time_emd)

        x = self.norm(x)
        x = self.act(x)
        x = self.out_conv(x)

        return x