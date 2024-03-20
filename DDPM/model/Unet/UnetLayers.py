from tensorflow.keras.layers import Layer, Dense, BatchNormalization, MaxPooling2D, Add, Conv2D, UpSampling2D, Activation, Multiply, Attention
from tensorflow.keras.activations import gelu

class DownBlock(Layer):
    def __init__(self, out_channels):
        super(DownBlock, self).__init__()
        self.norm1 = BatchNormalization(momentum=0.99)
        self.conv1 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')

        self.norm2 = BatchNormalization(momentum=0.99)
        self.conv2 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')
        
        self.time_emd = Dense(out_channels, activation=gelu)

        self.res_conv = Conv2D(filters=out_channels, kernel_size = 2, activation=gelu, padding='same')
        
        self.norm3 = BatchNormalization(momentum=0.99)
        self.attention = Attention(score_mode='dot')
        
        self.down_sample = MaxPooling2D((2, 2))
        
            
    def call(self, img, t_emd):
        
        x = self.norm1(img)
        x = self.conv1(x)

        x = Add()([x, self.time_emd(t_emd)[:,None,None,:]])
        
        x = self.norm2(x)
        x = self.conv2(x)

        y = Add()([x, self.res_conv(img)])
        
        x = self.norm3(y)
        x = self.attention([x, x, x])
        
        out = Add()([x, y])
        
        x = self.down_sample(out)
        
        return x, out


class MidBlock(Layer):

    def __init__(self, out_channels):
        
        super(MidBlock, self).__init__()
        self.norm1 = BatchNormalization(momentum=0.99)
        self.conv1 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')

        self.norm2 = BatchNormalization(momentum=0.99)
        self.conv2 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')
        
        self.time_emd = Dense(out_channels, activation=gelu)

        self.res_conv1 = Conv2D(filters=out_channels, kernel_size = 2, activation=gelu, padding='same')

        self.norm3 = BatchNormalization(momentum=0.99)

        self.norm3 = BatchNormalization(momentum=0.99)
        self.conv3 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')

        self.norm4 = BatchNormalization(momentum=0.99)
        self.conv4 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')
        
        self.attention = Attention(score_mode='dot')

        
        self.norm5 = BatchNormalization(momentum=0.99)
        self.time_emd = Dense(out_channels, activation=gelu)

        self.res_conv2 = Conv2D(filters=out_channels, kernel_size = 2, activation=gelu, padding='same')
    

    def call(self, img, t_emd):

        x = self.norm1(img)
        x = self.conv1(x)

        x = Add()([x, self.time_emd(t_emd)[:,None,None,:]])
        
        x = self.norm2(x)
        x = self.conv2(x)

        y = Add()([x, self.res_conv1(img)])

        x = self.norm3(y)
        x = self.attention([x, x, x])
        
        y = Add()([x, y])

        x = self.norm4(y)
        x = self.conv3(x)

        x = Add()([x, self.time_emd(t_emd)[:,None,None,:]])
        
        x = self.norm5(x)
        x = self.conv4(x)

        x = Add()([x, self.res_conv2(y)])

        return x
        
class UpBlock(Layer):

    def __init__(self, out_channels):
        super(UpBlock, self).__init__()
        
        self.channels = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')
        
        self.norm1 = BatchNormalization(momentum=0.99)
        self.conv1 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')

        self.norm2 = BatchNormalization(momentum=0.99)
        self.conv2 = Conv2D(filters=out_channels, kernel_size = 3, activation=gelu, padding='same')
        
        self.time_emd = Dense(out_channels, activation=gelu)

        self.res_conv = Conv2D(filters=out_channels, kernel_size = 2, activation=gelu, padding='same')
        
        self.norm3 = BatchNormalization(momentum=0.99)
        self.attention = Attention(score_mode='dot')
        
        self.up_sample = UpSampling2D((2, 2))

    def call(self, img, down, t_emd):

        img = self.channels(img)
        img = self.up_sample(img)
        img = Add()([img, down])

        x = self.norm1(img)
        x = self.conv1(x)

        x = Add()([x, self.time_emd(t_emd)[:,None,None,:]])
        
        x = self.norm2(x)
        x = self.conv2(x)

        y = Add()([x, self.res_conv(img)])
        
        x = self.norm3(y)
        x = self.attention([x, x, x])
        
        x = Add()([x, y])
                
        return x