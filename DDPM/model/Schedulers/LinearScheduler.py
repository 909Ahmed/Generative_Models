import tensorflow as tf

class LinearScheduler:
    
    def __init__(self):
        
        self.betas = tf.linspace(0.0001, 0.02, 1000)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = tf.math.cumprod(self.alphas)
        self.sqrt_alpha_bar = tf.sqrt(self.alpha_bar)
        self.sqrt_one_alpha_bar = tf.sqrt(1 - self.alpha_bar)
        
    def forward(self, x0, noise, t):
        
        batch_size = x0.shape[0]
        
        sqrt_alpha_bar = tf.gather(self.sqrt_alpha_bar, t)
        sqrt_one_alpha_bar = tf.gather(self.sqrt_one_alpha_bar, t)        
                
        sqrt_alpha_bar = tf.reshape(sqrt_alpha_bar, shape=[batch_size, 1, 1, 1])
        sqrt_one_alpha_bar = tf.reshape(sqrt_one_alpha_bar, shape=[batch_size, 1, 1, 1])
        
        return sqrt_alpha_bar * x0 + sqrt_one_alpha_bar * noise
    
    def backward(self, xt, noise, t):
        
        sqrt_one_alpha_bar = tf.gather(self.sqrt_one_alpha_bar, t)
        alpha_bar = tf.gather(self.alpha_bar, t)
        betas = tf.gather(self.betas, t)
        alpha = tf.gather(self.alphas, t)
        alpha_bar_t = tf.gather(self.alpha_bar, t - 1)
        
        x0 = (xt - (sqrt_one_alpha_bar * noise)) / alpha_bar
        x0 = tf.clip_by_value(x0, -1.0, 1.0)
        mean = (xt - ((betas * noise) / sqrt_one_alpha_bar)) / tf.sqrt(alpha)
        
        if t == 0:
            return mean, mean
        else:
            std = tf.sqrt((betas * (1 - alpha_bar_t)) / (1 - alpha_bar))
            z = tf.random.normal(xt.shape)
            return mean + std * z, x0