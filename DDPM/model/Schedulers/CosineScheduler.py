import tensorflow as tf

class CosineScheduler:
    def __init__(self):

        start_angle = tf.acos(0.95)
        end_angle = tf.acos(0.02)
        
        diffusion_times = tf.linspace(0.0, 1.0, 1000)
        
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        self.sqrt_alpha_bar = tf.cos(diffusion_angles)
        self.sqrt_one_minus_alpha_bar = tf.sin(diffusion_angles)
        
        self.alpha_bar = tf.math.square(self.sqrt_alpha_bar)
        self.alphas = tf.divide(self.alpha_bar, tf.concat([tf.constant([1.]), self.alpha_bar[:-1]], axis=0))
        self.betas = 1.0 - self.alphas
            
    def forward(self, x0, noise, t):
        
        batch_size = x0.shape[0]
        
        sqrt_alpha_bar = tf.gather(self.sqrt_alpha_bar, t)
        sqrt_one_alpha_bar = tf.gather(self.sqrt_one_minus_alpha_bar, t)        
                
        sqrt_alpha_bar = tf.reshape(sqrt_alpha_bar, shape=[batch_size, 1, 1, 1])
        sqrt_one_alpha_bar = tf.reshape(sqrt_one_alpha_bar, shape=[batch_size, 1, 1, 1])
        
        return sqrt_alpha_bar * x0 + sqrt_one_alpha_bar * noise
    
    def backward(self, xt, noise, t):
        
        sqrt_one_alpha_bar = tf.gather(self.sqrt_one_minus_alpha_bar, t)
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