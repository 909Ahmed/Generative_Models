import tensorflow as tf
from tensorflow.keras.models import Model
from .Schedulers.CosineScheduler import CosineScheduler

class DDPM(Model):
    
    def __init__(self, Unet, BATCH_SIZE):
        super(DDPM, self).__init__()
        self.unet = Unet
        self.sdlr = CosineScheduler()
        self.BATCH_SIZE = BATCH_SIZE
        
    @property
    def metrics(self):
         return [self.loss_metric]
        
    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_metric = tf.keras.metrics.MeanSquaredError(name="loss")
        
    @tf.function
    def train_step(self, images):
        
        with tf.GradientTape() as tape:
        
            noise = tf.random.normal([self.BATCH_SIZE, images.shape[1], images.shape[2], images.shape[3]])
            times = tf.random.uniform(shape=(self.BATCH_SIZE,), minval=0, maxval=1000, dtype=tf.int32)
            noisy_images = self.sdlr.forward(images, noise, times)
            
            pred_noise = self.unet(noisy_images, times)
            loss = self.loss_fn(pred_noise, noise)

        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        
        self.loss_metric.update_state(pred_noise, noise)
        return {"loss": self.loss_metric.result()}