import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.activations import gelu
import numpy as np

class TimeEmbedding(Layer):
    
    def __init__(self, emd_dim):
        super(TimeEmbedding, self).__init__()
        
        self.emd_dim = emd_dim
        self.deno = 1 / np.power(10000, np.arange(0, self.emd_dim, 2) / self.emd_dim)
        
        self.fc1 = Dense(emd_dim)
        self.act = gelu
        self.fc2 = Dense(emd_dim)
        
        
    def call(self, time_batch):
    
        time_batch = tf.cast(time_batch, tf.float32)
        sin_inp = tf.einsum("i,j->ij", time_batch, self.deno)
                
        emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
        emb = tf.reshape(emb, (*emb.shape[:-2], -1))
            
        x = self.fc1(emb)
        x = self.act(x)
        x = self.fc2(x)
        
        return x