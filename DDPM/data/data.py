import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X, y), (X_test, y_test) = mnist.load_data()
X, X_test = X / 255 * 2 - 1, X_test / 255.0 * 2 - 1
X = X.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X = tf.image.resize(
    X,
    (32, 32),
    method='bilinear',
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)

def get_data(batch_size):

    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    return dataset