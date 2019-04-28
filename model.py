import functools
import tensorflow as tf

conv2d = functools.partial(tf.layers.conv2d, padding='same', 
                           kernel_initializer=tf.keras.initializers.he_normal())


class Model(object):
    def __init__(self, channel=24, depth=7):
        super(Model, self).__init__()
        self.channel = channel
        self.depth = depth

    def forward(self, O):
        with tf.variable_scope('can'):
            x = conv2d(O, self.channel, 3, activation=tf.nn.leaky_relu,
                       name='enc')
            for i in range(self.depth - 3):
                dilation = 2 ** i
                x = conv2d(x, self.channel, 3, activation=tf.nn.leaky_relu,
                           dilation_rate=(dilation, dilation), 
                           name='conv'+str(i))
            x = conv2d(x, self.channel, 3, activation=tf.nn.leaky_relu,
                       name='dec1')
            O_R = conv2d(x, 3, 1, activation=None, name='dec2')

        return O_R

    def get_metrics(self, B, P, R, O_R):
        metrics = {
            'loss': tf.losses.mean_squared_error(R, O_R),
            'psnr': tf.reduce_mean(tf.image.psnr(B, P, max_val=1.0)),
            'ssim': tf.reduce_mean(tf.image.ssim(B, P, max_val=1.0)),
        }

        return metrics

