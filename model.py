import tensorflow as tf


class Model(object):
    def __init__(self, channel=24, depth=7):
        super(Model, self).__init__()
        self.channel = channel
        self.depth = depth

    def forward(self, O):
        with tf.variable_scope('can'):
            x = tf.layers.conv2d(O, self.channel, 3, padding='same',
                                 activation=tf.nn.leaky_relu, name='enc')
            for i in range(self.depth - 3):
                dilation = 2 ** i
                x = tf.layers.conv2d(x, self.channel, 3, padding='same', 
                                     dilation_rate=(dilation, dilation),
                                     activation=tf.nn.leaky_relu, 
                                     name='conv'+str(i))
            x = tf.layers.conv2d(x, self.channel, 3, padding='same', 
                                 activation=tf.nn.leaky_relu, name='dec1')
            O_R = tf.layers.conv2d(x, 3, 1, padding='same', name='dec2')

        return O_R

    def get_metrics(self, B, P, R, O_R):
        metrics = {
            'loss': tf.losses.mean_squared_error(R, O_R),
            'psnr': tf.reduce_mean(tf.image.psnr(B, P, max_val=1.0)),
            'ssim': tf.reduce_mean(tf.image.ssim(B, P, max_val=1.0)),
        }

        return metrics

