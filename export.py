import argparse
import collections
import enum
import os

import numpy as np
import tensorflow as tf

import model

CHANNEL_NUM = 24
NET_DEPTH = 7


@enum.unique
class Layer(enum.Enum):
    Input = 0
    Conv = 1
    Depth_to_space = 2



@enum.unique
class Activation(enum.Enum):
    Relu = 0
    Tanh = 1
    Sigmoid = 2
    Null = 3
    LeakyRelu = 4


@enum.unique
class Padding(enum.Enum):
    Valid = 0
    Same = 1


def get_arguments():
    """Parse arguments from the input."""
    parser = argparse.ArgumentParser(
        description='export c hf with model weights and binary model file')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Where to put generated files')
    parser.add_argument('--ckpt_dir', type=str, default='',
                        help='Folder to the model checkpoint')
    parser.add_argument('--channel_num', type=int, default=CHANNEL_NUM,
                        help='Number of images in batch')
    parser.add_argument('--net_depth', type=int, default=NET_DEPTH,
                        help='Number of images in batch')

    return parser.parse_args()


def write_conv_layer(kernel, bias, dilation_rate, padding, activation,
                     model_file):
    kernel = np.transpose(kernel, [3, 0, 1, 2])
    param = np.array(
        [Layer.Conv.value, dilation_rate, padding.value, activation.value,
         kernel.shape[3], kernel.shape[0], kernel.shape[1]], dtype=np.int32)
    print(param)
    param.tofile(model_file)
    kernel.tofile(model_file)
    bias.tofile(model_file)


def write_to_weight_file(weights, model_file, net_depth):
    np.array([net_depth], dtype=np.int32).tofile(model_file)
    write_conv_layer(
        kernel=weights['can/enc/kernel:0'],
        bias=weights['can/enc/bias:0'],
        dilation_rate=1,
        padding=Padding.Same,
        activation=Activation.LeakyRelu,
        model_file=model_file)
    for i in range(net_depth-3):
        write_conv_layer(
            kernel=weights['can/conv%d/kernel:0' % i],
            bias=weights['can/conv%d/bias:0' % i],
            dilation_rate=2**i,
            padding=Padding.Same,
            activation=Activation.LeakyRelu,
            model_file=model_file)
    write_conv_layer( 
        kernel=weights['can/dec1/kernel:0'],
        bias=weights['can/dec1/bias:0'],
        dilation_rate=1,
        padding=Padding.Same,
        activation=Activation.LeakyRelu,
        model_file=model_file)
    write_conv_layer( 
        kernel=weights['can/dec2/kernel:0'],
        bias=weights['can/dec2/bias:0'],
        dilation_rate=1,
        padding=Padding.Same,
        activation=Activation.Null,
        model_file=model_file)


def write_kernel_weight(h_file, values, name):
    h_file.write('\nstatic const float ' + name + '[] = {\n')

    values_flatten = values.flatten()

    max_len = 0
    for value in values_flatten:
        if len(str(value)) > max_len:
            max_len = len(str(value))

    counter = 0
    for i in range(len(values_flatten)):
        counter += 1
        if counter == 4:
            h_file.write(str(values_flatten[i]) + 'f')
            if i != len(values_flatten) - 1:
                h_file.write(',')
            h_file.write('\n')
            counter = 0
        else:
            if counter == 1:
                h_file.write('    ')
            h_file.write(str(values_flatten[i]) + 'f')
            if i != len(values_flatten) - 1:
                h_file.write(',')
            h_file.write(' ' * (1 + max_len - len(str(values_flatten[i]))))
    if counter != 0:
        h_file.write('\n')
    h_file.write('};\n')

    h_file.write('\nstatic const long int ' + name + '_dims[] = {\n')
    for i in range(len(values.shape)):
        h_file.write('    ')
        h_file.write(str(values.shape[i]))
        if i != len(values.shape) - 1:
            h_file.write(',\n')
    h_file.write('\n};\n')


def write_to_h_file(h_file, variables):
    h_file.write('/**\n')
    h_file.write(' * @file\n')
    h_file.write(' * Default cnn weights for x deraining with CAN \n')
    h_file.write(' */\n\n')

    h_file.write('#ifndef AVFILTER_DNN_CAN_H\n')
    h_file.write('#define AVFILTER_DNN_CAN_H\n')

    variables = tf.trainable_variables()
    var_dict = collections.OrderedDict()
    for variable in variables:
        var_name = variable.name.split(':')[0].replace('/', '_')
        value = variable.eval()
        if 'kernel' in var_name:
            value = np.transpose(value, axes=(3, 0, 1, 2))
        var_dict[var_name] = value

    for name, value in var_dict.items():
        write_kernel_weight(h_file, value, name)

    h_file.write('#endif\n')


def main():
    """Main entry for training process."""
    args = get_arguments()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    net = model.Model(args.channel_num, args.net_depth)

    with tf.Session() as sess:
        O = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="x")
        O_R = net.forward(O)
        P = O - O_R
        P = tf.clip_by_value(P, 0, 1, name='y')

        if os.path.isdir(args.ckpt_dir):
            ckpt_path = tf.train.latest_checkpoint(args.ckpt_dir)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        with open(os.path.join(args.output_dir, 'can.model'), 'wb') as mf:
            weights = {var.name: sess.run(var)
                       for var in tf.trainable_variables()}
            write_to_weight_file(weights, mf, args.net_depth)

        with open(os.path.join(args.output_dir, 'dnn_can.h'), 'w') as hf:
            variables = tf.trainable_variables()
            write_to_h_file(hf, variables)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['y'])
        tf.train.write_graph(output_graph_def, args.output_dir, 'can.pb', 
                             as_text=False)


if __name__ == '__main__':
    main()

