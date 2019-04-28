import argparse
import os

import numpy as np
import tensorflow as tf
import tqdm

import dataset
import model

BATCH_SIZE = 64
CHANNEL_NUM = 24
NET_DEPTH = 7
DATASET_PATH = 'dataset.tfrecords'
BUFFER_SIZE = 3000
OPTIMIZER='adam'
LEARNING_RATE = 5e-3
NUM_STEPS = 20000
SAVE_NUM = 4
STEPS_PER_LOG = 100
STEPS_PER_SAVE = 1000
LOGDIR = './logdir/train'


def get_arguments():
    """Parse arguments from the input."""
    parser = argparse.ArgumentParser(
        description='train image deraing using Rain100H dataset')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch')
    parser.add_argument('--channel_num', type=int, default=CHANNEL_NUM,
                        help='Number of images in batch')
    parser.add_argument('--net_depth', type=int, default=NET_DEPTH,
                        help='Number of images in batch')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help='Path to the dataset')
    parser.add_argument('--ckpt_dir', default=None,
                        help='Path to the model checkpoint folder')
    parser.add_argument('--buffer_size', type=int, default=BUFFER_SIZE,
                        help='Buffer size used for shuffling examples')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate used for training')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps')
    parser.add_argument('--save_num', type=int, default=SAVE_NUM,
                        help='How many images to write to summary')
    parser.add_argument('--steps_per_log', type=int, default=STEPS_PER_LOG,
                        help='How often to save summaries')
    parser.add_argument('--steps_per_save', type=int, default=STEPS_PER_SAVE,
                        help='How often to save checkpoints')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Where to save checkpoints and summaries')

    return parser.parse_args()


def main():
    """Main entry for training process."""
    args = get_arguments()
    net = model.Model(args.channel_num, args.net_depth)
    train_dt = dataset.Dataset('train', args.dataset_path, args.batch_size,
                               shuffle=True, repeat=True)
    data_iter = train_dt.get_data_iterator()

    with tf.Session() as sess:
        O, B = data_iter.get_next()
        R = O - B
        O_R = net.forward(O)
        P = O - O_R
        for name, var in {'O': O, 'B': B, 'P': P}.items():
            tf.summary.image(name, var, max_outputs=args.save_num)

        metrics = net.get_metrics(B, P, R, O_R)
        for name, metric in metrics.items():
            tf.summary.scalar(name.upper(), metric)

        global_step = tf.Variable(0, trainable=False)
        lr_values = [num * args.learning_rate for num in [1., 0.1, 0.01]]
        lr = tf.train.piecewise_constant(
            global_step,
            boundaries=[args.num_steps * 2 // 3, args.num_steps * 8 // 9],
            values=lr_values)
        tf.summary.scalar('Learning_rate', lr)

        optimizer = tf.train.AdamOptimizer(lr)
        loss = metrics['loss']
        vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars), 0.1)
        train_op = optimizer.apply_gradients(zip(grads, vars), 
                                             global_step=global_step)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)

        saver = tf.train.Saver()
        start_step = 0

        if args.ckpt_dir is None:
            init_op = tf.group(tf.global_variables_initializer(), 
                               tf.local_variables_initializer())
            sess.run(init_op)
        else:
            if os.path.isdir(args.ckpt_dir):
                ckpt_path = tf.train.latest_checkpoint(args.ckpt_dir)
            start_step = int(ckpt_path.split('.')[0].split('-')[-1])
            saver.restore(sess, ckpt_path)

        sess.run(data_iter.initializer)
        bar = tqdm.tqdm(range(args.num_steps), total=args.num_steps, 
                        unit='step', smoothing=1.0)
        for step in bar:
            _, cur_loss, cur_summary, = sess.run([train_op, loss, summary])
            bar.set_description('Loss: ' + str(cur_loss))
            bar.refresh()

            cur_step = start_step + step + 1
            if cur_step % args.steps_per_log == 0:
                summary_writer.add_summary(cur_summary, cur_step)
            if cur_step % args.steps_per_save == 0:
                save_path = os.path.join(args.logdir, 'model')
                saver.save(sess, save_path, global_step=cur_step)


if __name__ == '__main__':
    main()
