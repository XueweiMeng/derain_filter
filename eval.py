import argparse
import os

import numpy as np
import tensorflow as tf
import tqdm

import dataset
import model

BATCH_SIZE = 1
CHANNEL_NUM = 24
NET_DEPTH = 7
DATASET_PATH = 'dataset.tfrecords'
SAVE_NUM = 10
LOGDIR = 'evaluation_logdir/default'
STEPS_PER_LOG = 10


def get_arguments():
    """Parse arguments from the input."""
    parser = argparse.ArgumentParser(
        description='evaluate image deraining using Rain100H dataset')
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
    parser.add_argument('--save_num', type=int, default=SAVE_NUM,
                        help='How many images to write to summary')
    parser.add_argument('--steps_per_log', type=int, default=STEPS_PER_LOG,
                        help='How often to save image summaries')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Where to save summaries')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Evaluation steps.')

    return parser.parse_args()


def main():
    """Main entry for evaluating process."""
    args = get_arguments()
    net = model.Model(args.channel_num, args.net_depth)
    eval_dt = dataset.Dataset('test', args.dataset_path, args.batch_size)
    data_iter = eval_dt.get_data_iterator()

    with tf.Session() as sess:
        O, B = data_iter.get_next()
        R = O - B
        O_R = net.forward(O)
        P = tf.clip_by_value(O - O_R, 0, 1)
        for name, var in {'O': O, 'B': B, 'P': P}.items():
            tf.summary.image(name, var, max_outputs=args.save_num)

        metrics = net.get_metrics(B, P, R, O_R)
        for name, metric in metrics.items():
            tf.summary.scalar(name.upper(), metric)

        if args.ckpt_dir is None:
            print('Path to the checkpoint file was not provided')
            exit(1)

        if os.path.isdir(args.ckpt_dir):
            ckpt_path = tf.train.latest_checkpoint(args.ckpt_dir)
        step = int(ckpt_path.split('.')[0].split('-')[-1])
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)

        logged_step = 0
        sess.run(data_iter.initializer)

        metrics_results = [[name, np.array([])] for name in metrics]
        eval_ops = [metric for metric in metrics.values()] + [summary]

        for i in tqdm.tqdm(range(args.num_steps), total=args.num_steps, 
                      unit='step'):
            results = sess.run(eval_ops)

            cur_metrics = results[:-1]
            for j in range(len(cur_metrics)):
                if len(results[j].shape) != len(metrics_results[j][1].shape):
                    new_metric = [results[j]]
                metrics_results[j][1] = np.concatenate((metrics_results[j][1], 
                                                        new_metric))

            cur_summary = results[-1]
            if (i + 1) % args.steps_per_log == 0:
                summary_writer.add_summary(cur_summary, logged_step)
                logged_step += 1

        mean_metrics = [(metric[0], np.mean(metric[1]))
                        for metric in metrics_results]
        metric_summaries = []
        for metric in mean_metrics:
            print('Mean ' + metric[0] + ': ', metric[1])
            metric_summaries.append(tf.summary.scalar(metric[0], metric[1]))

        metric_summary = tf.summary.merge(metric_summaries)
        metric_summary_res = sess.run(metric_summary)
        summary_writer.add_summary(metric_summary_res, step)


if __name__ == '__main__':
    main()
