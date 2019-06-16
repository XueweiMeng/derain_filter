import argparse
import cv2
import os

import tensorflow as tf
import tqdm


def get_arguments():
    """Parse the arguments from the command line.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Script for converting derain image pairs to dataset')
    parser.add_argument('--images_folder', default='./dataset/images',
                        help='Folder with Rain100 images')
    parser.add_argument('--dataset_folder', default='./tfrecords',
                        help='Folder where to save dataset examples')
    parser.add_argument('--crop_size', type=int, default=64,
                        help='Crop size for extracted patches.')
    parser.add_argument('--stride', type=int, default=32,
                        help='Crop stride for extracted patches.')
    parser.add_argument('--mode', default='patch', choices=['patch', 'full'],
                        help=('Dataset mode. \'patch\' for training',
                              '\'full\' for test.'))

    return parser.parse_args()


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _convert_one_samp(file_path_rain, file_path_norain, mode, crop_size, stride):
    """Proprocess the image pair of rain image(O) and background(B).

    Args:
        file_path_rain: The file path of the rain image.
        file_path_norain: The file path of the norain image,
        crop_size: Crop size for extracted patches.
        stride: Sliding window stride for cropping.

    Yields:
        Sample with preprocessed O and B.
    
    Raises:
        ValueError: Width of the image pair are odd.
    """
    image_rain = cv2.imread(file_path_rain)
    image_norain = cv2.imread(file_path_norain)
    image_rain = cv2.cvtColor(image_rain, cv2.COLOR_BGR2RGB)
    image_norain = cv2.cvtColor(image_norain, cv2.COLOR_BGR2RGB)
    height, width = image_rain.shape[:-1]
    height_no, width_no = image_norain.shape[:-1]
    if width != width_no or height != height_no:
        raise ValueError('The size of rain image should be the same as norain image\'s')

    B, O = image_norain[:, :], image_rain[:, :]
    if mode == 'full':
        crop_height, crop_width = height, width
        stride_height, stride_width = height, width
    else:
        crop_height = crop_width = crop_size
        stride_height, stride_width = stride, stride

    file_name = os.path.split(file_path_rain)[-1]
    format_ = file_name.split('.')[-1]

    for start_row in range(0, height, stride_height):
        for start_col in range(0, width, stride_width):
            end_row = min(height, start_row + crop_height)
            end_col = min(width, start_col + crop_width)
            start_row = max(0, end_row - crop_height)
            start_col = max(0, end_col - crop_width)

            B_patch = B[start_row: end_row, start_col:end_col]
            O_patch = O[start_row: end_row, start_col:end_col]
            O_data, B_data = O_patch.tostring(), B_patch.tostring()
            yield tf.train.Example(features=tf.train.Features(feature={
                'O': _bytes_list_feature(O_data),
                'B': _bytes_list_feature(B_data),
                'height': _int64_list_feature(crop_height),
                'width': _int64_list_feature(crop_width),
            }))

     
def main():
    args = get_arguments()

    if not os.path.exists(args.images_folder):
        raise FileNotFoundError('Folder %s not found!' % args.images_folder)
    if not os.path.exists(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    print('Start building the derain dataset.')

    examples_num = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(args.dataset_folder,
                                         'dataset.tfrecords'))
    rain_folder = os.path.join(args.images_folder, 'rain');
    rain_folder = os.path.join(rain_folder, 'X2')
    norain_folder = os.path.join(args.images_folder, 'norain');
    image_names = os.listdir(os.path.join(rain_folder))
    image_nums = len(image_names)
    for i in tqdm.tqdm(range(image_nums), total=image_nums, unit='image'):
        image_path_rain = os.path.join(rain_folder, image_names[i])
        image_rain_name = os.path.splitext(image_names[i])[0]
        image_norain_name = image_rain_name[0:len(image_rain_name) - 2] + os.path.splitext(image_names[i])[-1]

        image_path_norain = os.path.join(norain_folder, image_norain_name)
        example_generator = _convert_one_samp(image_path_rain, image_path_norain, args.mode,
                                              args.crop_size, args.stride)
        for example in example_generator:
            writer.write(example.SerializeToString())
            examples_num += 1

    print('Dataset prepared. Total number of examples: ' + str(examples_num))
    writer.close()


if __name__ == '__main__':
    main()
