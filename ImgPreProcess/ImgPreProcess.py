import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def image_prepocess(image_path):
    # 图片解码
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read()
    image_data = tf.image.decode_jpeg(image_raw_data)

    image_grayscale_data = tf.image.rgb_to_grayscale(image_data)

    image_grayscale_data = tf.image.central_crop(image_grayscale_data, 0.4)

    # 设置图片尺寸
    image_grayscale_data = tf.image.resize_images(image_grayscale_data, [28, 28], method=0)

    return image_grayscale_data


if __name__ == '__main__':
    image_path = '1861559547146.jpg'

    with tf.Session() as sess:
        plt.imshow(image_prepocess(image_path).eval()[:,:,0], cmap='gray')
        plt.show()
