import tensorflow as tf


def image_prepocess(image_path):
    # 图片解码
    image_raw_data = tf.gfile.FastGFile(image_path, 'r').read()
    image_data = tf.image.decode_jpeg(image_raw_data)

    # 转换为灰度图
    image_grayscale_data = tf.image.rgb_to_grayscale(image_data)

    # 中心截取

    # 压缩图片的值为0～1
    image_grayscale_data = tf.clip_by_value(image_grayscale_data, 0.0, 1.0)

    return image_grayscale_data
