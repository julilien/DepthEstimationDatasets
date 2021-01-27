import tensorflow as tf
import numpy as np
import os


class TFRecordWriter:
    """
    This class can be used to create a new tf record dataset.

    After initializing the writer you can add entries by calling "add_new_entry". This will fill the current tfrecord
    file or create a new one if the n_examples_per_file is reached
    """

    def __init__(self, path, n_examples_per_file):
        self.path = path
        self.n_examples_per_file = n_examples_per_file

        # Initialize the first file
        self.current_file_index = -1
        self.open_new_file()
        self.example_index = 0

    def open_new_file(self):
        self.current_file_index += 1
        record_file_name = f"{self.current_file_index:06d}.tfrecord"
        self.writer = tf.io.TFRecordWriter(os.path.join(self.path, record_file_name))

    def add_new_entry(self, image, depth):
        """
        ToDo: Make this generic and accept other data (intrinsics, semantic maps, ...)
        :param image: numpy array with shape [H, W, 3] as uint8
        :param depth: numpy array with shape [H, W] as float
        """
        # Check if the limit of the current tf record file is reached
        if self.example_index != 0 and self.example_index % self.n_examples_per_file == 0:
            self.open_new_file()

        # Encode image as png and store depth as float16
        image_bytes = tf.io.encode_png(image).numpy()
        depth_bytes = depth.astype(np.float16).tobytes()

        # Create feature
        height, width = image.shape[:2]
        feature = dict(
            image=tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
            depth=tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth_bytes])),
            height=tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            width=tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        )
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Write to file
        self.writer.write(example.SerializeToString())
        self.example_index += 1

    def close(self):
        self.writer = None


if __name__ == "__main__":
    image_example = np.array(np.random.rand(1000, 1000, 3) * 255, dtype=np.uint8)
    depth_example = np.array(np.random.rand(1000, 1000), dtype=np.float32)

    writer = TFRecordWriter("..", n_examples_per_file=5)
    for i in range(20):
        writer.add_new_entry(image_example, depth_example)
    writer.close()
