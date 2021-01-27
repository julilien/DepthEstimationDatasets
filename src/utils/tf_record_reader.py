from glob import glob
import tensorflow as tf


def parse_record_entry(entry):
    """
    This function reads a single raw tfrecord entry (as returned by TFRecordDataset) and returns a dict containing
    the parsed entries
    :param entry: a dict containing entries image and depth
    """
    # For now we hardcode the entries we expect, but i'm sure there is a way to do this dynamically
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_entry = tf.io.parse_single_example(entry, image_feature_description)

    # Extract data
    height, width = parsed_entry["height"], parsed_entry["width"]
    image = tf.io.decode_png(parsed_entry["image"])
    depth = tf.io.decode_raw(parsed_entry["depth"], out_type=tf.half)
    depth = tf.reshape(depth, [height, width])
    return dict(image=image, depth=depth)


def load_tfrecord_dataset(file_names, batch_size, num_workers):
    ds = tf.data.TFRecordDataset(file_names, compression_type=None, buffer_size=int(1e7), num_parallel_reads=num_workers)
    ds = ds.map(parse_record_entry).batch(batch_size=batch_size)
    return ds


if __name__ == "__main__":
    dataset_path = ".."
    file_names = glob(f"{dataset_path}/*.tfrecord")
    ds = load_tfrecord_dataset(file_names, batch_size=5, num_workers=1)
    for entry in ds:
        print("")
