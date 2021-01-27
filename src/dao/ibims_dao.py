import tensorflow as tf
from scipy import io
import os

from src.dao.dao import DatasetDAO
from src.utils.tf_record_writer import TFRecordWriter


class IbimsDatasetDAO(DatasetDAO):
    def get_meta_information(self):
        raise NotImplementedError("Not yet implemented.")

    @staticmethod
    def read_raw_mat(file_path):
        raw_data = io.loadmat(file_path)['data']

        image = raw_data[0][0][2]
        depth = raw_data[0][0][3]

        return image, depth.astype(np.float32)

    def convert_dataset_to_tf_records(self, src_root_path, trgt_root_path):
        file_names = [s for s in tf.data.Dataset.list_files(os.path.join(src_root_path, '*.mat'),
                                                            shuffle=False).as_numpy_iterator()]

        # TODO: Use global parameter for n_examples_per_file
        writer = TFRecordWriter(trgt_root_path, n_examples_per_file=50)

        for file_name in file_names:
            image, depth = IbimsDatasetDAO.read_raw_mat(file_name)
            writer.add_new_entry(image, depth)
        writer.close()


if __name__ == "__main__":
    dao = IbimsDatasetDAO("IBIMS")
    # dao.convert_dataset_to_tf_records("data/Images/Ibims",
    #                                   "data/tf_records/ibims/test")
    test_ds = dao.provide_test_split()
    test_ds_it = test_ds.as_numpy_iterator()
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(2):
        elem = next(test_ds_it)
        plt.title("Image Train {}".format(i))
        plt.imshow(np.squeeze(elem["image"][0]))
        plt.show()
        plt.title("GT Train {}".format(i))
        plt.imshow(np.squeeze(elem["depth"][0]).astype(np.float32))
        plt.show()
