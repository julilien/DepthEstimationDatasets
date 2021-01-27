import abc
from glob import glob

from src.utils.ds_config_parser import DatasetConfigParser
from src.utils.tf_record_reader import load_tfrecord_dataset

TRAIN_PATH_LIT = "TRAIN_PATH"
VAL_PATH_LIT = "VAL_PATH"
TEST_PATH_LIT = "TEST_PATH"


class DatasetDAO:
    def __init__(self, dataset_name, config_file_path=None):
        self.config = DatasetConfigParser.get_dataset_configuration(dataset_name, config_file_path=config_file_path)

    @abc.abstractmethod
    def get_meta_information(self):
        # TODO: Pass dict or something? Where to gather this? Maybe also by relying on the configuration file?
        pass

    @staticmethod
    def read_record(root_path):
        file_names = glob(f"{root_path}/*.tfrecord")
        # TODO: Specify num workers and batch size (using app_conf!?)
        return load_tfrecord_dataset(file_names, batch_size=5, num_workers=1)

    def provide_train_split(self):
        return DatasetDAO.read_record(self.config[TRAIN_PATH_LIT])

    def provide_val_split(self):
        return DatasetDAO.read_record(self.config[VAL_PATH_LIT])

    def provide_test_split(self):
        return DatasetDAO.read_record(self.config[TEST_PATH_LIT])

    def has_train_split(self):
        return TRAIN_PATH_LIT in self.config and self.config[TRAIN_PATH_LIT] != ""

    def has_val_split(self):
        return VAL_PATH_LIT in self.config and self.config[VAL_PATH_LIT] != ""

    def has_test_split(self):
        return TEST_PATH_LIT in self.config and self.config[TEST_PATH_LIT] != ""

    @abc.abstractmethod
    def convert_dataset_to_tf_records(self, src_root_path, trgt_root_path):
        """
        Implementation depends on dataset files. Intended to be executed only once to construct tf records.
        :return:
        """
        pass
