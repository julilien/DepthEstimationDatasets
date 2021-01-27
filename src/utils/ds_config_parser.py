import configparser
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../.."
DS_CONFIG_FILE = 'conf/ds_conf.ini'
DS_CONFIG_FILE_ENCODING = 'utf-8-sig'


# TODO: Support app_conf
class DatasetConfigParser:
    @staticmethod
    def get_config(config_file_path=None):
        config = configparser.ConfigParser()

        if config_file_path is None:
            config_file_path = os.path.join(ROOT_DIR, DS_CONFIG_FILE)
        config.read(config_file_path, encoding=DS_CONFIG_FILE_ENCODING)
        return config

    @staticmethod
    def get_dataset_configuration(dataset_name, config_file_path=None):
        config = DatasetConfigParser.get_config(config_file_path)
        if dataset_name not in config:
            raise ValueError("Dataset {} not given in configuration.".format(dataset_name))

        return config[dataset_name]
