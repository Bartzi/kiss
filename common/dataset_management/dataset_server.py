import configparser
import inspect
import os
from multiprocessing.managers import SyncManager

import numpy

SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 15000
SERVER_PASSPHRASE = b'chainer_memory_manager'


class SharedNPZMemoryManager(SyncManager):
    pass


class DatasetClient:

    def __getattribute__(self, item):
        if str(item).startswith("get_"):
            assert self.memory_manager is not None
            value = getattr(object.__getattribute__(self, 'memory_manager'), item)
        else:
            value = object.__getattribute__(self, item)
        return value

    def __init__(self):
        self.address = SERVER_ADDRESS
        self.port = SERVER_PORT
        self.passphrase = SERVER_PASSPHRASE

        self.register_methods()
        self.memory_manager = SharedNPZMemoryManager(address=(self.address, self.port), authkey=self.passphrase)

    def register_methods(self):
        method_names = inspect.getmembers(DatasetServer)
        method_names = [method_name for method_name, _ in method_names if method_name.startswith("get_")]
        for method_name in method_names:
            SharedNPZMemoryManager.register(method_name)

    def connect(self):
        self.memory_manager.connect()
        print("Connected to Memory Manager")


class DatasetServer:

    def __init__(self, args):
        self.datasets = self.parse_config(args.config)

        self.address = SERVER_ADDRESS
        self.port = SERVER_PORT
        self.passphrase = SERVER_PASSPHRASE

        self.open_datasets = {}
        self.memory_blocks = []

    def parse_config(self, file_name):
        config = configparser.ConfigParser()
        config.read(file_name)

        datasets = {}

        for key in config['PATHS']:
            value = config['PATHS'][key]
            if os.path.splitext(value)[-1] != '.npz':
                # we currently only support npz loading
                continue
            if len(value) == 0:
                value = None
            datasets[key] = value

        for key in config["TEST_DATASETS"]:
            value = config['TEST_DATASETS'][key]
            if len(value) == 0:
                value = None
            datasets[f"test_dataset_{key}"] = value

        return datasets

    def __enter__(self):
        self.register_methods()
        self.memory_manager = SharedNPZMemoryManager(address=(self.address, self.port), authkey=self.passphrase)
        self.memory_manager.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.memory_manager is not None:
            self.memory_manager.shutdown()

        self.memory_manager = None

    def load_npz_file(self, base_name, file_path):
        with numpy.load(file_path, allow_pickle=True) as gt_data:
            datasets = {key: gt_data[key] for key in gt_data.keys()}
            shared_datasets = {}
            for key, data in datasets.items():
                print(f"setting up: {base_name} -> {key}")
                array = numpy.ndarray(data.shape, dtype=data.dtype)

                if len(array.shape) == 0:
                    # a single value array, we need to have a shape of at least 1
                    array = array.reshape(1)

                # we have to copy the data from the file otherwise we won't the data once the file is closed
                array[:] = data
                shared_datasets[key] = array

            self.open_datasets[base_name] = shared_datasets
        self.open_datasets = self.open_datasets

    def load_data(self):
        for base_name, path in self.datasets.items():
            self.load_npz_file(base_name, path)

    def register_methods(self):
        method_names = inspect.getmembers(self.__class__)
        method_names = [method_name for method_name, _ in method_names if method_name.startswith("get_")]
        for method_name in method_names:
            SharedNPZMemoryManager.register(method_name, callable=getattr(self, method_name))

    def get_dataset_names(self):
        return {key: value.keys() for key, value in self.open_datasets.items()}

    def get_data(self, base_name, dataset_name):
        return self.open_datasets[base_name][dataset_name]

    def get_data_member(self, base_name, dataset_name, index):
        return numpy.copy(self.open_datasets[base_name][dataset_name][index])

    def get_shape(self, base_name, dataset_name):
        return list(self.open_datasets[base_name][dataset_name].shape)
