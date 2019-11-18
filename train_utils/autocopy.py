import numpy

import chainer
from chainer import functions as F
from chainer.backends import cuda


def recurse_copy(object, device):
    if isinstance(object, tuple) or isinstance(object, list):
        object = [recurse_copy(sub_object, device) for sub_object in object]
    elif isinstance(object, chainer.Variable) or isinstance(object, cuda.cupy.ndarray) or isinstance(object, numpy.ndarray):
        object = F.copy(object, device)

    return object


def maybe_copy(func):
    def decorator(self, *args, **kwargs):
        # if input data is not on the same device as chain, we copy it to the corresponding device and put it
        # back where it came from

        current_device_id = self._device_id
        new_args = []
        original_device = None
        for arg in args:
            if hasattr(arg, 'data') and hasattr(arg.data, 'device') and arg.data.device.id != current_device_id:
                original_device = arg.data.device.id
                arg = recurse_copy(arg, current_device_id)
            new_args.append(arg)

        with cuda.Device(current_device_id):
            ret_val = func(self, *new_args, **kwargs)

        if original_device is not None:
            ret_val = recurse_copy(ret_val, original_device)

        return ret_val
    return decorator


class DeviceChanger:

    def __init__(self, updater, new_device):
        self.new_device = new_device
        self.old_device = updater.device
        self.updater = updater

    def __enter__(self):
        self.updater.device = self.new_device

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.updater.device = self.old_device


def change_device_of(updater, new_device):
    return DeviceChanger(updater, new_device)
