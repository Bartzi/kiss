from itertools import zip_longest

import os
import shutil

from chainer.training.extensions import LogReport


class Logger(LogReport):

    def __init__(self, backup_base_path, log_dir, tensorboard_writer=None, keys=None, trigger=(1, 'epoch'), postprocess=None, log_name='log', exclusion_filters=(r'*logs*',), resume=False):
        super(Logger, self).__init__(keys=keys, trigger=trigger, postprocess=postprocess, log_name=log_name)
        self.tensorboard_writer = tensorboard_writer
        if not resume:
            self.full_backup(backup_base_path, log_dir, exclusion_filters)
        self.log_dir = log_dir

    def full_backup(self, backup_base_path, log_dir, exclusion_filters):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        ignore_patterns = shutil.ignore_patterns(*exclusion_filters)
        shutil.copytree(backup_base_path, os.path.join(log_dir, 'code'), ignore=ignore_patterns)

    def __call__(self, trainer):
        observation = trainer.observation
        observation_keys = observation.keys()
        if self._keys is not None:
            observation_keys = filter(lambda x: x in self._keys, observation_keys)

        if self.tensorboard_writer is not None:
            for key in observation_keys:
                self.tensorboard_writer.add_scalar(key, observation[key].data, trainer.updater.iteration)

        super().__call__(trainer)
