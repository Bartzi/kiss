from unittest.mock import Mock

from chainer.training import IntervalTrigger


class UpdaterTrigger(IntervalTrigger):

    def __call__(self, updater):
        mocked_trainer = Mock()
        mocked_trainer.updater = updater
        return super().__call__(mocked_trainer)
