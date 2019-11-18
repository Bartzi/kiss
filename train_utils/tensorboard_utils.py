from chainer.training.extensions import Evaluator


class TensorboardEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        self.tensorboard_handle = kwargs.pop('tensorboard_handle')
        self.base_key = kwargs.pop('base_key', 'eval')
        super().__init__(*args, **kwargs)

    def __call__(self, trainer=None):
        summary = super().__call__(trainer=trainer)
        self.log_summary(trainer, summary)
        return summary

    def log_summary(self, trainer, summary):
        for key, value in summary.items():
            self.tensorboard_handle.add_scalar('/'.join([self.base_key, key]), value, trainer.updater.iteration if trainer is not None else None)
