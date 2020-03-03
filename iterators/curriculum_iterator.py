import numpy
from chainer.iterators import MultithreadIterator, _statemachine


class CurriculumIterator(MultithreadIterator):

    def __init__(self, *args, **kwargs):
        self.curriculum_shift_intervals = kwargs.pop('curriculum_shift_intervals')
        super().__init__(*args, **kwargs)

        assert hasattr(self.dataset, 'level_up'), "Dataset for Curriculum iterator needs to have a level_up method!"

    def _invoke_prefetch(self):
        if self.is_new_epoch and self.epoch in self.curriculum_shift_intervals:
            self.dataset.level_up()
            if self.order_sampler is None:
                order = None
            else:
                order = self.order_sampler(numpy.arange(len(self.dataset)), 0)

            self._state = _statemachine.IteratorState(
                self._state.current_position,
                self._state.epoch,
                self._state.is_new_epoch,
                order,
            )

        super()._invoke_prefetch()
