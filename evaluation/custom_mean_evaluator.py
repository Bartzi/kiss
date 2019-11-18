import copy

from chainer import reporter as reporter_module, function
from chainer.training.extensions import Evaluator


class CustomMeanEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        self.max_num_iterations = kwargs.pop('num_iterations', None)
        super().__init__(*args, **kwargs)

    def calculate_mean_of_summary(self, summary):
        raise NotImplementedError

    def fixed_num_iterations_iterator(self, it):
        try:
            for _ in range(self.max_num_iterations):
                yield next(it)
        except StopIteration:
            pass

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Note that this function assumes that the main iterator raises
        ``StopIteration`` or code in the evaluation loop raises an exception.
        So, if this assumption is not held, the function could be caught in
        an infinite loop.

        Users can override this method to customize the evaluation routine.

        .. note::

            This method encloses :attr:`eval_func` calls with
            :func:`function.no_backprop_mode` context, so all calculations
            using :class:`~chainer.FunctionNode`\\s inside
            :attr:`eval_func` do not make computational graphs. It is for
            reducing the memory consumption.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
            :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        if self.max_num_iterations is not None:
            it = self.fixed_num_iterations_iterator(it)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)

        return self.calculate_mean_of_summary(summary)
