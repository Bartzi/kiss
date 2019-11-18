from chainer.backends import cuda


class TensorboardGradientPlotter:

    timing = "pre"
    name = "TensorboardGradientPlotter"
    call_for_each_param = False

    def __init__(self, summary_writer, log_interval):
        self.summary_writer = summary_writer
        self.log_interval = log_interval
        self.iteration = 0

    def __call__(self, optimizer):
        self.iteration += 1

        if self.iteration % self.log_interval != 0:
            return

        for param_name, param in optimizer.target.namedparams(False):
            weights, gradients = cuda.to_cpu(param.array), cuda.to_cpu(param.grad)
            if weights is None or gradients is None:
                return

            self.summary_writer.add_histogram(f'localizer{param_name}/weight', weights, self.iteration)
            self.summary_writer.add_histogram(f'localizer{param_name}/gradients', gradients, self.iteration)
