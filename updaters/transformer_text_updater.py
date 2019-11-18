from unittest.mock import Mock

import chainer
import chainer.functions as F
import numpy
from chainer.backends import cuda
from chainer.training import get_trigger

from common.utils import DirectionLossCalculator, OutOfImageLossCalculator, Size
from train_utils.disable_chain import disable_chains
from train_utils.autocopy import change_device_of

num_word_kernel = cuda.cupy.ElementwiseKernel(
    "T element, raw int32 numWordIndicator, int32 numBoxes, T wordValue, T noWordValue",
    "T outElement",
    """
        // determine current batch position and also position in word
        int wordPosition = i % numBoxes;
        int batchPosition = i / numBoxes;

        // determine at which position in the word we have to change the label
        T labelIndicator = numWordIndicator[batchPosition];

        outElement = wordPosition < labelIndicator ? wordValue : noWordValue; 
    """,
    name="num_word_iou"
)


class TransformerTextRecognitionUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.localizer, self.recognizer = kwargs.pop('models')
        self.tensorboard_handle = kwargs.pop('tensorboard_handle', None)
        tensorboard_log_interval = kwargs.pop('tensorboard_log_interval', (1, 'iteration'))
        self.recognizer_update_interval = kwargs.pop('recognizer_update_interval', 1)
        self.tensorboard_trigger = get_trigger(tensorboard_log_interval)

        self.mocked_trainer = Mock()
        self.mocked_trainer.updater = self

        super().__init__(*args, **kwargs)

        self.regularizers = [
            DirectionLossCalculator(self.localizer.xp),
            OutOfImageLossCalculator(self.localizer.xp),
        ]

    def update_core(self):
        with self.device.device:
            loss_localizer = self.update_localizer()
            # loss_recognizer = self.update_recognizer()

            self.log_results({
                "loss/localizer": loss_localizer,
                # "loss/recognizer": loss_recognizer
            })

    def update_localizer(self):
        batch = next(self.get_iterator('main'))
        batch = self.converter(batch, self.device)
        xp = self.localizer.xp

        localizer_output, bboxes = self.localizer(batch['image'])
        recognizer_output = self.recognizer(localizer_output, batch['words'])
        batch_size, num_boxes, num_channels, out_height, out_width = localizer_output.shape

        loss_localizer = self.recognizer.calc_loss(recognizer_output, batch['words'])

        if "num_words" in batch:
            bboxes = self.clear_bboxes(bboxes, batch['num_words'], xp)
        bboxes = F.reshape(bboxes, (-1, 2, out_height, out_width))

        for regularizer in self.regularizers:
            loss_localizer += regularizer.calc_loss(bboxes, Size._make(batch['image'].shape[-2:]), normalize=True)

        chains_to_disable = []
        if self.iteration % self.recognizer_update_interval != 0:
            chains_to_disable.append(self.recognizer)

        localizer_optimizer = self.get_optimizer('opt_gen')
        recognizer_optimizer = self.get_optimizer('opt_rec')

        with disable_chains(chains_to_disable):
            self.localizer.cleargrads()
            self.recognizer.cleargrads()
            loss_localizer.backward()
            localizer_optimizer.update()
            recognizer_optimizer.update()

        localizer_losses = {
            'loss': loss_localizer,
        }
        return localizer_losses

    def log_results(self, losses):
        def log(base_name, losses):
            for loss_type, loss_value in losses.items():
                chainer.reporter.report({f"{base_name}/{loss_type}": loss_value})
                if self.tensorboard_handle is not None and self.tensorboard_trigger(self.mocked_trainer):
                    self.tensorboard_handle.add_scalar(f"{base_name}/{loss_type}", float(loss_value.data),
                                                       self.iteration)

        for loss_key, loss_value in losses.items():
            log(loss_key, loss_value)

    def update_recognizer(self):
        recognizer_optimizer = self.get_optimizer('opt_rec')

        batch = next(self.get_iterator('rec'))
        batch = self.converter(batch, self.device)

        recognizer_output = self.recognizer(
            batch['image'],
            batch['words'],
        )
        loss = self.recognizer.calc_loss(recognizer_output, batch['words'])

        self.recognizer.cleargrads()
        loss.backward()
        recognizer_optimizer.update()

        recognizer_losses = {
            'loss': loss
        }
        return recognizer_losses

    def build_iou_labels(self, shape, num_word_indicator, xp):
        base_array = xp.empty(shape, dtype=chainer.get_dtype())
        if xp == numpy:
            labels = self.fill_iou_array_cpu(base_array, num_word_indicator)
        else:
            labels = self.fill_iou_array_gpu(base_array, num_word_indicator)
        return xp.ravel(labels)

    def fill_iou_array_cpu(self, base_array, num_word_indicator):
        for i in range(len(base_array)):
            base_array[i, :num_word_indicator[i]] = self.localizer_iou_target
            base_array[i, num_word_indicator[i]:] = 0
        return base_array

    def fill_iou_array_gpu(self, base_array, num_word_indicator):
        return num_word_kernel(
            base_array,
            num_word_indicator.astype('int32'),
            base_array.shape[1],
            self.localizer_iou_target,
            0
        )

    def clear_bboxes(self, bboxes, num_word_indicators, xp):
        condition = xp.ones_like(bboxes.array)
        for i in range(len(bboxes)):
            condition[i, num_word_indicators[i]:] = 0

        return F.where(condition.astype(xp.bool), bboxes, xp.zeros_like(bboxes.array))


class TransformerRecognizerOnlyUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.recognizer = kwargs.pop('model')
        self.tensorboard_handle = kwargs.pop('tensorboard_handle', None)
        tensorboard_log_interval = kwargs.pop('tensorboard_log_interval', (1, 'iteration'))
        self.tensorboard_trigger = get_trigger(tensorboard_log_interval)

        self.mocked_trainer = Mock()
        self.mocked_trainer.updater = self

        super().__init__(*args, **kwargs)

    def update_core(self):
        with self.device.device:
            loss_recognizer = self.update_recognizer()

            self.log_results({
                "loss/recognizer": loss_recognizer
            })

    def log_results(self, losses):
        def log(base_name, losses):
            for loss_type, loss_value in losses.items():
                chainer.reporter.report({f"{base_name}/{loss_type}": loss_value})
                if self.tensorboard_handle is not None and self.tensorboard_trigger(self.mocked_trainer):
                    self.tensorboard_handle.add_scalar(f"{base_name}/{loss_type}", float(loss_value.data),
                                                       self.iteration)

        for loss_key, loss_value in losses.items():
            log(loss_key, loss_value)

    def update_recognizer(self):
        recognizer_optimizer = self.get_optimizer('opt_rec')

        batch = next(self.get_iterator('main'))
        batch = self.converter(batch, self.device)

        recognizer_output = self.recognizer(
            batch['image'],
            batch['words'].squeeze()
        )
        loss = self.recognizer.calc_loss(recognizer_output, batch['words'])

        batch_size, num_chars, num_classes = recognizer_output.shape
        recognizer_output = F.reshape(recognizer_output, (-1, num_classes))
        char_accuracy = F.accuracy(F.softmax(recognizer_output, axis=1), batch['words'].ravel())

        self.recognizer.cleargrads()
        loss.backward()
        recognizer_optimizer.update()

        recognizer_losses = {
            'loss': loss,
            'char_accuracy': char_accuracy,
        }
        return recognizer_losses
