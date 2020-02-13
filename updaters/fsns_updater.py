from updaters.transformer_text_updater import TransformerTextRecognitionUpdater


class FSNSUpdater(TransformerTextRecognitionUpdater):

    def clear_bboxes(self, bboxes, num_word_indicators, xp):
        num_word_indicators = xp.reshape(num_word_indicators, (-1,) + num_word_indicators.shape[2:])
        return super().clear_bboxes(bboxes, num_word_indicators, xp)
