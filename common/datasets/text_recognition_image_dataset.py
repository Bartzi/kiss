import json

import numpy as np

from common.datasets.image_dataset import BaseNPZImageDataset
from train_utils.dataset_utils import retry_get_example


class TextRecognitionImageDataset(BaseNPZImageDataset):

    def __init__(self, *args, **kwargs):
        char_map = kwargs.pop('char_map')
        self.label_dtype = kwargs.pop('label_dtype', np.int32)

        with open(char_map) as f:
            self.char_map = json.load(f)
        self.blank_label = self.char_map['metadata']['blank_label']
        self.bos_token = self.char_map['metadata'].get('bos_token', None)

        del self.char_map['metadata']
        self.reverse_char_map = {v: k for k, v in self.char_map.items()}
        super().__init__(*args, **kwargs)

        self.num_chars = int(self.get_gt_item('num_chars', 0))
        self.num_words = int(self.get_gt_item('num_words', 0))

    # if you think this is stupid: I was just too lazy to recreate the NPZ files!!!
    @property
    def num_chars_per_word(self):
        return self.num_words

    # same here!
    @property
    def num_words_per_image(self):
        return self.num_chars

    @property
    def num_classes(self):
        return len(self.char_map)

    def get_word(self, i):
        return str(self.get_gt_item('text', i))

    def get_words(self, i):
        word = self.get_word(i)
        labels = [np.array([int(self.reverse_char_map[character])], dtype=self.label_dtype) for character in word]
        labels += [np.full_like(labels[0], self.blank_label)] * (self.num_chars_per_word - len(labels))
        return np.stack(labels, axis=1), len(word)

    def decode_chars(self, chars):
        word = ''.join(self.char_map[str(char)] for char in chars).strip(self.char_map[str(self.blank_label)])
        return word

    @retry_get_example
    def get_example(self, i):
        image = super().get_example(i)

        words, num_words = self.get_words(i)
        return {
            "image": image,
            "words": words,
            "num_words": np.array(num_words, dtype='int32'),
        }
