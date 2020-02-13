import chainer


class SubDataset(chainer.datasets.SubDataset):

    def __getattribute__(self, item):
        try:
            v = object.__getattribute__(self, item)
        except AttributeError:
            v = getattr(object.__getattribute__(self, '_dataset'), item)
        return v
