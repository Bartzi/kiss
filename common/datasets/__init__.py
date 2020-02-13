import numpy

from common.datasets.sub_dataset import SubDataset


def scatter_dataset(dataset, comm, root=0, shuffle=False,
                    seed=None, max_buf_len=256 * 1024 * 1024):
    """Scatter the given dataset to the workers in the communicator.

    The dataset of worker ``root``
    (i.e., the worker whose ``comm.rank`` is ``root``) is
    scattered to all workers. The given dataset of other workers are ignored.
    The dataset is split to sub datasets of almost equal sizes and scattered
    to workers. To create a sub dataset, ``chainer.datasets.SubDataset`` is
    used.

    Args:
        dataset: A dataset (e.g., ``list``, ``numpy.ndarray``,
            ``chainer.datasets.TupleDataset``, ...).
        comm: ChainerMN communicator or MPI4py communicator.
        shuffle (bool): If ``True``, the order of examples is shuffled
            before being scattered.
        root (int): The root process of the scatter operation.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer being convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.
        max_buf_len (int): Max buffer size to be used at broadcasting
            binaries. Must not be larger than 2147483647.
    Returns:
        Scattered dataset.
    """

    assert 0 <= root and root < comm.size

    order = None
    if shuffle and dataset is not None:
        n_total_samples = len(dataset)
        order = numpy.random.RandomState(seed).permutation(
            n_total_samples)

    data = None
    if comm.rank == root:
        data = (dataset, order)

    data = comm.bcast_obj(data, max_buf_len=max_buf_len, root=root)
    assert data is not None
    (dataset, order) = data

    if comm.rank == root:
        mine = None
        n_total_samples = len(dataset)
        n_sub_samples = (n_total_samples + comm.size - 1) // comm.size

        for i in range(comm.size):
            b = n_total_samples * i // comm.size
            e = b + n_sub_samples

            if i == root:
                mine = SubDataset(dataset, b, e, order)
            else:
                comm.send_obj((b, e), dest=i)
        assert mine is not None
        return mine

    else:
        data = comm.recv_obj(source=root)
        assert data is not None
        (b, e) = data
        return SubDataset(dataset, b, e, order)
