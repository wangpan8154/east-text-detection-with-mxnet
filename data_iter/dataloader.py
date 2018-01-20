import os
import logging
import numpy as np
import collections
import multiprocessing
from .sampler import SequentialSampler, RandomSampler
import math
import sys
import threading
import mxnet as mx
from .core import ExceptionWrapper
import signal


def _worker_loop(dataset, index_queue, data_queue, collate_fn, batch_size=None, extract_feat=False):
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            batch = [dataset[i] for i in batch_indices]
            batch_indices = np.array([x[2] for x in batch])
            #  check error code
            inds = list(filter(lambda i: batch[i][2] == batch_indices[i], range(len(batch_indices))))
            #  inds = filter(lambda i: batch[i] is not None, range(len(batch_indices)))
            batch = [batch[i] for i in inds]+[batch[i] for i in list(set(range(len(batch_indices)))-set(inds))]
            batch_indices = [batch_indices[i] for i in inds]+[batch_indices[i] for i in list(set(range(len(batch_indices)))-set(inds))]
            pad = 0
            if batch_size is not None:
                if not extract_feat:
                    pad = batch_size-len(inds)
                    n_miss = batch_size-len(batch)
                    if n_miss > 0:
                        if (batch_size-n_miss) > 0:
                            batch += [batch[i%(batch_size-n_miss)] for i in range(n_miss)]
                        else:
                            # TODO
                            raise NotImplementedError
                else:
                    pad = 0
            batch_data = collate_fn([x[0] for x in batch])
            batch_label = collate_fn([x[1] for x in batch])
            samples = mx.io.DataBatch(batch_data, batch_label, pad=pad, index=np.array(batch_indices))
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


def default_collate(batch):
    if type(batch[0]).__module__ == 'numpy':  # this allows to not import numpy
        ret = np.array(batch)
        return ret
        #  return mx.nd.array(np.array(batch))
    elif isinstance(batch[0], int) or isinstance(batch[0], float) or isinstance(batch[0], str):
        return np.array(batch)
    elif isinstance(batch[0], collections.Iterable):
        # if each batch element is not a tensor, then it should be a tuple
        # of tensors; in that case we collate each element in the tuple
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    else:
        raise TypeError(("batch must contain ndarrays, numbers, or lists; found {}".format(type(batch[0]))))


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn
        self.sampler = loader.sampler
        self.num_workers = loader.num_workers
        self.done_event = threading.Event()
        self.extract_feat = loader.extract_feat

        self.samples_remaining = len(self.sampler)
        self.sample_iter = iter(self.sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.Queue()
            self.data_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn, self.batch_size, self.extract_feat))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

        self.pid = os.getpid()
        #  if self.num_workers > 0:
            #  signal.signal(signal.SIGTERM, self._catch_signal)

    def __len__(self):
        return int(math.ceil(len(self.sampler) / float(self.batch_size)))

    def __next__(self):
        if self.num_workers == 0:
            # same-process loading
            if self.samples_remaining == 0:
                raise StopIteration
            indices = self._next_indices()
            batch = [self.dataset[i] for i in indices]
            indices = np.array([x[2] for x in batch])
            #  check error code
            inds = filter(lambda i: batch[i][2] == indices[i], range(len(indices)))
            #  inds = filter(lambda i: batch[i] is not None, range(len(indices)))
            batch = [batch[i] for i in inds]+[batch[i] for i in list(set(range(len(indices)))-set(inds))]
            indices = [indices[i] for i in inds]+[indices[i] for i in list(set(range(len(indices)))-set(inds))]
            pad = self.batch_size-len(inds)
            n_miss = self.batch_size-len(batch)
            if n_miss > 0:
                batch += [batch[i%(self.batch_size-n_miss)] for i in range(n_miss)]
            batch_data = self.collate_fn([x[0] for x in batch])
            batch_label = self.collate_fn([x[1] for x in batch])
            batch_data = map(mx.nd.array, batch_data)
            batch_label = map(mx.nd.array, batch_label)
            return mx.io.DataBatch(batch_data, batch_label, pad=pad, index=np.array(indices))

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _next_indices(self):
        batch_size = min(self.samples_remaining, self.batch_size)
        batch = [next(self.sample_iter) for _ in range(batch_size)]
        self.samples_remaining -= len(batch)
        return batch

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        if self.samples_remaining > 0:
            self.index_queue.put((self.send_idx, self._next_indices()))
            self.batches_outstanding += 1
            self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        batch = mx.io.DataBatch(data = list(map(mx.nd.array, batch.data)),
                                label = list(map(mx.nd.array, batch.label)),
                                pad = batch.pad,
                                index = batch.index)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0 and os.getpid() == self.pid:
            self._shutdown_workers()
            #  signal.signal(signal.SIGTERM, signal.SIG_DFL)

    def _catch_signal(self, signum, frame):
        if os.getpid() == self.pid and self.num_workers > 0:
            if signum == signal.SIGTERM:
                logging.warning('Process %d get SIGTERM. Will shutdown all workers.' % os.getpid())
                self._shutdown_workers()
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                raise Exception('DataLoaderIter (pid=%d) get SIGTERM, will shutdown and exit.' % os.getpid())

class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, the ``shuffle`` argument is ignored.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 num_workers=0, collate_fn=default_collate,
                 provide_data=None, provide_label=None, extract_feat=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.provide_data = provide_data
        self.provide_label = provide_label
        self.extract_feat = extract_feat

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        elif not shuffle:
            self.sampler = SequentialSampler(dataset)

    def reset(self):
        pass

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return int(math.ceil(len(self.sampler) / float(self.batch_size)))
