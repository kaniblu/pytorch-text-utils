"""A Collection of Pipeline Components for Data Generator

A typical autoencoder data generator pipeline consists of:
  - DirectoryReader
  - SentenceWordTokenizer
  - BatchGenerator
  - FunctionMapper (with Preprocessor)
  - MemoryPinner

As an another example, Skip-Thoughts requires an additional step in the pipeline:
  - DirectoryReader
  - SentenceWordTokenizer
  - BatchGenerator
  - FunctionMapper (with Preprocessor)
  - ContextGenerator
  - MemoryPinner
"""

import torch
import torch.utils.data as D

from torchtextutils import common
from torchtextutils.iterator import SplitWordIterator


class BatchGenerator(common.Generator):
    def __init__(self, items, batch_size, allow_residual=True):
        self.items = items
        self.batch_size = batch_size
        self.allow_residual = allow_residual

    def generate(self):
        batch = []

        for item in self.items:
            batch.append(item)

            if len(batch) < self.batch_size:
                continue

            yield batch

            del batch
            batch = []

        if self.allow_residual and batch:
            yield batch
            del batch


class SentenceWordTokenizer(common.Generator):
    def __init__(self, sents, max_length=None,
                 word_iterator=SplitWordIterator):
        self.sents = sents
        self.max_length = max_length
        self.word_iterator = word_iterator

    def generate(self):
        for sent in self.sents:
            words = list(self.word_iterator(sent))

            if self.max_length is not None and len(words) > self.max_length:
                continue

            yield words


class FunctionMapper(common.Generator):
    def __init__(self, items, func):
        self.items = items
        self.func = func

    def generate(self):
        for item in self.items:
            yield self.func(item)


class MemoryPinner(FunctionMapper):
    def __init__(self, items):
        super(MemoryPinner, self).__init__(items, self._func)

    @staticmethod
    def _pin_memory(tensor):
        return tensor.pin_memory()

    @staticmethod
    def _func(item):
        return tuple(map(MemoryPinner._pin_memory, item))


class ContextDataGenerator(common.Generator):
    def __init__(self, prep_batches,
                 n_before=1, n_after=1, predict_self=False):
        self.n_before = n_before
        self.n_after = n_after
        self.predict_self = predict_self
        self.prep_batches = prep_batches

    def generate(self):
        n_bef, n_aft = self.n_before, self.n_after

        for batch, lens in self.prep_batches:
            batch_size = len(lens)

            inp_data = batch[n_bef:len(batch) - n_aft].clone()
            inp_lens = lens[n_bef:len(batch) - n_aft].clone()

            if n_bef or n_aft:
                _offset = n_bef + n_aft + 1
                out_data = [batch[i:i + _offset].unsqueeze(1)
                            for i in range(batch_size)]
                out_lens = [lens[i:i + _offset].unsqueeze(1)
                            for i in range(batch_size)]
                out_data = torch.cat(out_data, 1)
                out_lens = torch.cat(out_lens, 1)

                if not self.predict_self:
                    if n_bef and n_aft:
                        splits = out_data[:n_bef], out_data[-n_aft:]
                        splits_lens = out_lens[:n_bef], out_lens[-n_aft:]
                        out_data = torch.cat(splits, 0)
                        out_lens = torch.cat(splits_lens, 0)
                    elif n_bef:
                        out_data = out_data[:n_bef]
                        out_lens = out_lens[:n_bef]
                    elif n_aft:
                        out_data = out_data[-n_aft:]
                        out_lens = out_lens[-n_aft:]
            else:
                out_data = batch.unsqueeze(0)
                out_lens = lens.unsqueeze(0)

            inp_data, inp_lens = inp_data.contiguous(), inp_lens.contiguous()
            out_data, out_lens = out_data.contiguous(), out_lens.contiguous()

            yield inp_data, inp_lens, out_data, out_lens


def create_generator_ae(sents, batch_size, preprocessor,
                        pin_memory=True, allow_residual=True, max_length=None,
                        word_iterator=SplitWordIterator):
    sent_tokens = SentenceWordTokenizer(sents, max_length,
                                        word_iterator=word_iterator)
    batches = BatchGenerator(sent_tokens, batch_size,
                             allow_residual=allow_residual)
    prep_batches = FunctionMapper(batches, preprocessor)

    if pin_memory:
        ret = MemoryPinner(prep_batches)
    else:
        ret = prep_batches

    return ret


def create_generator_st(sents, batch_size, preprocessor,
                        pin_memory=True, allow_residual=True, max_length=None,
                        n_before=1, n_after=1, predict_self=False,
                        word_iterator=SplitWordIterator):
    sent_tokens = SentenceWordTokenizer(sents, max_length,
                                        word_iterator=word_iterator)
    batches = BatchGenerator(sent_tokens, batch_size,
                             allow_residual=allow_residual)
    prep_batches = FunctionMapper(batches, preprocessor)
    context_data = ContextDataGenerator(prep_batches, n_before, n_after,
                                        predict_self=predict_self)

    if pin_memory:
        ret = MemoryPinner(context_data)
    else:
        ret = prep_batches

    return ret
