"""Text data reader.

This can handle variable-length text data."""

import os
import io
import glob
import random

import numpy as np
import torch
import torch.utils.data as D

from .ipo import ImmutablePropertiesObject


class TextFileReader(ImmutablePropertiesObject):
    def __init__(self, path, shuffle_files=False):
        super(TextFileReader, self).__init__(
            path=path,
            shuffle=shuffle_files,
        )

    def __iter__(self):
        return self.generate()

    def generate(self):
        if os.path.isfile(self.path):
            filenames = [os.path.abspath(self.path)]
        elif os.path.isdir(self.path):
            filenames = glob.glob(os.path.join(self.data_dir, "*.txt"))
        else:
            raise ValueError("Path does not exist: {}".format(self.path))

        if self.shuffle:
            random.shuffle(filenames)

        for filename in filenames:
            with io.open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()


class DataGenerator(ImmutablePropertiesObject):
    def __init__(self, datagen, vocab, batch_size, max_length, preprocessor,
                 n_before=0, n_after=0, predict_self=True, pin_memory=True):
        assert n_before or n_after or predict_self

        super(DataGenerator, self).__init__(
            datagen=datagen,
            batch_size=batch_size,
            max_length=max_length,
            vocab=vocab,
            preprocessor=preprocessor,
            n_before=n_before,
            n_after=n_after,
            predict_self=predict_self,
            pin_memory=pin_memory
        )

    def __iter__(self):
        return self.generate()

    def generate(self):
        n_bef, n_aft = self.n_before, self.n_after
        batch = []
        f_batch_size = self.batch_size + self.n_before + self.n_after

        for line in self.datagen:
            words = line.split()

            if len(words) > self.max_length:
                continue

            batch.append(words)

            if len(batch) < f_batch_size:
                continue

            batch, lens = self.preprocessor(batch)
            inp_data = batch[n_bef:-n_aft].clone()
            inp_lens = lens[n_bef:-n_aft].clone()

            _offset = n_bef + n_aft + 1
            out_data = [batch[i:i + _offset].unsqueeze(0)
                        for i in range(self.batch_size)]
            out_lens = [lens[i:i + _offset].unsqueeze(0)
                        for i in range(self.batch_size)]
            out_data = torch.cat(out_data, 0)
            out_lens = torch.cat(out_lens, 0)

            if not self.predict_self:
                splits = out_data[:, :n_bef], out_data[:, -n_aft:]
                splits_lens = out_lens[:, :n_bef], out_lens[:, -n_aft:]
                out_data = torch.cat(splits, 1)
                out_lens = torch.cat(splits_lens, 1)

            print(out_data.size(), out_lens.size())
            out_data = out_data.transpose(1, 0)
            out_lens = out_lens.transpose(1, 0)

            if self.pin_memory:
                print(inp_data.size(), inp_lens.size())
                inp_data, inp_lens = inp_data.pin_memory(), inp_lens.pin_memory()
                out_data, out_lens = out_data.pin_memory(), out_lens.pin_memory()

            yield inp_data, inp_lens, out_data, out_lens

            del batch
            batch = []

        del batch
