"""Text data reader.

This can handle variable-length text data."""

import os
import io
import glob

import numpy as np
import torch

from .ipo import ImmutablePropertiesObject


class DirectoryTextFilesReader(ImmutablePropertiesObject):
    def __init__(self, data_dir):
        super(DirectoryTextFilesReader, self).__init__(
            data_dir=data_dir
        )

    def __iter__(self):
        return self.generate()

    def generate(self):
        for filename in glob.glob(os.path.join(self.data_dir, "*.txt")):
            with io.open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()


class DataGenerator(ImmutablePropertiesObject):
    def __init__(self, datagen, vocab, batch_size, max_length,
                 preprocessor):
        super(DataGenerator, self).__init__(
            datagen=datagen,
            batch_size=batch_size,
            max_length=max_length,
            vocab=vocab,
            preprocessor=preprocessor
        )


    def __iter__(self):
        return self.generate()

    def generate(self):
        batch = []

        for line in self.datagen:
            words = line.split()

            if len(words) > self.max_length:
                continue

            batch.append(words)

            if len(batch) < self.batch_size:
                continue

            batch_length = [len(s) + 2 for s in batch]
            input_data = torch.LongTensor(self.preprocessor(batch, True))
            target_data = torch.LongTensor(self.preprocessor(batch, False))
            length = torch.LongTensor(batch_length)

            length, idx = torch.sort(length, dim=0, descending=True)
            input_data = input_data[idx]
            target_data = target_data[idx]

            input_data = input_data.pin_memory()
            target_data = target_data.pin_memory()

            yield input_data, target_data, length

            del batch
            batch = []

        del batch
