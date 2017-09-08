import os
import io
import glob
import random

from torchtextutils import common


class DirectoryReader(common.Generator):
    """Sentence Generator

    Assumes that each line of the file or the files contained in the directory
    is a sentence."""

    def __init__(self, path, shuffle_files=False):
        self.path = path
        self.shuffle = shuffle_files

    def generate(self):
        if os.path.isfile(self.path):
            filenames = [os.path.abspath(self.path)]
        elif os.path.isdir(self.path):
            filenames = glob.glob(os.path.join(self.path, "*.txt"))
        else:
            raise ValueError("Path does not exist: {}".format(self.path))

        if self.shuffle:
            random.shuffle(filenames)

        for filename in filenames:
            with io.open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()