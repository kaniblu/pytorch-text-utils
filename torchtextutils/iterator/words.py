from .. import common


class SplitWordIterator(common.Generator):
    def __init__(self, sents, delimiter=None):
        self.sents = sents
        self.delimiter = delimiter

    def _split(self, str):
        if self.delimiter is None:
            return str.split()
        else:
            return str.split(self.delimiter)

    def generate(self):
        for sent in self.sents:
            for w in self._split(sent):
                yield w
