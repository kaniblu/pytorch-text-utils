from .. import common


class SplitWordIterator(common.Generator):
    def __init__(self, sent, delimiter=None):
        self.sent = sent
        self.delimiter = delimiter

    def _split(self, str):
        if self.delimiter is None:
            return str.split()
        else:
            return str.split(self.delimiter)

    def generate(self):
        for w in self._split(self.sent):
            yield w
