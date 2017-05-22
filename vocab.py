"""Vocabulary class."""

from .ipo import ImmutablePropertiesObject


class Vocabulary(ImmutablePropertiesObject):
    def __init__(self, **reserved):
        self._words = []
        self._f2i = {}
        self._i2f = {}
        self._reserved = reserved

        super(Vocabulary, self).__init__(**reserved)

        for word, token in reserved.items():
            self.add(token)

    def add(self, w, ignore_duplicates=True):
        if w in self._f2i:
            if not ignore_duplicates:
                raise ValueError("'{}' already exists "
                                 "in the vocab.".format(w))
            return self._f2i[w]

        index = len(self._words)
        self._words.append(w)

        self._f2i[w] = index
        self._i2f[index] = w

        return self._f2i[w]

    def remove(self, w):
        """
        Removes a word from the vocab. The indices are unchanged.
        """
        if w not in self._f2i:
            raise ValueError("'{}' does not exist.".format(w))

        if w in self.reserved:
            raise ValueError("'{}' is one of the reserved words, and thus"
                             "cannot be removed.".format(w))

        index = self._f2i[w]
        del self._f2i[w]
        del self._i2f[index]

        self._words.remove(w)

    def reconstruct_indices(self):
        """
        Reconstruct word indices in case of word removals.
        Vocabulary does not handle empty indices when words are removed,
          hence it need to be told explicity about when to reconstruct them.
        """
        del self._i2f, self._f2i
        self._f2i, self._i2f = {}, {}

        for i, w in enumerate(self._words):
            self._f2i[w] = i
            self._i2f[i] = w

    @property
    def words(self):
        return self._words

    @property
    def unk_tok(self):
        return self._unk_tok

    @property
    def reserved(self):
        return self._reserved

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._i2f[item]
        elif isinstance(item, str):
            return self._f2i[item]
        elif hasattr(item, "__iter__"):
            return [self[ele] for ele in item]
        else:
            raise ValueError("Unknown type: {}".format(type(item)))

    def __contains__(self, item):
        return item in self._f2i or item in self._i2f

    def __len__(self):
        return len(self._f2i)
