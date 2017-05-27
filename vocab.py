"""Run this module independently as a script to prepare vocab for inference
and training.
"""

import os
import pickle
from collections import Counter

import tqdm
import configargparse as argparse

from .ipo import ImmutablePropertiesObject
from .generator import TextFileReader


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


def create_parser():
    parser = argparse.ArgParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--eos", type=str, default="<EOS>")
    parser.add_argument("--bos", type=str, default="<BOS>")
    parser.add_argument("--pad", type=str, default="<PAD>")
    parser.add_argument("--unk", type=str, default="<UNK>")
    parser.add_argument("--cutoff", type=int, default=30000)

    return parser


def populate_vocab(sents, vocab, cutoff):
    counter = Counter()

    for sent in tqdm.tqdm(sents, desc="counting words"):
        words = sent.strip().split()
        counter.update(words)

    topk = counter.most_common(cutoff)
    words = set(w for w, c in topk)

    for w in words:
        vocab.add(w)

    return vocab


def main():
    parser = create_parser()
    args = parser.parse_args()

    input_dir = args.data_dir
    output_path = args.vocab_path
    eos = args.eos
    bos = args.bos
    pad = args.pad
    unk = args.unk
    cutoff = args.cutoff

    sents = TextFileReader(input_dir)
    vocab = Vocabulary(pad=pad, eos=eos, bos=bos, unk=unk)

    print("Populating vocabulary...")
    vocab = populate_vocab(sents, vocab, cutoff)

    parent_dir = os.path.dirname(output_path)

    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    print("Dumping vocabulary...")
    with open(output_path, "wb") as f:
        pickle.dump(vocab, f)

    print("Vocabulary written to '{}'.".format(output_path))
    print("Done!")


if __name__ == '__main__':
    main()
