"""Run this module independently as a script to prepare vocab for inference
and training.
"""

import os
import pickle
from collections import Counter

import yaap as argparse
import tqdm

from .iterator import DirectoryReader
from .iterator import SplitWordIterator


class Vocabulary(object):
    def __init__(self):
        self.words = []
        self.f2i = {}
        self.i2f = {}

    def add(self, w, ignore_duplicates=True):
        if w in self.f2i:
            if not ignore_duplicates:
                raise ValueError("'{}' already exists "
                                 "in the vocab.".format(w))
            return self.f2i[w]

        index = len(self.words)
        self.words.append(w)

        self.f2i[w] = index
        self.i2f[index] = w

        return self.f2i[w]

    def remove(self, w):
        """
        Removes a word from the vocab. The indices are unchanged.
        """
        if w not in self.f2i:
            raise ValueError("'{}' does not exist.".format(w))

        if w in self.reserved:
            raise ValueError("'{}' is one of the reserved words, and thus"
                             "cannot be removed.".format(w))

        index = self.f2i[w]
        del self.f2i[w]
        del self.i2f[index]

        self.words.remove(w)

    def reconstruct_indices(self):
        """
        Reconstruct word indices in case of word removals.
        Vocabulary does not handle empty indices when words are removed,
          hence it need to be told explicity about when to reconstruct them.
        """
        del self.i2f, self.f2i
        self.f2i, self.i2f = {}, {}

        for i, w in enumerate(self.words):
            self.f2i[w] = i
            self.i2f[i] = w

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2f[item]
        elif isinstance(item, str):
            return self.f2i[item]
        elif hasattr(item, "__iter__"):
            return [self[ele] for ele in item]
        else:
            raise ValueError("Unknown type: {}".format(type(item)))

    def __contains__(self, item):
        return item in self.f2i or item in self.i2f

    def __len__(self):
        return len(self.f2i)


def create_parser():
    parser = argparse.ArgParser()
    parser.add_argument("--data-dir", action="append", type=argparse.path,
                        required=True, help="")
    parser.add_argument("--vocab-path", type=argparse.path, required=True)
    parser.add_argument("--reserved", action="append", type=str)
    parser.add_argument("--cutoff", type=int, default=30000)

    return parser


def populate_vocab(words, vocab, cutoff):
    counter = Counter()

    for word in tqdm.tqdm(words, desc="counting words"):
        counter.update([word])

    topk = counter.most_common(cutoff)
    words = set(w for w, c in topk)

    for w in words:
        vocab.add(w)

    return vocab


def sent_words_iterator(sents):
    for sent in sents:
        for w in SplitWordIterator(sent):
            yield w


def multiple_directory_reader(input_dirs):
    for input_dir in input_dirs:
        for sent in DirectoryReader(input_dir):
            yield sent


def main():
    parser = create_parser()
    args = parser.parse_args()

    input_dirs = args.data_dir
    output_path = args.vocab_path
    reserved_words = args.reserved
    cutoff = args.cutoff

    reader = multiple_directory_reader(input_dirs)
    words = sent_words_iterator(reader)
    vocab = Vocabulary()

    print("Populating vocabulary...")
    vocab = populate_vocab(words, vocab, cutoff)
    for w in reserved_words:
        vocab.add(w)

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
