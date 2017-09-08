import time
import random

from torchtextutils import populate_vocab
from torchtextutils import SplitWordIterator
from torchtextutils import Vocabulary
from torchtextutils import BatchPreprocessor
from torchtextutils import create_generator_ae


def generate_dataset(n=None):
    def gen():
        length = random.randint(3, 15)
        words = [str(random.randint(0, 255)) for i in range(length)]
        sent = " ".join(words)
        return sent

    if n is None:
        while True:
            yield gen()
    else:
        for i in range(n):
            yield gen()


def iterate_words(sents):
    for sent in sents:
        for w in SplitWordIterator(sent):
            yield w


def test_all():
    vocab = Vocabulary(bos="<bos>", eos="<eos>",
                       unk="<unk>", pad="<pad>")
    samples = generate_dataset(1000)
    populate_vocab(iterate_words(samples), vocab, 100)

    preprocessor = BatchPreprocessor(vocab)
    data_gen = create_generator_ae(
        sents=generate_dataset(),
        batch_size=4,
        preprocessor=preprocessor
    )

    for i, (x, lens) in enumerate(data_gen):

        if i > 1000:
            break

        assert x.shape[0] <= 4 and len(lens) <= 4