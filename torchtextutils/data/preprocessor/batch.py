import torch


class BatchPreprocessor(object):
    def __init__(self, vocab, add_bos=True, add_eos=True):
        self.vocab = vocab
        self.unk_idx = vocab[vocab.unk]
        self.pad_idx = vocab[vocab.pad]
        self.bos_idx = vocab[vocab.bos]
        self.eos_idx = vocab[vocab.eos]
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, batch):
        batch = [[self.vocab.f2i[w]
                                   if w in self.vocab.f2i else self.unk_idx
                  for w in words] for words in batch]

        if self.add_bos:
            batch = [[self.bos_idx] + words for words in batch]

        if self.add_eos:
            batch = [words + [self.eos_idx] for words in batch]

        lens = [len(s) for s in batch]
        max_len = max(lens)
        batch = [s + [self.pad_idx] * (max_len - len(s)) for s in batch]

        batch = torch.LongTensor(batch)
        lens = torch.LongTensor(lens)

        return batch, lens