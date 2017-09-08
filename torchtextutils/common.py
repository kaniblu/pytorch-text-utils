import os


class Generator(object):
    def __iter__(self):
        return self.generate()

    def generate(self):
        raise NotImplementedError()


def ensure_dir_exists(path):
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)