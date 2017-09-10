from setuptools import setup


__VERSION__ = "0.1.6"

setup(
    name="pytorch-text-utils",
    version=__VERSION__,
    description="PyTorch Text Processing Utilities",
    url="https://github.com/kaniblu/pytorch-text-utils",
    author="Kang Min Yoo",
    author_email="k@nib.lu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3"
    ],
    keywords="pytorch text nlp utility",
    packages=[
        "torchtextutils",
        "torchtextutils.data",
        "torchtextutils.data.preprocessor",
        "torchtextutils.iterator"
    ],
    install_requires=[
        "pyaap",
        "tqdm"
    ]
)