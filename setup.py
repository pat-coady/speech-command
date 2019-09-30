"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kws',
    version='1.0.0',
    description='Keyword detection.',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3.6', install_requires=['numpy', 'tensorflow']
)
