"""Setup script for YggDrasil.

Установка:
    pip install .
    pip install -e .
    pip install yggdrasil   # после публикации на PyPI

Метаданные (name, version, dependencies, entry points) берутся из pyproject.toml.
"""
from setuptools import setup, find_packages

setup(
    packages=find_packages(where=".", include=("yggdrasil", "yggdrasil.*")),
    package_dir={"": "."},
)
