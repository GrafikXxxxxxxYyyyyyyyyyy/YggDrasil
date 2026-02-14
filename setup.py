"""Setup script for YggDrasil — ensures package discovery works with setuptools."""
from setuptools import setup, find_packages

# Explicit package discovery for reliable build (editable and wheel)
setup(
    packages=find_packages(where=".", include=("yggdrasil", "yggdrasil.*")),
    package_dir={"": "."},
)
