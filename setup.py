import os
from setuptools import find_packages, setup

__version__ = "0.0.1"

setup(
    name="acyrlkg",
    packages=[package for package in find_packages() if package.startswith("acyrlkg")],
    package_data={"acyrlkg": ["py.typed"]},
    install_requires=[],
    description=".",
    version=__version__,
    python_requires=">=3.8",
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
