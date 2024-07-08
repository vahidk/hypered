import os
import sys
from distutils.core import setup

from setuptools import find_packages

# List of runtime dependencies required by this built package
install_requires = ["scikit-optimize", "flask"]
if sys.version_info <= (2, 7):
    install_requires += ["future", "typing"]

# read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setup(
    name="hypered",
    version="1.0.7",
    description="Simple hyper parameter tuning model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vahid Kazemi",
    author_email="vkazemi@gmail.com",
    url="https://github.com/vahidk/hypered",
    packages=find_packages(),
    license="MIT",
    install_requires=install_requires,
    test_suite='tests',
    package_data={
        'hypered': ['templates/*.html'],
    },
    entry_points={
        'console_scripts': [
            'hypered=hypered.cli:main',
            'hypered-dash=hypered.dash:main',
        ],
    },
)
