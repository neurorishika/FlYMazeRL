from distutils.core import setup
from setuptools import find_packages
import os

# get project description from README.md (if it exists)
current_directory = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(current_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="flymazerl",  # Project name
    version="1.0",  # Version number
    description="flymazerl is a reinforcement learning framework for a Y-Maze assay designed for fruitflies in the Turner Lab at Janelia Research Campus.",  # Project description (short)
    long_description=long_description,  # Project description (long)
    long_description_content_type="text/markdown",  # Project description (long) type
    author="Rishika Mohanta",  # Author
    author_email="neurorishika@gmail.com",  # Author email
    url="https://github.com/neurorishika/FlYMazeRL",  # Project URL
    license="BSD(3-clause) License",  # License
    keywords=["reinforcement learning", "drosophila", "y-maze", "2AFC"],  # keywords
    install_requires=["numpy", "matplotlib", "pandas", "seaborn", "pymc3", "arviz", "torch", "tqdm",],  # Dependencies
    packages=["flymazerl"],  # Packages to be installed
)
