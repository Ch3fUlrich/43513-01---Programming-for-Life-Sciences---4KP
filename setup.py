from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    # Ensure the path is relative to the setup.py directory
    here = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(here, filename)
    try:
        with open(filepath) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="GillespieSimulation",
    version="0.1.0",
    author="Sergej Maul",
    author_email="maulser@gmail.com",
    description="A package for simulating stochastic chemical reactions using the Gillespie algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ch3fUlrich/43513-01---Programming-for-Life-Sciences---4KP",
    packages=find_packages(include=["Project", "Project.*"]),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": parse_requirements("requirements_dev.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
