from setuptools import setup, find_packages


def parse_requirements(file):
    with open(file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="GillespieSimulation",
    version="0.1.0",
    author="Sergej Maul",
    author_email="maulser@gmail.com",
    description="A package for simulating stochastic chemical reactions using the Gillespie algorithm.",
    long_description=open("README.md").read(),
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
    python_requires=">=3.10",
)
