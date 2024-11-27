from setuptools import setup, find_packages

setup(
    name="gillespie_simulation",  # Package name
    version="0.1.0",  # Initial version
    author="Sergej Maul",
    author_email="maulser@gmail.com",
    description="A package for simulating stochastic chemical reactions using the Gillespie algorithm.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ch3fUlrich/43513-01---Programming-for-Life-Sciences---4KP",
    packages=find_packages(include=["Project", "Project.*"]),
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={
        "dev": open("requirements_dev.txt").read().splitlines(),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

...................