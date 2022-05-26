import setuptools

from dee import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="dee",
    version=__version__,
    author="Tong Zhu",
    author_email="tzhu1997@outlook.com",
    description="Document Event Extraction Toolkit",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(".", exclude=("*tests*",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.5.1",
        "pytorch-mcrf>=0.0.3",
        "gpu-watchmen>=0.3.8",
        "loguru>=0.5.3",
        "matplotlib>=3.3.0",
        "numpy>=1.21",
        "transformers>=4.9.1",
        "dgl>=0.6.1",
        "tqdm>=4.53.0",
        "networkx>=2.4",
        "tensorboard>=2.4.1",
    ],
    extras_require={
        "dev": ["pytest", "coverage", "black", "flake8", "isort", "sphinx"]
    },
    package_data={},
)
