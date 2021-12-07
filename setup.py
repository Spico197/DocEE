import setuptools


from dee import __version__


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='dee',
    version=__version__,
    author="Tong Zhu",
    author_email="tzhu1997@outlook.com",
    description="Document Event Extraction Toolkit",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages('.', exclude=('*tests*',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.7',
    install_requires=[],
    package_data={},
    include_package_data=False,
)
