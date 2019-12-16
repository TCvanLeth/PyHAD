# -*- coding: utf-8 -*-
import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
        name="phad-tcvanleth",
        version="0.1.0",
        author="T.C. van Leth",
        author_email="tommy.vanleth@wur.nl",
        description="Physical Hierarchical Array Database",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages(),
        install_requires=[
                'numpy',
                'scipy',
                'dask',
                'h5py',
        ],
        classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
                "Operating System :: OS Independent",
        ],
)
