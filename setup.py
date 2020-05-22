#!/usr/bin/env python

import os.path
from setuptools import setup, find_packages, Extension


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


setup(
    name="mcalf",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),

    python_requires=">=3.6",
    install_requires=["docutils>=0.3", "numpy>=1.17", "scipy>=1.3",
                      "pyyaml>=5.1", "pathos>=0.2.5", "scikit-learn>=0.21",
                      "matplotlib>=3.1", "astropy>=3.2", "pytest"],

    ext_modules=[Extension("mcalf.profiles.ext_voigtlib", ["cextern/voigt.c"])],

    author="Conor MacBride",
    author_email="cmacbride01@qub.ac.uk",
    licence="BSD 2-Clause",
    description="MCALF: Multi-Component Atmospheric Line Fitting",
    keywords="spectrum spectra fitting absorption emission voigt",
    url="https://github.com/ConorMacBride/mcalf/",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    long_description=read('README.rst')
)
