#!/usr/bin/env python

import os
from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext


class build_ext(build_ext):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypes)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            # Ensure that the extension ends in ".so"
            # Modified version of parent method
            from distutils.sysconfig import get_config_var
            ext_suffix = get_config_var('EXT_SUFFIX')
            expanded_suffix = ext_suffix.split('.')
            expanded_suffix[-1] = "so"
            ext_suffix = ".".join(expanded_suffix)
            ext_path = ext_name.split('.')
            return os.path.join(*ext_path) + ext_suffix
        return super().get_ext_filename(ext_name)


class CTypes(Extension):
    pass


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


is_nt = '_nt' if os.name == 'nt' else ''

setup(
    name="mcalf",
    version="0.1.1",
    package_dir={"": "src"},
    packages=find_packages("src"),

    python_requires=">=3.6",
    install_requires=["docutils>=0.3", "numpy>=1.17", "scipy>=1.3",
                      "pyyaml>=5.1", "pathos>=0.2.5", "scikit-learn>=0.21",
                      "matplotlib>=3.1", "astropy>=3.2", "pytest", "pytest-cov"],

    ext_modules=[CTypes("mcalf.profiles.ext_voigtlib", ["cextern/voigt{}.c".format(is_nt)])],
    cmdclass={'build_ext': build_ext},

    author="Conor MacBride",
    author_email="cmacbride01@qub.ac.uk",
    license="BSD 2-Clause",
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
