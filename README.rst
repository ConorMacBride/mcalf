===============================================
MCALF: Multi-Component Atmospheric Line Fitting
===============================================

|Travis Status| |PyPI Version| |GitHub License|

MCALF is an open-source Python package for accurately constraining velocity
information from spectral imaging observations using machine learning
techniques.

This software package provides a ‘toolkit’ that can be used to define a
spectral model optimised for a particular dataset.
A sample model is provided for an IBIS Ca II 8542 Å spectral imaging sunspot
dataset.
This dataset typically contains spectra with multiple atmospheric
components and this package supports the isolation of the individual
components such that velocity information can be constrained for each
component.
Using this sample model, as well as the separate base (template) model it is
built upon, a custom model can easily be built for a specific dataset.

Installation
------------

.. code:: bash

    $ pip install mcalf

We recommend installing this program inside a `virtual environment`_.
Alternatively, you can install Anaconda_ (or Miniconda_), and then install
the package using the above command inside an `new conda environment`_.

Testing
-------

First, install the package as usual, and then download the code
associated with your installed MCALF version.
Unzip the file and navigate to it in the terminal.
Run the following command (in the same directory as ``setup.py``) to test
your installation,

.. code:: bash

    $ python -m pytest

Make sure you are inside the virtual environment where it was installed.

Getting Started
---------------

Some examples are included `here <examples/>`_.
If you are interested in using this package in your research or you are
interested in contributing to it, please contact `Conor MacBride`_.

License
-------

MCALF is licensed under the terms of the BSD 2-Clause license.

.. |Travis Status| image:: https://img.shields.io/travis/com/ConorMacBride/mcalf
    :target: https://travis-ci.com/ConorMacBride/mcalf
    :alt: Travis
.. |PyPI Version| image:: https://img.shields.io/pypi/v/mcalf
    :target: https://pypi.python.org/pypi/mcalf
    :alt: PyPI
.. |GitHub License| image:: https://img.shields.io/github/license/ConorMacBride/mcalf
    :target: LICENSE.rst
    :alt: GitHub

.. _virtual environment: https://docs.python.org/3/tutorial/venv.html
.. _Anaconda: https://www.anaconda.com/products/individual#Downloads
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _new conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

.. _Conor MacBride: https://macbride.me/
