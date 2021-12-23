===============================================
MCALF: Multi-Component Atmospheric Line Fitting
===============================================

|Azure Pipelines Status| |Codecov| |PyPI Version| |Zenodo DOI| |Docs Status| |GitHub License|

MCALF is an open-source Python package for accurately constraining velocity
information from spectral imaging observations using machine learning
techniques.

This software package is intended to be used by solar physicists trying
to extract line-of-sight (LOS) Doppler velocity information from
spectral imaging observations (Stokes I measurements) of the Sun.
A ‘toolkit’ is provided that can be used to define a spectral model
optimised for a particular dataset.

This package is particularly suited for extracting velocity information
from spectral imaging observations where the individual spectra can
contain multiple spectral components.
Such multiple components are typically present when active solar phenomenon
occur within an isolated region of the solar disk.
Spectra within such a region will often have a large emission component
superimposed on top of the underlying absorption spectral profile from the
quiescent solar atmosphere.

A sample model is provided for an IBIS Ca II 8542 Å spectral imaging sunspot
dataset.
This dataset typically contains spectra with multiple atmospheric
components and this package supports the isolation of the individual
components such that velocity information can be constrained for each
component.
Using this sample model, as well as the separate base (template) model it is
built upon, a custom model can easily be built for a specific dataset.

The custom model can be designed to take into account the spectral shape of
each particular spectrum in the dataset.
By training a neural network classifier using a sample of spectra from the
dataset labelled with their spectral shapes, the spectral shape of any
spectrum in the dataset can be found.
The fitting algorithm can then be adjusted for each spectrum based on
the particular spectral shape the neural network assigned it.

This package is designed to run in parallel over large data cubes, as well
as in serial.
As each spectrum is processed in isolation, this package scales very well
across many processor cores.
Numerous functions are provided to plot the results in a clearly.
The MCALF API also contains many useful functions which have the potential
of being integrated into other Python packages.

Installation
------------

For easier package management we recommend using `Miniconda`_ (or `Anaconda`_)
and creating a `new conda environment`_ to install MCALF inside.
To install MCALF using `Miniconda`_, run the following commands in your
system's command prompt, or if you are using Windows, in the
'Anaconda Prompt':

.. code:: bash

    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict
    $ conda install mcalf

MCALF is updated to the latest version by running:

.. code:: bash

    $ conda update mcalf

Alternatively, you can install MCALF using ``pip``:

.. code:: bash

    $ pip install mcalf

Testing
-------

A test suite is included with the package. The package is tested on
multiple platforms, however you may wish to run the tests on your
system also. More details on running our tox/pytest test suite are
available in our `documentation`_.

Getting Started
---------------

Documentation is `available here <https://mcalf.macbride.me/>`_.
Some examples are included `here <examples/>`_.
If you are interested in using this package in your research and would
like advice on how to use this package, please contact `Conor MacBride`_.

Contributing
------------

|Contributor Covenant|

If you find this package useful and have time to make it even better,
you are very welcome to contribute to this package, regardless of how much
prior experience you have.
Types of ways you can contribute include, expanding the documentation with
more use cases and examples, reporting bugs through the GitHub issue tracker,
reviewing pull requests and the existing code, fixing bugs and implementing new
features in the code.

You are encouraged to submit any `bug reports`_ and `pull requests`_ directly
to the `GitHub repository`_.
If you have any questions regarding contributing to this package please
contact `Conor MacBride`_.

Please note that this project is released with a Contributor Code of Conduct.
By participating in this project you agree to abide by its terms.

Citation
--------

If you have used this package in work that leads to a publication, we would
be very grateful if you could acknowledge your use of this package in the
main text of the publication.
Please cite the following publications,

    MacBride CD, Jess DB. 2021
    MCALF: Multi-Component Atmospheric Line Fitting.
    *Journal of Open Source Software*. **6(61)**, 3265.
    (`doi:10.21105/joss.03265 <https://doi.org/10.21105/joss.03265>`_)

..

    MacBride CD, Jess DB, Grant SDT, Khomenko E, Keys PH, Stangalini M. 2020
    Accurately constraining velocity information from spectral imaging
    observations using machine learning techniques.
    *Philosophical Transactions of the Royal Society A*. **379**, 2190.
    (`doi:10.1098/rsta.2020.0171 <https://doi.org/10.1098/rsta.2020.0171>`_)

Please also cite the `Zenodo DOI`_ for the package version you used.
Please also consider integrating your code and examples into the package.

License
-------

MCALF is licensed under the terms of the BSD 2-Clause license.

.. |Azure Pipelines Status| image:: https://dev.azure.com/ConorMacBride/mcalf/_apis/build/status/ConorMacBride.mcalf?repoName=ConorMacBride%2Fmcalf&branchName=main
    :target: https://dev.azure.com/ConorMacBride/mcalf/_build/latest?definitionId=5&repoName=ConorMacBride%2Fmcalf&branchName=main
    :alt: Azure Pipelines
.. |Codecov| image:: https://codecov.io/gh/ConorMacBride/mcalf/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/ConorMacBride/mcalf
    :alt: Codecov
.. |PyPI Version| image:: https://img.shields.io/pypi/v/mcalf
    :target: https://pypi.python.org/pypi/mcalf
    :alt: PyPI
.. |Zenodo DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3924527.svg
    :target: https://doi.org/10.5281/zenodo.3924527
    :alt: DOI
.. |Docs Status| image:: https://readthedocs.org/projects/mcalf/badge/?version=latest&style=flat
    :target: https://mcalf.macbride.me/
    :alt: Documentation
.. |GitHub License| image:: https://img.shields.io/github/license/ConorMacBride/mcalf
    :target: LICENSE.rst
    :alt: License
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg
    :target: CODE_OF_CONDUCT.rst
    :alt: Code of Conduct

.. _Anaconda: https://www.anaconda.com/products/individual#Downloads
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _new conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _documentation: https://mcalf.macbride.me/en/latest/guide/index.html#testing

.. _Conor MacBride: https://macbride.me/

.. _bug reports: https://github.com/ConorMacBride/mcalf/issues
.. _pull requests: https://github.com/ConorMacBride/mcalf/pulls
.. _GitHub repository: https://github.com/ConorMacBride/mcalf

.. _Zenodo DOI: https://doi.org/10.5281/zenodo.3924527
