====================================
example1: Basic usage of the package
====================================

``FittingIBIS.ipynb``
---------------------

* `View code <FittingIBIS.ipynb>`_
* :download:`Download FittingIBIS.ipynb <FittingIBIS.ipynb>`

This file is an IPython Notebook containing examples of how to use the package
to accomplish typical tasks.

``FittingIBIS.pro``
-------------------

* :download:`Download FittingIBIS.pro <FittingIBIS.pro>`

This file is similar to ``FittingIBIS.ipynb`` file, except it written is IDL.
It is not recommended to use the IDL wrapper in production, just use it to
explore the code if you are familiar with IDL and not Python.
If you wish to use this package, please use the Python implementation.
IDL is not fully supported in the current version of the code for reasons
such as, the Python tuple datatype cannot be passed from IDL to Python,
resulting in certain function calls not being possible.

``config.yml``
--------------

* :download:`Download config.yml <config.yml>`

This is an example configuration file containing default parameters.
This can be easier than setting the parameters in the code.
The file follows the YAML_ format.

.. _YAML: https://pyyaml.org/wiki/PyYAMLDocumentation
