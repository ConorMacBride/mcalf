import os
import pkg_resources
from pathlib import Path
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


__all__ = ['data_path_function', 'get_hash_library_name', 'figure_test']


def data_path_function(mod):
    # Returns a function that provides the filename and path of a data file for a particular module
    def data_path(*args, module=mod):
        return pkg_resources.resource_filename('mcalf', os.path.join('tests', module, 'data', *args))
    return data_path


def get_hash_library_name():
    """
    Generate the hash library name for this env.
    Notes
    -----
    Based on functions at https://github.com/sunpy/sunpy/blob/v2.0.7/sunpy/tests/helpers.py
    The SunPy Community et al. SunPy (Version v2.0.7). Zenodo. http://doi.org/10.5281/zenodo.4423217
    """
    ft2_version = f"{mpl.ft2font.__freetype_version__.replace('.', '')}"
    mpl_version = "dev" if "+" in mpl.__version__ else mpl.__version__.replace('.', '')
    return f"figure_hashes_mpl_{mpl_version}_ft_{ft2_version}.json"


def figure_test(test_function):
    """
    A decorator for a test that verifies the hash of the current figure or the
    returned figure, with the name of the test function as the hash identifier
    in the library. A PNG is also created in the 'result_image' directory,
    which is created on the current path.
    All such decorated tests are marked with `pytest.mark.figure` for convenient filtering.
    Examples
    --------
    @figure_test
    def test_simple_plot():
        plt.plot([0,1])
    Notes
    -----
    Based on functions at https://github.com/sunpy/sunpy/blob/v2.0.7/sunpy/tests/helpers.py
    The SunPy Community et al. SunPy (Version v2.0.7). Zenodo. http://doi.org/10.5281/zenodo.4423217
    """
    hash_library_name = get_hash_library_name()
    hash_library_file = Path(__file__).parent / hash_library_name

    @pytest.mark.mpl_image_compare(hash_library=hash_library_file,
                                   savefig_kwargs={'metadata': {'Software': None}},
                                   style='default')
    @wraps(test_function)
    def test_wrapper(pytestconfig, *args, **kwargs):
        ret = test_function(pytestconfig, *args, **kwargs)
        if ret is None:
            ret = plt.gcf()
        if pytestconfig.getoption('--mpl', default=None) is None:
            print("close ", end='')
            plt.close(fig=ret)
        else:
            print("return ", end='')
            return ret

    return test_wrapper
