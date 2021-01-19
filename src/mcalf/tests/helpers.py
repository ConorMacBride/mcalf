import os
import pkg_resources


__all__ = ['data_path_function']


def data_path_function(mod):
    # Returns a function that provides the filename and path of a data file for a particular module
    def data_path(*args, module=mod):
        return pkg_resources.resource_filename('mcalf', os.path.join('tests', module, 'data', *args))
    return data_path
