#!/usr/bin/env python

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class build_ext(build_ext):

    def get_export_symbols(self, ext):
        if isinstance(ext, CTypes):
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if isinstance(self.ext_map[ext_name], CTypes):
            # Ensure that the extension ends in ".so"
            # Modified version of parent method
            from setuptools.command.build_ext import get_config_var
            ext_suffix = get_config_var('EXT_SUFFIX')
            expanded_suffix = ext_suffix.split('.')
            expanded_suffix[-1] = "so"
            ext_suffix = ".".join(expanded_suffix)
            ext_path = ext_name.split('.')
            return os.path.join(*ext_path) + ext_suffix
        return super().get_ext_filename(ext_name)


class CTypes(Extension):
    pass


is_nt = '_nt' if os.name == 'nt' else ''

setup(
    use_scm_version={'write_to': os.path.join('src', 'mcalf', '_version.py')},
    setup_requires=['setuptools_scm'],
    ext_modules=[CTypes("mcalf.profiles.ext_voigtlib", ["cextern/voigt{}.c".format(is_nt)])],
    cmdclass={'build_ext': build_ext},
)
