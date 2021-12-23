import os.path

try:
    from setuptools_scm import get_version
    version = get_version(root=os.path.join('..', '..'), relative_to=__file__)
except ImportError:
    from ._version import version  # noqa: F401
